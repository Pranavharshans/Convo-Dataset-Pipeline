from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import numpy as np

from convo_ds.audio import concat_with_gaps, silence, wav_duration_sec, write_wav
from convo_ds.config import ConversationBucket, PipelineConfig
from convo_ds.jsonl import read_jsonl, write_json
from convo_ds.schemas import AudioTurn, ConversationMetadata, DialogueScript, Speaker, as_posix_relative


class SpeechSynthesizer(Protocol):
    sample_rate: int

    def synthesize_turn(self, speaker: Speaker, text: str) -> np.ndarray:
        ...


class MockSynthesizer:
    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    def synthesize_turn(self, speaker: Speaker, text: str) -> np.ndarray:
        words = max(1, len(text.split()))
        duration = max(0.6, words / 2.8)
        frequency = 220.0 if speaker == Speaker.s0 else 330.0
        samples = np.arange(int(duration * self.sample_rate), dtype=np.float32)
        envelope = np.minimum(1.0, np.linspace(0, 1, samples.size, dtype=np.float32) * 10)
        tone = 0.08 * np.sin(2 * math.pi * frequency * samples / self.sample_rate)
        return (tone * envelope).astype(np.float32)


class DiaSynthesizer:
    def __init__(self, model_id: str, sample_rate: int, precision: str = "bfloat16", compile_model: bool = True) -> None:
        try:
            from dia.model import Dia
        except Exception as exc:  # pragma: no cover - GPU dependency path
            raise RuntimeError("Dia is not installed. Install Dia on a CUDA machine or use --mock.") from exc
        self.sample_rate = sample_rate
        self.model = Dia.from_pretrained(model_id)
        self.precision = precision
        self.compile_model = compile_model

    def synthesize_turn(self, speaker: Speaker, text: str) -> np.ndarray:
        tag = "[S1]" if speaker == Speaker.s0 else "[S2]"
        output = self.model.generate(f"{tag} {text}")
        audio = output.audio if hasattr(output, "audio") else output
        return np.asarray(audio, dtype=np.float32)


def synthesize_stage3(
    config: PipelineConfig,
    scripts_path: Path,
    output_dir: Path,
    limit: int | None = None,
    mock: bool = False,
) -> dict:
    synthesizer: SpeechSynthesizer
    if mock:
        synthesizer = MockSynthesizer(sample_rate=config.dia.output_sample_rate)
    else:
        synthesizer = DiaSynthesizer(
            model_id=config.dia.model_id,
            sample_rate=config.dia.output_sample_rate,
            precision=config.dia.precision,
            compile_model=config.dia.compile,
        )

    bucket_map = {bucket.name: bucket for bucket in config.buckets}
    scripts = list(read_jsonl(scripts_path, DialogueScript))
    if limit is not None:
        scripts = scripts[:limit]

    written = 0
    skipped = 0
    for script in scripts:
        bucket = bucket_map.get(script.category)
        if bucket is None:
            skipped += 1
            continue
        try:
            metadata = _synthesize_one(config, script, bucket, output_dir, synthesizer)
        except ValueError:
            skipped += 1
            continue
        json_path = output_dir / "conversations" / f"{script.conversation_id}.json"
        write_json(json_path, metadata.model_dump())
        written += 1
    return {"written": written, "skipped": skipped}


def _synthesize_one(
    config: PipelineConfig,
    script: DialogueScript,
    bucket: ConversationBucket,
    output_dir: Path,
    synthesizer: SpeechSynthesizer,
) -> ConversationMetadata:
    last_error: ValueError | None = None
    for _attempt in range(config.dia.duration_regen_limit + 1):
        metadata = _render_conversation(script, output_dir, synthesizer)
        # Dia duration can drift from text estimates; measured audio is the source of truth.
        if bucket.duration_min_sec <= metadata.duration_sec <= bucket.duration_max_sec:
            return metadata
        last_error = ValueError(
            f"{script.conversation_id} duration {metadata.duration_sec:.2f}s outside "
            f"{bucket.duration_min_sec}-{bucket.duration_max_sec}s"
        )
    raise last_error or ValueError("stage3 synthesis failed")


def _render_conversation(script: DialogueScript, output_dir: Path, synthesizer: SpeechSynthesizer) -> ConversationMetadata:
    turn_audio_dir = output_dir / "audio" / script.conversation_id
    turn_gap_sec = 0.1
    chunks: list[np.ndarray] = []
    turns: list[AudioTurn] = []
    cursor_ms = 0

    for index, turn in enumerate(script.turns):
        audio = synthesizer.synthesize_turn(turn.speaker, turn.text)
        turn_path = turn_audio_dir / f"turn_{index:03d}.wav"
        write_wav(turn_path, audio, synthesizer.sample_rate)
        duration_ms = int(round(wav_duration_sec(turn_path) * 1000))
        start_ms = cursor_ms
        end_ms = start_ms + duration_ms
        turns.append(
            AudioTurn(
                speaker=0 if turn.speaker == Speaker.s0 else 1,
                text=turn.text,
                audio_path=as_posix_relative(turn_path, output_dir),
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
        chunks.append(audio)
        cursor_ms = end_ms + int(turn_gap_sec * 1000)

    mixed = concat_with_gaps(chunks, turn_gap_sec, synthesizer.sample_rate)
    mixed_path = output_dir / "audio" / f"{script.conversation_id}.wav"
    write_wav(mixed_path, mixed, synthesizer.sample_rate)
    duration_sec = wav_duration_sec(mixed_path)
    return ConversationMetadata(
        conversation_id=script.conversation_id,
        category=script.category,
        duration_sec=duration_sec,
        script_path=None,
        mixed_audio_path=as_posix_relative(mixed_path, output_dir),
        turns=turns,
    )
