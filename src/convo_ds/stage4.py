from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np

from convo_ds.audio import read_wav, wav_duration_sec, write_wav
from convo_ds.config import PipelineConfig
from convo_ds.jsonl import write_json
from convo_ds.schemas import AudioTurn, ConversationMetadata, OverlapMarker, as_posix_relative


OVERLAP_TYPES = ("backchannel", "interruption", "simultaneous_start")


def inject_overlaps(
    config: PipelineConfig,
    stage3_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    seed: int = 13,
) -> dict:
    rng = random.Random(seed)
    metadata_paths = sorted((stage3_dir / "conversations").glob("*.json"))
    if limit is not None:
        metadata_paths = metadata_paths[:limit]
    written = 0
    skipped = 0
    for path in metadata_paths:
        metadata = ConversationMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        if len(metadata.turns) < 3:
            skipped += 1
            continue
        try:
            overlapped = _inject_one(config, stage3_dir, output_dir, metadata, rng)
        except ValueError:
            skipped += 1
            continue
        write_json(output_dir / "conversations" / f"{overlapped.conversation_id}.json", overlapped.model_dump())
        written += 1
    return {"written": written, "skipped": skipped}


def _inject_one(
    config: PipelineConfig,
    stage3_dir: Path,
    output_dir: Path,
    metadata: ConversationMetadata,
    rng: random.Random,
) -> ConversationMetadata:
    overlap_type = _choose_overlap_type(config, rng)
    turn_b_index = _choose_turn_to_shift(metadata, rng)
    turn_a_index = max(0, turn_b_index - 1)
    shifted_turns = _copy_turns(stage3_dir, output_dir, metadata)

    turn_a = shifted_turns[turn_a_index]
    turn_b = shifted_turns[turn_b_index]
    original_b_duration = turn_b.end_ms - turn_b.start_ms
    overlap_ms = _overlap_duration_ms(overlap_type, original_b_duration)
    if overlap_type == "simultaneous_start":
        new_start = turn_a.start_ms
    elif overlap_type == "interruption":
        new_start = max(turn_a.start_ms, turn_a.end_ms - overlap_ms)
    else:
        new_start = max(turn_a.start_ms + 250, turn_a.end_ms - overlap_ms)
    turn_b.start_ms = new_start
    turn_b.end_ms = new_start + original_b_duration

    overlap_start = max(turn_a.start_ms, turn_b.start_ms)
    overlap_end = min(turn_a.end_ms, turn_b.end_ms)
    if overlap_end <= overlap_start:
        raise ValueError("selected shift did not create an overlap")

    mixed_path = output_dir / "audio" / f"{metadata.conversation_id}.wav"
    _mix_shifted_timeline(stage3_dir, output_dir, shifted_turns, mixed_path, config.overlap.overlap_attenuation, metadata.duration_sec)

    return ConversationMetadata(
        conversation_id=metadata.conversation_id,
        category=metadata.category,
        language=metadata.language,
        duration_sec=wav_duration_sec(mixed_path),
        script_path=metadata.script_path,
        mixed_audio_path=as_posix_relative(mixed_path, output_dir),
        turns=shifted_turns,
        overlaps=[
            OverlapMarker(
                turn_a=turn_a_index,
                turn_b=turn_b_index,
                overlap_start_ms=overlap_start,
                overlap_end_ms=overlap_end,
                type=overlap_type,
            )
        ],
    )


def _choose_overlap_type(config: PipelineConfig, rng: random.Random) -> str:
    draw = rng.random()
    if draw < config.overlap.backchannel_ratio:
        return "backchannel"
    if draw < config.overlap.backchannel_ratio + config.overlap.interruption_ratio:
        return "interruption"
    return "simultaneous_start"


def _choose_turn_to_shift(metadata: ConversationMetadata, rng: random.Random) -> int:
    candidates = [index for index in range(1, len(metadata.turns)) if metadata.turns[index].speaker != metadata.turns[index - 1].speaker]
    if not candidates:
        raise ValueError("no alternating turn pair available")
    return rng.choice(candidates)


def _copy_turns(stage3_dir: Path, output_dir: Path, metadata: ConversationMetadata) -> list[AudioTurn]:
    copied: list[AudioTurn] = []
    for index, turn in enumerate(metadata.turns):
        source = stage3_dir / turn.audio_path
        target = output_dir / "audio" / metadata.conversation_id / f"turn_{index:03d}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(
            AudioTurn(
                speaker=turn.speaker,
                text=turn.text,
                audio_path=as_posix_relative(target, output_dir),
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
            )
        )
    return copied


def _overlap_duration_ms(overlap_type: str, turn_duration_ms: int) -> int:
    if overlap_type == "backchannel":
        return min(max(300, turn_duration_ms // 2), 900)
    if overlap_type == "interruption":
        return min(max(500, turn_duration_ms // 2), 1400)
    return min(max(350, turn_duration_ms // 3), 1000)


def _mix_shifted_timeline(
    stage3_dir: Path,
    output_dir: Path,
    turns: list[AudioTurn],
    mixed_path: Path,
    attenuation: float,
    minimum_duration_sec: float,
) -> None:
    sample_rate: int | None = None
    loaded: list[tuple[AudioTurn, np.ndarray]] = []
    for turn in turns:
        audio, sr = read_wav(output_dir / turn.audio_path)
        sample_rate = sample_rate or sr
        if sr != sample_rate:
            raise ValueError("all turn audio must have the same sample rate")
        loaded.append((turn, audio))

    assert sample_rate is not None
    total_samples = max(int(minimum_duration_sec * sample_rate), max(int(turn.end_ms * sample_rate / 1000) for turn in turns))
    mix = np.zeros(total_samples, dtype=np.float32)
    occupied = np.zeros(total_samples, dtype=bool)
    for turn, audio in loaded:
        start = int(turn.start_ms * sample_rate / 1000)
        end = min(total_samples, start + audio.size)
        if end <= start:
            continue
        segment = audio[: end - start]
        existing = occupied[start:end]
        # Attenuate both sides in overlapped windows to prevent clipping after summing speakers.
        mix[start:end][existing] *= attenuation
        mix[start:end] += np.where(existing, segment * attenuation, segment)
        occupied[start:end] = True
    write_wav(mixed_path, np.clip(mix, -1.0, 1.0), sample_rate)
