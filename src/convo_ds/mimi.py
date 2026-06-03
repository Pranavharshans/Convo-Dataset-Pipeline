from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from convo_ds.audio import read_wav
from convo_ds.config import PipelineConfig
from convo_ds.jsonl import append_jsonl
from convo_ds.schemas import ConversationMetadata, MimiConversationTokens, TrainingSequence


class MimiTokenizer(Protocol):
    def encode(self, audio_path: Path) -> list[list[int]]:
        ...


class MockMimiTokenizer:
    def __init__(self, codebooks: int = 16, token_count: int = 2048) -> None:
        self.codebooks = codebooks
        self.token_count = token_count

    def encode(self, audio_path: Path) -> list[list[int]]:
        audio, sample_rate = read_wav(audio_path)
        frame_count = max(1, int((audio.size / sample_rate) * 12.5))
        seed = int(hashlib.sha1(audio_path.as_posix().encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.token_count, size=(self.codebooks, frame_count)).astype(int).tolist()


class TransformersMimiTokenizer:
    def __init__(self, model_id: str) -> None:
        try:
            import torch
            from transformers import AutoFeatureExtractor, MimiModel
        except Exception as exc:  # pragma: no cover - GPU dependency path
            raise RuntimeError("Mimi dependencies are not installed. Install GPU extras or use --mock.") from exc
        self.torch = torch
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = MimiModel.from_pretrained(model_id)
        self.model.eval()

    def encode(self, audio_path: Path) -> list[list[int]]:
        audio, sample_rate = read_wav(audio_path)
        inputs = self.feature_extractor(raw_audio=audio, sampling_rate=sample_rate, return_tensors="pt")
        with self.torch.no_grad():
            encoded = self.model.encode(inputs["input_values"])
        codes = encoded.audio_codes if hasattr(encoded, "audio_codes") else encoded[0]
        codes_np = codes.squeeze(0).detach().cpu().numpy()
        if codes_np.ndim == 3:
            codes_np = codes_np.reshape(codes_np.shape[0] * codes_np.shape[1], codes_np.shape[2])
        return codes_np.astype(int).tolist()


def tokenize_mimi(
    config: PipelineConfig,
    stage_dir: Path,
    output_path: Path,
    limit: int | None = None,
    mock: bool = False,
) -> dict:
    tokenizer: MimiTokenizer
    if mock:
        tokenizer = MockMimiTokenizer(token_count=config.mimi.token_count)
    else:
        tokenizer = TransformersMimiTokenizer(config.mimi.model_id)

    metadata_paths = sorted((stage_dir / "conversations").glob("*.json"))
    if limit is not None:
        metadata_paths = metadata_paths[:limit]
    records: list[MimiConversationTokens] = []
    for path in metadata_paths:
        metadata = ConversationMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        if not metadata.mixed_audio_path:
            continue
        codes_full = tokenizer.encode(stage_dir / metadata.mixed_audio_path)
        training_count = min(config.mimi.training_codebooks, len(codes_full))
        # Store all codebooks for auditability, but keep training export pinned to the first 8.
        records.append(
            MimiConversationTokens(
                conversation_id=metadata.conversation_id,
                source_json=path.as_posix(),
                codebook_count=len(codes_full),
                training_codebooks=training_count,
                codes_full=codes_full,
                codes_training=codes_full[:training_count],
            )
        )
    append_jsonl(output_path, records)
    return {"written": len(records)}


def assemble_training_shards(tokens_path: Path, output_path: Path) -> dict:
    from convo_ds.jsonl import read_jsonl

    records: list[TrainingSequence] = []
    for token_record in read_jsonl(tokens_path, MimiConversationTokens):
        metadata = ConversationMetadata.model_validate_json(Path(token_record.source_json).read_text(encoding="utf-8"))
        tokens: list[str | int] = []
        loss_mask: list[int] = []
        for turn in metadata.turns:
            text_marker = "<|s0_text|>" if turn.speaker == 0 else "<|s1_text|>"
            audio_marker = "<|s0_audio|>" if turn.speaker == 0 else "<|s1_audio|>"
            _append(tokens, loss_mask, text_marker, 0)
            for word in turn.text.split():
                _append(tokens, loss_mask, word, 0)
            _append(tokens, loss_mask, audio_marker, 0)
        for code in _flatten_frame_major(token_record.codes_training):
            _append(tokens, loss_mask, f"<|mimi_{code}|>", 1)
        records.append(
            TrainingSequence(
                conversation_id=token_record.conversation_id,
                source_json=token_record.source_json,
                tokens=tokens,
                loss_mask=loss_mask,
            )
        )
    append_jsonl(output_path, records)
    return {"written": len(records)}


def _flatten_frame_major(codebooks: list[list[int]]) -> list[int]:
    if not codebooks:
        return []
    frame_count = min(len(codebook) for codebook in codebooks)
    flattened: list[int] = []
    for frame in range(frame_count):
        for codebook in codebooks:
            flattened.append(codebook[frame])
    return flattened


def _append(tokens: list[str | int], mask: list[int], token: str | int, loss: int) -> None:
    tokens.append(token)
    mask.append(loss)
