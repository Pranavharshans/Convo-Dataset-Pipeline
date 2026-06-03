from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from convo_ds.audio import wav_duration_sec
from convo_ds.config import PipelineConfig
from convo_ds.jsonl import read_jsonl
from convo_ds.schemas import ConversationMetadata, DialogueScript, TrainingSequence
from convo_ds.scripts import validate_script_for_bucket


@dataclass
class ValidationReport:
    checked: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "checked": self.checked, "errors": self.errors}


def validate_scripts(config: PipelineConfig, scripts_path: Path) -> ValidationReport:
    report = ValidationReport()
    buckets = {bucket.name: bucket for bucket in config.buckets}
    seen: set[str] = set()
    for script in read_jsonl(scripts_path, DialogueScript):
        report.checked += 1
        if script.conversation_id in seen:
            report.add_error(f"duplicate conversation_id: {script.conversation_id}")
        seen.add(script.conversation_id)
        bucket = buckets.get(script.category)
        if bucket is None:
            report.add_error(f"{script.conversation_id}: unknown category {script.category}")
            continue
        try:
            validate_script_for_bucket(script, bucket)
        except ValueError as exc:
            report.add_error(f"{script.conversation_id}: {exc}")
    if report.checked == 0:
        report.add_error(f"no scripts found at {scripts_path}")
    return report


def validate_stage_dir(config: PipelineConfig, stage_dir: Path, require_overlaps: bool = False) -> ValidationReport:
    report = ValidationReport()
    buckets = {bucket.name: bucket for bucket in config.buckets}
    conversation_dir = stage_dir / "conversations"
    for json_path in sorted(conversation_dir.glob("*.json")):
        report.checked += 1
        try:
            metadata = ConversationMetadata.model_validate_json(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.add_error(f"{json_path.name}: invalid metadata: {exc}")
            continue
        bucket = buckets.get(metadata.category)
        if bucket is None:
            report.add_error(f"{metadata.conversation_id}: unknown category {metadata.category}")
        elif not bucket.duration_min_sec <= metadata.duration_sec <= bucket.duration_max_sec:
            report.add_error(f"{metadata.conversation_id}: duration {metadata.duration_sec:.2f}s outside bucket {metadata.category}")
        if require_overlaps and not metadata.overlaps:
            report.add_error(f"{metadata.conversation_id}: missing overlaps")
        if metadata.mixed_audio_path:
            audio_path = stage_dir / metadata.mixed_audio_path
            if not audio_path.exists():
                report.add_error(f"{metadata.conversation_id}: missing mixed audio {metadata.mixed_audio_path}")
            else:
                measured = wav_duration_sec(audio_path)
                if abs(measured - metadata.duration_sec) > 0.1:
                    report.add_error(f"{metadata.conversation_id}: metadata duration does not match WAV")
        previous_end = -1
        for index, turn in enumerate(metadata.turns):
            if not require_overlaps and turn.start_ms < previous_end:
                report.add_error(f"{metadata.conversation_id}: turn {index} starts before previous turn ends")
            previous_end = turn.end_ms
            turn_path = stage_dir / turn.audio_path
            if not turn_path.exists():
                report.add_error(f"{metadata.conversation_id}: missing turn audio {turn.audio_path}")
    if report.checked == 0:
        report.add_error(f"no conversation metadata found under {conversation_dir}")
    return report


def validate_shards(shards_path: Path) -> ValidationReport:
    report = ValidationReport()
    for sequence in read_jsonl(shards_path, TrainingSequence):
        report.checked += 1
        if not sequence.tokens:
            report.add_error(f"{sequence.conversation_id}: empty token sequence")
        if sum(sequence.loss_mask) == 0:
            report.add_error(f"{sequence.conversation_id}: loss mask has no trainable positions")
    if report.checked == 0:
        report.add_error(f"no shard records found at {shards_path}")
    return report
