from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class Speaker(str, Enum):
    s0 = "S0"
    s1 = "S1"


class ScriptTurn(BaseModel):
    speaker: Speaker
    text: str

    @field_validator("text")
    @classmethod
    def text_is_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("turn text must not be empty")
        return stripped


class DialogueScript(BaseModel):
    conversation_id: str
    category: str
    topic: str
    turns: list[ScriptTurn]
    estimated_duration_sec: float | None = None

    @model_validator(mode="after")
    def speakers_alternate(self) -> "DialogueScript":
        for left, right in zip(self.turns, self.turns[1:]):
            if left.speaker == right.speaker:
                raise ValueError("adjacent script turns must alternate speakers")
        return self


class AudioTurn(BaseModel):
    speaker: int
    text: str
    audio_path: str
    start_ms: int
    end_ms: int

    @model_validator(mode="after")
    def end_after_start(self) -> "AudioTurn":
        if self.end_ms <= self.start_ms:
            raise ValueError("turn end_ms must be greater than start_ms")
        return self


class OverlapMarker(BaseModel):
    turn_a: int
    turn_b: int
    overlap_start_ms: int
    overlap_end_ms: int
    type: str


class ConversationMetadata(BaseModel):
    conversation_id: str
    category: str
    language: str = "en"
    duration_sec: float
    script_path: str | None = None
    mixed_audio_path: str | None = None
    turns: list[AudioTurn]
    overlaps: list[OverlapMarker] = Field(default_factory=list)


class MimiConversationTokens(BaseModel):
    conversation_id: str
    source_json: str
    codebook_count: int
    training_codebooks: int
    codes_full: list[list[int]]
    codes_training: list[list[int]]


class TrainingSequence(BaseModel):
    conversation_id: str
    source_json: str
    tokens: list[str | int]
    loss_mask: list[int]

    @model_validator(mode="after")
    def mask_matches_tokens(self) -> "TrainingSequence":
        if len(self.tokens) != len(self.loss_mask):
            raise ValueError("loss_mask must have same length as tokens")
        return self


def as_posix_relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
