from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


class ConversationBucket(BaseModel):
    name: str
    count: int
    turn_min: int
    turn_max: int
    duration_min_sec: float
    duration_max_sec: float
    duration_avg_sec: float
    words_min: int
    words_max: int
    description: str


class ScriptGenerationConfig(BaseModel):
    base_url: str | None = None
    api_key_env: str = "LLM_API_KEY"
    model: str | None = None
    api_timeout_sec: float = 240.0
    api_retries: int = 5
    api_retry_backoff_sec: float = 5.0
    requests_per_minute: int = 35
    conversations_per_request: int = 5
    progress_every: int = 500
    max_retries: int = 3
    topics: list[str] = Field(default_factory=list)


class DiaConfig(BaseModel):
    model_id: str = "nari-labs/Dia-1.6B"
    sample_rate: int = 44100
    output_sample_rate: int = 24000
    precision: str = "bfloat16"
    compile: bool = True
    duration_regen_limit: int = 2


class OverlapConfig(BaseModel):
    target_hours: float = 200.0
    backchannel_ratio: float = 0.7
    interruption_ratio: float = 0.25
    simultaneous_start_ratio: float = 0.05
    overlap_attenuation: float = 0.82


class MimiConfig(BaseModel):
    model_id: str = "kyutai/mimi"
    training_codebooks: int = 8
    token_count: int = 2048


class HuggingFaceConfig(BaseModel):
    repo_env: str = "HF_DATASET_REPO"
    token_env: str = "HF_TOKEN"


class PipelineConfig(BaseModel):
    output_dir: Path = Path("data")
    buckets: list[ConversationBucket]
    script_generation: ScriptGenerationConfig = Field(default_factory=ScriptGenerationConfig)
    dia: DiaConfig = Field(default_factory=DiaConfig)
    overlap: OverlapConfig = Field(default_factory=OverlapConfig)
    mimi: MimiConfig = Field(default_factory=MimiConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)


def default_buckets() -> list[ConversationBucket]:
    return [
        ConversationBucket(
            name="short",
            count=15000,
            turn_min=4,
            turn_max=6,
            duration_min_sec=12,
            duration_max_sec=20,
            duration_avg_sec=16,
            words_min=28,
            words_max=55,
            description="Fast alternation, greetings, quick plans, yes/no exchanges.",
        ),
        ConversationBucket(
            name="medium",
            count=20000,
            turn_min=6,
            turn_max=8,
            duration_min_sec=25,
            duration_max_sec=40,
            duration_avg_sec=30,
            words_min=70,
            words_max=115,
            description="Natural conversational flow and main workhorse examples.",
        ),
        ConversationBucket(
            name="long",
            count=10000,
            turn_min=8,
            turn_max=10,
            duration_min_sec=40,
            duration_max_sec=55,
            duration_avg_sec=45,
            words_min=115,
            words_max=170,
            description="Sustained attention and nuanced topic development.",
        ),
        ConversationBucket(
            name="extended",
            count=5000,
            turn_min=10,
            turn_max=14,
            duration_min_sec=55,
            duration_max_sec=90,
            duration_avg_sec=65,
            words_min=170,
            words_max=260,
            description="Story arcs, debates, and long-range coherence.",
        ),
    ]


def default_config() -> PipelineConfig:
    return PipelineConfig(buckets=default_buckets())


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return default_config()
    raw = _read_mapping(path)
    if "buckets" not in raw:
        raw["buckets"] = [bucket.model_dump() for bucket in default_buckets()]
    return PipelineConfig.model_validate(raw)


def _read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".json":
        return json.loads(data.decode("utf-8"))
    if suffix in {".toml", ".tml"}:
        return tomllib.loads(data.decode("utf-8"))
    raise ValueError(f"Unsupported config format: {path}")
