from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from convo_ds.config import ConversationBucket, PipelineConfig
from convo_ds.jsonl import append_jsonl, read_json, read_jsonl, write_json
from convo_ds.schemas import DialogueScript, ScriptTurn, Speaker

SCRIPT_LINE_RE = re.compile(r"^\[(S0|S1|S2)\]\s*(.+)$")


DEFAULT_TOPICS = [
    "making evening plans",
    "fixing a work proposal",
    "switching careers",
    "a strange conference story",
    "planning a weekend trip",
    "resolving a small misunderstanding",
    "discussing a product launch",
    "remembering a school incident",
    "helping a friend make a decision",
    "debating where to eat",
]


@dataclass
class RateLimiter:
    requests_per_minute: int
    last_request_at: float = 0.0

    def wait(self) -> None:
        if self.requests_per_minute <= 0:
            return
        interval = 60.0 / self.requests_per_minute
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self.last_request_at = time.monotonic()


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


def parse_script_block(text: str, conversation_id: str, category: str, topic: str) -> DialogueScript:
    turns: list[ScriptTurn] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SCRIPT_LINE_RE.match(line)
        if not match:
            continue
        raw_speaker, turn_text = match.groups()
        speaker = Speaker.s0 if raw_speaker in {"S0", "S1"} else Speaker.s1
        if raw_speaker == "S2":
            speaker = Speaker.s1
        turns.append(ScriptTurn(speaker=speaker, text=turn_text))
    return DialogueScript(
        conversation_id=conversation_id,
        category=category,
        topic=topic,
        turns=turns,
        estimated_duration_sec=estimate_duration_sec(turns),
    )


def validate_script_for_bucket(script: DialogueScript, bucket: ConversationBucket) -> None:
    if script.category != bucket.name:
        raise ValueError(f"script category {script.category} does not match bucket {bucket.name}")
    if not bucket.turn_min <= len(script.turns) <= bucket.turn_max:
        raise ValueError(f"{script.conversation_id} has {len(script.turns)} turns, expected {bucket.turn_min}-{bucket.turn_max}")
    if script.estimated_duration_sec is not None:
        lower = bucket.duration_min_sec * 0.25
        upper = bucket.duration_max_sec * 1.5
        if not lower <= script.estimated_duration_sec <= upper:
            raise ValueError(f"{script.conversation_id} estimated duration is outside prompt tolerance")


def estimate_duration_sec(turns: list[ScriptTurn]) -> float:
    words = sum(len(turn.text.split()) for turn in turns)
    nonverbal_bonus = sum(turn.text.count("(") for turn in turns) * 0.4
    return max(1.0, words / 2.6 + nonverbal_bonus)


def build_prompt(bucket: ConversationBucket, topics: list[str], count: int) -> str:
    topic_list = topics or DEFAULT_TOPICS
    sampled_topics = random.sample(topic_list, k=min(len(topic_list), max(1, count)))
    return (
        "Write natural two-person English dialogue scripts for synthetic speech generation.\n"
        "Use exactly [S1] and [S2] speaker tags, alternating speakers every turn.\n"
        "Include occasional non-verbal tags such as (laughs), (sighs), (gasps), (whispers), or (coughs), but keep them natural.\n"
        f"Create {count} separate conversations for category '{bucket.name}'.\n"
        f"Each conversation must have {bucket.turn_min}-{bucket.turn_max} turns and target about {bucket.duration_avg_sec:.0f} seconds.\n"
        f"Category purpose: {bucket.description}\n"
        f"Use varied topics like: {', '.join(sampled_topics)}.\n"
        "Separate conversations with a line containing only ---.\n"
        "Do not add titles, numbering, markdown fences, or explanations."
    )


def split_response_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"^\s*---\s*$", text, flags=re.MULTILINE) if block.strip()]
    return blocks or [text.strip()]


def generate_scripts(config: PipelineConfig, output_path: Path, limit: int | None = None, dry_run: bool = False) -> dict:
    generated_ids = {script.conversation_id for script in read_jsonl(output_path, DialogueScript)}
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = read_json(manifest_path) if manifest_path.exists() else {"completed": 0, "failed": 0}

    client = _make_client(config)
    limiter = RateLimiter(config.script_generation.requests_per_minute)
    total_target = limit if limit is not None else sum(bucket.count for bucket in config.buckets)
    written = 0

    for bucket in config.buckets:
        bucket_target = min(bucket.count, max(0, total_target - written))
        while written < total_target and _bucket_count(output_path, bucket.name) < bucket_target:
            batch_count = min(config.script_generation.conversations_per_request, bucket_target - _bucket_count(output_path, bucket.name))
            if batch_count <= 0:
                break
            scripts = _generate_batch(config, client, limiter, bucket, batch_count, generated_ids, dry_run)
            append_jsonl(output_path, scripts)
            for script in scripts:
                generated_ids.add(script.conversation_id)
            written += len(scripts)
            manifest["completed"] = manifest.get("completed", 0) + len(scripts)
            write_json(manifest_path, manifest)
        if written >= total_target:
            break
    return manifest


def _generate_batch(
    config: PipelineConfig,
    client: OpenAICompatibleClient | None,
    limiter: RateLimiter,
    bucket: ConversationBucket,
    batch_count: int,
    generated_ids: set[str],
    dry_run: bool,
) -> list[DialogueScript]:
    prompt = build_prompt(bucket, config.script_generation.topics, batch_count)
    for attempt in range(config.script_generation.max_retries + 1):
        if dry_run:
            response = _dry_run_response(bucket, batch_count)
        else:
            if client is None:
                raise RuntimeError("LLM client is not configured; set base_url/model/api key or use --dry-run")
            limiter.wait()
            response = client.complete(prompt)
        scripts: list[DialogueScript] = []
        for block in split_response_blocks(response):
            conversation_id = _next_id(bucket.name, generated_ids)
            try:
                script = parse_script_block(block, conversation_id, bucket.name, random.choice(config.script_generation.topics or DEFAULT_TOPICS))
                validate_script_for_bucket(script, bucket)
            except ValueError:
                continue
            scripts.append(script)
            generated_ids.add(conversation_id)
            if len(scripts) == batch_count:
                return scripts
        if attempt >= config.script_generation.max_retries:
            raise RuntimeError(f"Could not generate valid {bucket.name} batch after retries")
    return []


def _make_client(config: PipelineConfig) -> OpenAICompatibleClient | None:
    settings = config.script_generation
    api_key = os.environ.get(settings.api_key_env)
    if not settings.base_url or not settings.model or not api_key:
        return None
    return OpenAICompatibleClient(settings.base_url, api_key, settings.model)


def _next_id(category: str, existing: set[str]) -> str:
    index = len([item for item in existing if item.startswith(f"{category}_")])
    while True:
        candidate = f"{category}_{index:06d}"
        if candidate not in existing:
            return candidate
        index += 1


def _bucket_count(output_path: Path, bucket_name: str) -> int:
    return sum(1 for script in read_jsonl(output_path, DialogueScript) if script.category == bucket_name)


def _dry_run_response(bucket: ConversationBucket, count: int) -> str:
    blocks = []
    turn_count = bucket.turn_min
    target_words_per_turn = max(4, int(bucket.duration_avg_sec * 2.4 / turn_count))
    for _ in range(count):
        lines = []
        for index in range(turn_count):
            tag = "[S1]" if index % 2 == 0 else "[S2]"
            words = ["This", "is", "a", bucket.name, "dialogue", "turn", str(index + 1)]
            while len(words) < target_words_per_turn:
                words.extend(["with", "natural", "spoken", "detail"])
            lines.append(f"{tag} {' '.join(words[:target_words_per_turn])}.")
        blocks.append("\n".join(lines))
    return "\n---\n".join(blocks)
