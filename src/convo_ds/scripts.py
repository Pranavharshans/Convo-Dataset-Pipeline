from __future__ import annotations

import os
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import httpx

from convo_ds.config import ConversationBucket, PipelineConfig
from convo_ds.jsonl import append_jsonl, read_json, read_jsonl, write_json
from convo_ds.schemas import DialogueScript, ScriptTurn, Speaker

SCRIPT_LINE_RE = re.compile(r"^\[(S0|S1|S2)\]\s*(.+)$")
PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
BRACKETED_ACTION_RE = re.compile(r"\[(?!S[012]\])[^]]+\]")
TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?]$")


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
    lock: Lock | None = None

    def wait(self) -> None:
        if self.requests_per_minute <= 0:
            return
        if self.lock is None:
            self.lock = Lock()
        with self.lock:
            interval = 60.0 / self.requests_per_minute
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self.last_request_at = time.monotonic()


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.9,
        top_p: float = 1.0,
        max_tokens: int | None = 4096,
        timeout: float = 240.0,
        retries: int = 5,
        retry_backoff_sec: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff_sec = retry_backoff_sec

    def complete(self, prompt: str) -> str:
        last_error: Exception | None = None
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                return payload["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
                if attempt >= self.retries:
                    break
                wait_sec = self.retry_backoff_sec * (attempt + 1)
                print(f"[zipvoice-scripts] api_retry={attempt + 1}/{self.retries} wait_sec={wait_sec:.1f} error={type(exc).__name__}", flush=True)
                time.sleep(wait_sec)
        raise RuntimeError(f"LLM API failed after retries: {last_error}") from last_error


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
        turns.append(ScriptTurn(speaker=speaker, text=sanitize_turn_text(turn_text)))
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
    word_count = count_script_words(script)
    if not bucket.words_min <= word_count <= bucket.words_max:
        raise ValueError(f"{script.conversation_id} has {word_count} words, expected {bucket.words_min}-{bucket.words_max}")
    for turn in script.turns:
        if PARENTHETICAL_RE.search(turn.text) or BRACKETED_ACTION_RE.search(turn.text):
            raise ValueError(f"{script.conversation_id} contains stage directions")
        if not TERMINAL_PUNCTUATION_RE.search(turn.text):
            raise ValueError(f"{script.conversation_id} turn is missing terminal punctuation")
    if script.estimated_duration_sec is not None:
        lower = bucket.duration_min_sec * 0.25
        upper = bucket.duration_max_sec * 1.5
        if not lower <= script.estimated_duration_sec <= upper:
            raise ValueError(f"{script.conversation_id} estimated duration is outside prompt tolerance")


def estimate_duration_sec(turns: list[ScriptTurn]) -> float:
    words = sum(len(turn.text.split()) for turn in turns)
    nonverbal_bonus = sum(turn.text.count("(") for turn in turns) * 0.4
    return max(1.0, words / 2.6 + nonverbal_bonus)


def count_script_words(script: DialogueScript) -> int:
    return sum(len(re.findall(r"\b[\w']+\b", turn.text)) for turn in script.turns)


def sanitize_turn_text(text: str) -> str:
    cleaned = PARENTHETICAL_RE.sub("", text)
    cleaned = BRACKETED_ACTION_RE.sub("", cleaned)
    cleaned = re.sub(r"^[\s:：\-–—]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = normalize_terminal_punctuation(cleaned)
    return cleaned


def normalize_terminal_punctuation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if TERMINAL_PUNCTUATION_RE.search(stripped):
        return stripped
    if stripped.lower().startswith(("who ", "what ", "when ", "where ", "why ", "how ", "do ", "does ", "did ", "can ", "could ", "would ", "will ", "should ", "are ", "is ", "was ", "were ")):
        return f"{stripped}?"
    return f"{stripped}."


def build_prompt(bucket: ConversationBucket, topics: list[str], count: int) -> tuple[str, list[str]]:
    topic_list = topics or DEFAULT_TOPICS
    if len(topic_list) >= count:
        sampled_topics = random.sample(topic_list, k=max(1, count))
    else:
        sampled_topics = [random.choice(topic_list) for _ in range(max(1, count))]
    numbered_topics = "\n".join(f"{index + 1}. {topic}" for index, topic in enumerate(sampled_topics))
    return (
        "Write natural two-person English dialogue scripts for ZipVoice-Dialog-Stereo TTS.\n"
        "Use exactly [S1] and [S2] speaker tags, alternating speakers every turn.\n"
        "Format every turn exactly as '[S1] text' or '[S2] text'. Do not write '[S1]:' with a colon.\n"
        "Do not include emotion tags, nonverbal tags, parenthetical stage directions, action tags, markdown, numbering, titles, or explanations.\n"
        "Forbidden examples: (laughs), (sighs), (gasps), (smiles), (checks phone), [laugh], [sigh], [whisper].\n"
        "Use plain speakable dialogue only; every word should be safe to send directly to TTS.\n"
        "Use normal English punctuation. Every turn must end with a period, question mark, or exclamation mark.\n"
        "Do not omit punctuation between sentences.\n"
        f"Create {count} separate conversations for category '{bucket.name}'.\n"
        f"Each conversation must have {bucket.turn_min}-{bucket.turn_max} turns.\n"
        f"Each conversation must contain {bucket.words_min}-{bucket.words_max} spoken words total.\n"
        f"Target about {bucket.duration_avg_sec:.0f} seconds after TTS; do not make every category similar length.\n"
        "Short should feel quick. Medium should have more development. Long should sustain a topic. Extended should have a story arc or deeper explanation.\n"
        f"Category purpose: {bucket.description}\n"
        "Use these topics in order, one topic per conversation, and keep the conversation actually about its assigned topic:\n"
        f"{numbered_topics}\n"
        "Separate conversations with a line containing only ---.\n"
        "Do not add titles, numbering, markdown fences, or explanations."
    ), sampled_topics


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
    existing_total = min(len(generated_ids), total_target)
    written = existing_total
    progress_every = max(0, config.script_generation.progress_every)
    next_progress = _next_progress_mark(existing_total, progress_every)
    if existing_total:
        _log_progress(existing_total, total_target, "resume", 0)

    for bucket in config.buckets:
        bucket_target = min(bucket.count, max(0, total_target - written))
        if config.script_generation.concurrency > 1 and not dry_run:
            written, next_progress = _generate_bucket_concurrent(
                config,
                output_path,
                manifest_path,
                manifest,
                client,
                limiter,
                bucket,
                bucket_target,
                total_target,
                written,
                generated_ids,
                next_progress,
                progress_every,
            )
        else:
            written, next_progress = _generate_bucket_sequential(
                config,
                output_path,
                manifest_path,
                manifest,
                client,
                limiter,
                bucket,
                bucket_target,
                total_target,
                written,
                generated_ids,
                next_progress,
                progress_every,
                dry_run,
            )
        if written >= total_target:
            break
    _log_progress(written, total_target, "done", 0)
    return manifest


def _generate_bucket_sequential(
    config: PipelineConfig,
    output_path: Path,
    manifest_path: Path,
    manifest: dict,
    client: OpenAICompatibleClient | None,
    limiter: RateLimiter,
    bucket: ConversationBucket,
    bucket_target: int,
    total_target: int,
    written: int,
    generated_ids: set[str],
    next_progress: int,
    progress_every: int,
    dry_run: bool,
) -> tuple[int, int]:
    empty_batches = 0
    while written < total_target and _bucket_count(output_path, bucket.name) < bucket_target:
        bucket_before = _bucket_count(output_path, bucket.name)
        batch_count = min(config.script_generation.conversations_per_request, bucket_target - bucket_before)
        if batch_count <= 0:
            break
        start_index = _next_bucket_index(bucket.name, generated_ids)
        scripts = _generate_batch(config, client, limiter, bucket, batch_count, start_index, dry_run)
        if not scripts:
            empty_batches += 1
            manifest["failed"] = manifest.get("failed", 0) + 1
            write_json(manifest_path, manifest)
            _log_rejection(bucket.name, "empty accepted batch after retries")
            if empty_batches >= 5:
                raise RuntimeError(f"Too many empty {bucket.name} batches; inspect prompt/model output")
            continue
        empty_batches = 0
        written, next_progress = _commit_scripts(
            output_path, manifest_path, manifest, scripts, generated_ids, written, total_target, bucket.name, bucket_before, next_progress, progress_every
        )
    return written, next_progress


def _generate_bucket_concurrent(
    config: PipelineConfig,
    output_path: Path,
    manifest_path: Path,
    manifest: dict,
    client: OpenAICompatibleClient | None,
    limiter: RateLimiter,
    bucket: ConversationBucket,
    bucket_target: int,
    total_target: int,
    written: int,
    generated_ids: set[str],
    next_progress: int,
    progress_every: int,
) -> tuple[int, int]:
    if client is None:
        raise RuntimeError("LLM client is not configured; set base_url/model/api key or use --dry-run")
    bucket_completed = _bucket_count(output_path, bucket.name)
    next_start_index = _next_bucket_index(bucket.name, generated_ids)
    in_flight: set[Future[list[DialogueScript]]] = set()
    empty_batches = 0
    max_workers = max(1, config.script_generation.concurrency)
    print(f"[zipvoice-scripts] bucket={bucket.name} concurrency={max_workers} rpm={config.script_generation.requests_per_minute}", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while written < total_target and (bucket_completed < bucket_target or in_flight):
            while bucket_completed + _reserved_count(in_flight) < bucket_target and len(in_flight) < max_workers and written + _reserved_count(in_flight) < total_target:
                remaining = min(bucket_target - bucket_completed - _reserved_count(in_flight), total_target - written - _reserved_count(in_flight))
                batch_count = min(config.script_generation.conversations_per_request, remaining)
                if batch_count <= 0:
                    break
                future = executor.submit(_generate_batch, config, client, limiter, bucket, batch_count, next_start_index, False)
                future.reserved_count = batch_count  # type: ignore[attr-defined]
                in_flight.add(future)
                next_start_index += batch_count
            if not in_flight:
                break
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                scripts = future.result()
                if not scripts:
                    empty_batches += 1
                    manifest["failed"] = manifest.get("failed", 0) + 1
                    write_json(manifest_path, manifest)
                    _log_rejection(bucket.name, "empty accepted batch after retries")
                    if empty_batches >= 5:
                        raise RuntimeError(f"Too many empty {bucket.name} batches; inspect prompt/model output")
                    continue
                empty_batches = 0
                bucket_before = bucket_completed
                written, next_progress = _commit_scripts(
                    output_path, manifest_path, manifest, scripts, generated_ids, written, total_target, bucket.name, bucket_before, next_progress, progress_every
                )
                bucket_completed += len(scripts)
    return written, next_progress


def _commit_scripts(
    output_path: Path,
    manifest_path: Path,
    manifest: dict,
    scripts: list[DialogueScript],
    generated_ids: set[str],
    written: int,
    total_target: int,
    bucket_name: str,
    bucket_before: int,
    next_progress: int,
    progress_every: int,
) -> tuple[int, int]:
    append_jsonl(output_path, scripts)
    for script in scripts:
        generated_ids.add(script.conversation_id)
    written += len(scripts)
    manifest["completed"] = written
    write_json(manifest_path, manifest)
    bucket_after = bucket_before + len(scripts)
    if progress_every and written >= next_progress:
        _log_progress(written, total_target, bucket_name, bucket_after)
        next_progress = _next_progress_mark(written, progress_every)
    return written, next_progress


def generate_zipvoice_dialog_scripts(
    config: PipelineConfig,
    output_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "zipvoice_dialog_scripts.jsonl"
    manifest: dict[str, dict] = {"combined": generate_scripts(config, combined_path, limit=limit, dry_run=dry_run)}
    scripts_by_bucket: dict[str, list[DialogueScript]] = {bucket.name: [] for bucket in config.buckets}
    for script in read_jsonl(combined_path, DialogueScript):
        scripts_by_bucket.setdefault(script.category, []).append(script)
    for bucket_name, scripts in scripts_by_bucket.items():
        bucket_path = output_dir / f"{bucket_name}.jsonl"
        if bucket_path.exists():
            bucket_path.unlink()
        append_jsonl(bucket_path, scripts)
        manifest[bucket_name] = {"completed": len(scripts), "path": bucket_path.as_posix()}
    write_json(output_dir / "zipvoice_dialog_scripts.manifest.json", manifest)
    return manifest


def _generate_batch(
    config: PipelineConfig,
    client: OpenAICompatibleClient | None,
    limiter: RateLimiter,
    bucket: ConversationBucket,
    batch_count: int,
    start_index: int,
    dry_run: bool,
) -> list[DialogueScript]:
    prompt, assigned_topics = build_prompt(bucket, config.script_generation.topics, batch_count)
    accepted: list[DialogueScript] = []
    reject_reasons: dict[str, int] = {}
    for attempt in range(config.script_generation.max_retries + 1):
        if dry_run:
            response = _dry_run_response(bucket, batch_count - len(accepted))
        else:
            if client is None:
                raise RuntimeError("LLM client is not configured; set base_url/model/api key or use --dry-run")
            limiter.wait()
            response = client.complete(prompt)
        for block_index, block in enumerate(split_response_blocks(response)):
            conversation_id = f"{bucket.name}_{start_index + len(accepted):06d}"
            try:
                topic = assigned_topics[min(block_index, len(assigned_topics) - 1)] if assigned_topics else random.choice(config.script_generation.topics or DEFAULT_TOPICS)
                script = parse_script_block(block, conversation_id, bucket.name, topic)
                validate_script_for_bucket(script, bucket)
            except ValueError as exc:
                reason = _short_rejection_reason(str(exc))
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue
            accepted.append(script)
            if len(accepted) == batch_count:
                if reject_reasons:
                    _log_rejection(bucket.name, _format_reject_reasons(reject_reasons))
                return accepted
        if attempt >= config.script_generation.max_retries:
            if accepted:
                _log_rejection(bucket.name, f"accepted partial batch {len(accepted)}/{batch_count}; rejected {_format_reject_reasons(reject_reasons)}")
                return accepted
            _log_rejection(bucket.name, f"no valid conversations after retries; rejected {_format_reject_reasons(reject_reasons)}")
            return []
    return accepted


def _make_client(config: PipelineConfig) -> OpenAICompatibleClient | None:
    settings = config.script_generation
    api_key = os.environ.get(settings.api_key_env)
    if not settings.base_url or not settings.model or not api_key:
        return None
    return OpenAICompatibleClient(
        settings.base_url,
        api_key,
        settings.model,
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        timeout=settings.api_timeout_sec,
        retries=settings.api_retries,
        retry_backoff_sec=settings.api_retry_backoff_sec,
    )


def _next_bucket_index(category: str, existing: set[str]) -> int:
    index = len([item for item in existing if item.startswith(f"{category}_")])
    while True:
        candidate = f"{category}_{index:06d}"
        if candidate not in existing:
            return index
        index += 1


def _reserved_count(futures: set[Future[list[DialogueScript]]]) -> int:
    return sum(getattr(future, "reserved_count", 0) for future in futures)


def _bucket_count(output_path: Path, bucket_name: str) -> int:
    return sum(1 for script in read_jsonl(output_path, DialogueScript) if script.category == bucket_name)


def _next_progress_mark(current: int, progress_every: int) -> int:
    if progress_every <= 0:
        return 0
    return ((current // progress_every) + 1) * progress_every


def _log_progress(completed: int, total: int, bucket_name: str, bucket_completed: int) -> None:
    if bucket_completed:
        print(f"[zipvoice-scripts] completed={completed}/{total} bucket={bucket_name} bucket_completed={bucket_completed}", flush=True)
    else:
        print(f"[zipvoice-scripts] completed={completed}/{total} status={bucket_name}", flush=True)


def _log_rejection(bucket_name: str, reason: str) -> None:
    print(f"[zipvoice-scripts] bucket={bucket_name} rejected={reason}", flush=True)


def _short_rejection_reason(reason: str) -> str:
    reason = re.sub(r"^[^:]+:\s*", "", reason)
    if len(reason) > 140:
        return f"{reason[:137]}..."
    return reason


def _format_reject_reasons(reasons: dict[str, int]) -> str:
    if not reasons:
        return "none"
    return ", ".join(f"{reason} x{count}" for reason, count in sorted(reasons.items()))


def _dry_run_response(bucket: ConversationBucket, count: int) -> str:
    blocks = []
    turn_count = bucket.turn_min
    target_words_per_turn = max(4, int(((bucket.words_min + bucket.words_max) / 2) / turn_count))
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
