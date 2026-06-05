from pathlib import Path

import pytest
import httpx

from convo_ds.config import default_config
from convo_ds.jsonl import read_jsonl
from convo_ds.schemas import DialogueScript, Speaker
import convo_ds.scripts as scripts_module
from convo_ds.scripts import (
    OpenAICompatibleClient,
    generate_scripts,
    generate_zipvoice_dialog_scripts,
    normalize_terminal_punctuation,
    parse_script_block,
    sanitize_turn_text,
    validate_script_for_bucket,
)


class _FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "[S1] Hello there.\n[S2] Hi back."}}]}


def test_parse_script_block_accepts_dia_tags() -> None:
    script = parse_script_block(
        "[S1] Hey, are you free tonight after work for a quick movie?\n"
        "[S2] Yeah, I think so, as long as it starts after dinner.\n"
        "[S1] Perfect, there is a new one at seven near the station.\n"
        "[S2] Sure, that works for me, send me the ticket link.",
        "short_000000",
        "short",
        "plans",
    )
    assert [turn.speaker for turn in script.turns] == [Speaker.s0, Speaker.s1, Speaker.s0, Speaker.s1]


def test_parse_script_block_sanitizes_zipvoice_text() -> None:
    script = parse_script_block(
        "[S1]: Hey, are you free tonight? (smiles) [laugh]\n"
        "[S2]: Yeah, I am free after dinner. [sigh]\n"
        "[S1]: Great, let's meet near the station.\n"
        "[S2]: Perfect, send me the address.",
        "short_000000",
        "short",
        "plans",
    )
    assert script.turns[0].text == "Hey, are you free tonight?"
    assert script.turns[1].text == "Yeah, I am free after dinner."


def test_sanitize_turn_text_removes_actions_and_leading_punctuation() -> None:
    assert sanitize_turn_text(": Perfect! I'll book the cabin. (checks list) [laugh]") == "Perfect! I'll book the cabin."


def test_sanitize_turn_text_adds_terminal_punctuation() -> None:
    assert sanitize_turn_text("Will you be free after work") == "Will you be free after work?"
    assert sanitize_turn_text("Sounds good Will you be free after work") == "Sounds good Will you be free after work."
    assert sanitize_turn_text("I will reserve a table") == "I will reserve a table."


def test_normalize_terminal_punctuation_preserves_existing_marks() -> None:
    assert normalize_terminal_punctuation("Are you free?") == "Are you free?"
    assert normalize_terminal_punctuation("Great!") == "Great!"


def test_validate_script_for_bucket() -> None:
    config = default_config()
    script = parse_script_block(
        "[S1] Hey, are you free tonight after work for a quick movie?\n"
        "[S2] Yeah, I think so, as long as it starts after dinner.\n"
        "[S1] Perfect, there is a new one at seven near the station.\n"
        "[S2] Sure, that works for me, send me the ticket link.",
        "short_000000",
        "short",
        "plans",
    )
    validate_script_for_bucket(script, config.buckets[0])


def test_generate_scripts_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "dialogues.jsonl"
    manifest = generate_scripts(default_config(), output, limit=5, dry_run=True)
    scripts = list(read_jsonl(output, DialogueScript))
    assert manifest["completed"] == 5
    assert len(scripts) == 5
    assert scripts[0].conversation_id == "short_000000"


def test_generate_scripts_logs_progress(tmp_path: Path, capsys) -> None:
    config = default_config()
    config.script_generation.progress_every = 5
    output = tmp_path / "dialogues.jsonl"
    generate_scripts(config, output, limit=5, dry_run=True)
    captured = capsys.readouterr()
    assert "[zipvoice-scripts] completed=5/5" in captured.out


def test_generate_scripts_bounds_empty_batches(tmp_path: Path, monkeypatch) -> None:
    config = default_config()
    config.script_generation.progress_every = 0

    def always_empty(*args, **kwargs):
        return []

    monkeypatch.setattr(scripts_module, "_generate_batch", always_empty)
    with pytest.raises(RuntimeError, match="Too many empty short batches"):
        generate_scripts(config, tmp_path / "dialogues.jsonl", limit=5, dry_run=True)


def test_openai_client_retries_timeouts(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadTimeout("slow response")
        return _FakeResponse()

    monkeypatch.setattr(scripts_module.httpx, "post", fake_post)
    monkeypatch.setattr(scripts_module.time, "sleep", lambda _seconds: None)
    client = OpenAICompatibleClient("https://example.com/v1", "key", "model", retries=1, retry_backoff_sec=0)

    assert client.complete("prompt") == "[S1] Hello there.\n[S2] Hi back."
    assert calls["count"] == 2


def test_generate_zipvoice_dialog_scripts_writes_bucket_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "zipvoice_dialog"
    manifest = generate_zipvoice_dialog_scripts(default_config(), output_dir, limit=7, dry_run=True)
    combined = list(read_jsonl(output_dir / "zipvoice_dialog_scripts.jsonl", DialogueScript))
    short = list(read_jsonl(output_dir / "short.jsonl", DialogueScript))
    medium = list(read_jsonl(output_dir / "medium.jsonl", DialogueScript))
    assert manifest["combined"]["completed"] == 7
    assert len(combined) == 7
    assert len(short) == 7
    assert len(medium) == 0
