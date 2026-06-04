from pathlib import Path

from convo_ds.config import default_config
from convo_ds.jsonl import read_jsonl
from convo_ds.schemas import DialogueScript, Speaker
from convo_ds.scripts import generate_scripts, generate_zipvoice_dialog_scripts, parse_script_block, validate_script_for_bucket


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
