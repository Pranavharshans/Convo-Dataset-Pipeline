from pathlib import Path

from convo_ds.config import default_config
from convo_ds.jsonl import read_jsonl
from convo_ds.schemas import DialogueScript, Speaker
from convo_ds.scripts import generate_scripts, parse_script_block, validate_script_for_bucket


def test_parse_script_block_accepts_dia_tags() -> None:
    script = parse_script_block(
        "[S1] Hey, you free tonight?\n[S2] Yeah, why?\n[S1] Movie at seven?\n[S2] Sure, I'm in.",
        "short_000000",
        "short",
        "plans",
    )
    assert [turn.speaker for turn in script.turns] == [Speaker.s0, Speaker.s1, Speaker.s0, Speaker.s1]


def test_validate_script_for_bucket() -> None:
    config = default_config()
    script = parse_script_block(
        "[S1] Hey, you free tonight?\n[S2] Yeah, why?\n[S1] Movie at seven?\n[S2] Sure, I'm in.",
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
