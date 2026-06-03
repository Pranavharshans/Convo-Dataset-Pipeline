from pathlib import Path

from convo_ds.config import default_config
from convo_ds.jsonl import append_jsonl
from convo_ds.schemas import DialogueScript, ScriptTurn, Speaker
from convo_ds.stage3 import synthesize_stage3
from convo_ds.validation import validate_scripts, validate_stage_dir


def _script() -> DialogueScript:
    return DialogueScript(
        conversation_id="short_000000",
        category="short",
        topic="plans",
        turns=[
            ScriptTurn(speaker=Speaker.s0, text="Hey, are you free tonight after work for a quick movie?"),
            ScriptTurn(speaker=Speaker.s1, text="Yeah, I think so, as long as it starts after dinner."),
            ScriptTurn(speaker=Speaker.s0, text="Perfect, there is a new one at seven near the station."),
            ScriptTurn(speaker=Speaker.s1, text="Sure, that works for me, send me the ticket link."),
        ],
    )


def test_validate_scripts(tmp_path: Path) -> None:
    path = tmp_path / "scripts.jsonl"
    append_jsonl(path, [_script()])
    report = validate_scripts(default_config(), path)
    assert report.ok
    assert report.checked == 1


def test_validate_stage3_dir(tmp_path: Path) -> None:
    scripts_path = tmp_path / "scripts.jsonl"
    output_dir = tmp_path / "stage3"
    append_jsonl(scripts_path, [_script()])
    synthesize_stage3(default_config(), scripts_path, output_dir, mock=True)
    report = validate_stage_dir(default_config(), output_dir)
    assert report.ok
    assert report.checked == 1
