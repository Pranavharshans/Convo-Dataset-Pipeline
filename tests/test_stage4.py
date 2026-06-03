from pathlib import Path

from convo_ds.config import default_config
from convo_ds.jsonl import append_jsonl, read_json
from convo_ds.schemas import DialogueScript, ScriptTurn, Speaker
from convo_ds.stage3 import synthesize_stage3
from convo_ds.stage4 import inject_overlaps
from convo_ds.validation import validate_stage_dir


def test_inject_overlaps_from_stage3(tmp_path: Path) -> None:
    scripts_path = tmp_path / "scripts.jsonl"
    stage3_dir = tmp_path / "stage3"
    stage4_dir = tmp_path / "stage4"
    append_jsonl(
        scripts_path,
        [
            DialogueScript(
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
        ],
    )
    synthesize_stage3(default_config(), scripts_path, stage3_dir, mock=True)

    result = inject_overlaps(default_config(), stage3_dir, stage4_dir, limit=1)

    assert result == {"written": 1, "skipped": 0}
    metadata = read_json(stage4_dir / "conversations" / "short_000000.json")
    assert metadata["overlaps"]
    assert (stage4_dir / metadata["mixed_audio_path"]).exists()
    assert validate_stage_dir(default_config(), stage4_dir, require_overlaps=True).ok
