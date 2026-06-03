from pathlib import Path

from convo_ds.config import default_config
from convo_ds.jsonl import append_jsonl, read_json
from convo_ds.schemas import DialogueScript, ScriptTurn, Speaker
from convo_ds.stage3 import synthesize_stage3


def test_synthesize_stage3_mock_writes_metadata(tmp_path: Path) -> None:
    scripts_path = tmp_path / "scripts.jsonl"
    output_dir = tmp_path / "stage3"
    script = DialogueScript(
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
    append_jsonl(scripts_path, [script])

    result = synthesize_stage3(default_config(), scripts_path, output_dir, mock=True)

    assert result == {"written": 1, "skipped": 0}
    metadata = read_json(output_dir / "conversations" / "short_000000.json")
    assert metadata["conversation_id"] == "short_000000"
    assert len(metadata["turns"]) == 4
    assert (output_dir / metadata["mixed_audio_path"]).exists()
