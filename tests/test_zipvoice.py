from pathlib import Path

from convo_ds.jsonl import append_jsonl
from convo_ds.schemas import DialogueScript, ScriptTurn, Speaker
from convo_ds.zipvoice import export_zipvoice_dialog_tsv, script_to_zipvoice_text


def test_script_to_zipvoice_text() -> None:
    script = DialogueScript(
        conversation_id="short_000000",
        category="short",
        topic="plans",
        turns=[
            ScriptTurn(speaker=Speaker.s0, text="Hey there."),
            ScriptTurn(speaker=Speaker.s1, text="Hello back."),
        ],
    )
    assert script_to_zipvoice_text(script) == "[S1] Hey there. [S2] Hello back."


def test_export_zipvoice_dialog_tsv_split_prompt(tmp_path: Path) -> None:
    scripts_path = tmp_path / "scripts.jsonl"
    prompts_dir = tmp_path / "voice_prompts" / "pair_000"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "meta.json").write_text(
        '{"s1_prompt_transcription":"Hello.","s2_prompt_transcription":"Hi.","s1_prompt_wav":"s1.wav","s2_prompt_wav":"s2.wav"}',
        encoding="utf-8",
    )
    append_jsonl(
        scripts_path,
        [
            DialogueScript(
                conversation_id="short_000000",
                category="short",
                topic="plans",
                turns=[
                    ScriptTurn(speaker=Speaker.s0, text="Hey there."),
                    ScriptTurn(speaker=Speaker.s1, text="Hello back."),
                ],
            )
        ],
    )

    result = export_zipvoice_dialog_tsv(scripts_path, tmp_path / "voice_prompts", tmp_path / "test.tsv")

    line = (tmp_path / "test.tsv").read_text(encoding="utf-8").strip()
    assert result["written"] == 1
    assert line.split("\t") == [
        "short_000000.wav",
        "Hello.",
        "Hi.",
        str(prompts_dir / "s1.wav"),
        str(prompts_dir / "s2.wav"),
        "[S1] Hey there. [S2] Hello back.",
    ]
