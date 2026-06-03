from pathlib import Path

from convo_ds.config import default_config
from convo_ds.jsonl import append_jsonl, read_jsonl
from convo_ds.mimi import assemble_training_shards, tokenize_mimi
from convo_ds.schemas import DialogueScript, MimiConversationTokens, ScriptTurn, Speaker, TrainingSequence
from convo_ds.stage3 import synthesize_stage3
from convo_ds.validation import validate_shards


def test_tokenize_mimi_and_assemble_shards(tmp_path: Path) -> None:
    scripts_path = tmp_path / "scripts.jsonl"
    stage3_dir = tmp_path / "stage3"
    tokens_path = tmp_path / "tokens.jsonl"
    shards_path = tmp_path / "shards.jsonl"
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

    token_result = tokenize_mimi(default_config(), stage3_dir, tokens_path, mock=True)
    shard_result = assemble_training_shards(tokens_path, shards_path)

    token_records = list(read_jsonl(tokens_path, MimiConversationTokens))
    shard_records = list(read_jsonl(shards_path, TrainingSequence))
    assert token_result == {"written": 1}
    assert shard_result == {"written": 1}
    assert token_records[0].training_codebooks == 8
    assert token_records[0].codebook_count == 16
    assert any(token == "<|s0_audio|>" for token in shard_records[0].tokens)
    assert sum(shard_records[0].loss_mask) > 0
    assert validate_shards(shards_path).ok
