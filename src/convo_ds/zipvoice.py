from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from convo_ds.jsonl import read_jsonl
from convo_ds.schemas import DialogueScript, Speaker


@dataclass(frozen=True)
class VoicePromptPair:
    s1_prompt_transcription: str
    s2_prompt_transcription: str
    s1_prompt_wav: str
    s2_prompt_wav: str


def export_zipvoice_dialog_tsv(
    scripts_path: Path,
    voice_prompts_dir: Path,
    output_path: Path,
    bucket: str | None = None,
) -> dict:
    prompts = load_voice_prompt_pairs(voice_prompts_dir)
    if not prompts:
        raise ValueError(f"no voice prompt pairs found under {voice_prompts_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for index, script in enumerate(read_jsonl(scripts_path, DialogueScript)):
            if bucket and script.category != bucket:
                continue
            prompt = prompts[index % len(prompts)]
            writer.writerow(
                [
                    f"{script.conversation_id}.wav",
                    prompt.s1_prompt_transcription,
                    prompt.s2_prompt_transcription,
                    prompt.s1_prompt_wav,
                    prompt.s2_prompt_wav,
                    script_to_zipvoice_text(script),
                ]
            )
            written += 1
    return {"written": written, "output": output_path.as_posix()}


def load_voice_prompt_pairs(voice_prompts_dir: Path) -> list[VoicePromptPair]:
    pairs: list[VoicePromptPair] = []
    for meta_path in sorted(voice_prompts_dir.glob("**/meta.json")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        pairs.append(
            VoicePromptPair(
                s1_prompt_transcription=payload["s1_prompt_transcription"],
                s2_prompt_transcription=payload["s2_prompt_transcription"],
                s1_prompt_wav=_resolve_prompt_path(meta_path, payload["s1_prompt_wav"]),
                s2_prompt_wav=_resolve_prompt_path(meta_path, payload["s2_prompt_wav"]),
            )
        )
    return pairs


def script_to_zipvoice_text(script: DialogueScript) -> str:
    parts: list[str] = []
    for turn in script.turns:
        tag = "[S1]" if turn.speaker == Speaker.s0 else "[S2]"
        parts.append(f"{tag} {turn.text}")
    return " ".join(parts)


def _resolve_prompt_path(meta_path: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return path.as_posix()
    return (meta_path.parent / path).as_posix()
