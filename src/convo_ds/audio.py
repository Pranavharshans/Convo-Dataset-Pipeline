from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[0] <= 2 and array.shape[1] > array.shape[0]:
            array = array.T
        array = array.mean(axis=1)
    return np.clip(array, -1.0, 1.0)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, ensure_mono_float32(audio), sample_rate, subtype="PCM_16")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return ensure_mono_float32(audio), sample_rate


def wav_duration_sec(path: Path) -> float:
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def silence(duration_sec: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(duration_sec * sample_rate)), dtype=np.float32)


def concat_with_gaps(chunks: list[np.ndarray], gap_sec: float, sample_rate: int) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    gap = silence(gap_sec, sample_rate)
    output: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        if index:
            output.append(gap)
        output.append(ensure_mono_float32(chunk))
    return np.concatenate(output) if output else np.zeros(0, dtype=np.float32)
