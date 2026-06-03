from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from convo_ds.config import PipelineConfig

VALID_SUBSETS = {"scripts", "stage3", "stage4", "shards", "all"}


@dataclass
class UploadPlan:
    repo_id: str
    subset: str
    files: list[Path]
    total_bytes: int

    def as_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "subset": self.subset,
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
            "files": [path.as_posix() for path in self.files],
        }


def upload_to_hf(config: PipelineConfig, data_dir: Path, subset: str = "all", dry_run: bool = False) -> dict:
    if subset not in VALID_SUBSETS:
        raise ValueError(f"subset must be one of {sorted(VALID_SUBSETS)}")
    repo_id = os.environ.get(config.huggingface.repo_env)
    token = os.environ.get(config.huggingface.token_env)
    if not repo_id:
        raise RuntimeError(f"Missing {config.huggingface.repo_env}")
    if not token and not dry_run:
        raise RuntimeError(f"Missing {config.huggingface.token_env}")

    plans = [_build_plan(repo_id, data_dir, item) for item in _expand_subset(subset)]
    if dry_run:
        return {"dry_run": True, "plans": [plan.as_dict() for plan in plans]}

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    uploaded = []
    for plan in plans:
        for path in plan.files:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path.relative_to(data_dir).as_posix(),
                repo_id=repo_id,
                repo_type="dataset",
            )
            uploaded.append(path.as_posix())
    return {"dry_run": False, "repo_id": repo_id, "uploaded": uploaded}


def _expand_subset(subset: str) -> list[str]:
    if subset == "all":
        return ["scripts", "stage3", "stage4", "shards"]
    return [subset]


def _build_plan(repo_id: str, data_dir: Path, subset: str) -> UploadPlan:
    subset_dir = data_dir / subset
    files = sorted(path for path in subset_dir.rglob("*") if path.is_file()) if subset_dir.exists() else []
    return UploadPlan(
        repo_id=repo_id,
        subset=subset,
        files=files,
        total_bytes=sum(path.stat().st_size for path in files),
    )
