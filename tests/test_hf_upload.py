from pathlib import Path

from convo_ds.config import default_config
from convo_ds.hf_upload import upload_to_hf


def test_upload_to_hf_dry_run(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    scripts_dir = data_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "dialogues.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_DATASET_REPO", "user/test-dataset")

    result = upload_to_hf(default_config(), data_dir, subset="scripts", dry_run=True)

    assert result["dry_run"] is True
    assert result["plans"][0]["repo_id"] == "user/test-dataset"
    assert result["plans"][0]["file_count"] == 1
