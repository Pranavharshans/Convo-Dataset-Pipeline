from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from convo_ds.config import load_config
from convo_ds.hf_upload import upload_to_hf
from convo_ds.mimi import assemble_training_shards as run_assemble_training_shards
from convo_ds.mimi import tokenize_mimi as run_tokenize_mimi
from convo_ds.scripts import generate_scripts as run_generate_scripts
from convo_ds.scripts import generate_zipvoice_dialog_scripts as run_generate_zipvoice_dialog_scripts
from convo_ds.stage3 import synthesize_stage3 as run_synthesize_stage3
from convo_ds.stage4 import inject_overlaps as run_inject_overlaps
from convo_ds.validation import validate_scripts, validate_shards, validate_stage_dir
from convo_ds.zipvoice import export_zipvoice_dialog_tsv as run_export_zipvoice_dialog_tsv

app = typer.Typer(help="Synthetic conversation dataset pipeline.")


def _load(config: Optional[Path]):
    return load_config(config)


@app.command()
def show_config(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Print the resolved pipeline configuration."""
    cfg = _load(config)
    typer.echo(cfg.model_dump_json(indent=2))


@app.command()
def generate_scripts(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    output: Path = typer.Option(Path("data/scripts/dialogues.jsonl"), "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate text dialogue scripts from an OpenAI-compatible API."""
    cfg = _load(config)
    manifest = run_generate_scripts(cfg, output, limit=limit, dry_run=dry_run)
    typer.echo(f"generated scripts manifest: {manifest}")


@app.command()
def generate_zipvoice_dialog_scripts(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    output_dir: Path = typer.Option(Path("data/scripts/zipvoice_dialog"), "--output-dir", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate bucketed [S1]/[S2] dialogue scripts for ZipVoice-Dialog."""
    cfg = _load(config)
    manifest = run_generate_zipvoice_dialog_scripts(cfg, output_dir, limit=limit, dry_run=dry_run)
    typer.echo(f"zipvoice dialogue scripts manifest: {manifest}")


@app.command()
def export_zipvoice_dialog_tsv(
    scripts: Path = typer.Option(Path("data/scripts/zipvoice_dialog/zipvoice_dialog_scripts.jsonl"), "--scripts"),
    voice_prompts: Path = typer.Option(Path("voice_prompts"), "--voice-prompts"),
    output: Path = typer.Option(Path("data/zipvoice_dialog/test.tsv"), "--output", "-o"),
    bucket: Optional[str] = typer.Option(None, "--bucket"),
) -> None:
    """Export scripts to ZipVoice-Dialog split-prompt TSV format."""
    result = run_export_zipvoice_dialog_tsv(scripts, voice_prompts, output, bucket=bucket)
    typer.echo(f"export-zipvoice-dialog-tsv result: {result}")


@app.command()
def synth_stage3(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    scripts: Path = typer.Option(Path("data/scripts/dialogues.jsonl"), "--scripts"),
    output: Path = typer.Option(Path("data/stage3"), "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    mock: bool = typer.Option(False, "--mock"),
) -> None:
    """Synthesize clean Stage 3 dialogue audio."""
    cfg = _load(config)
    result = run_synthesize_stage3(cfg, scripts, output, limit=limit, mock=mock)
    typer.echo(f"synth-stage3 result: {result}")


@app.command()
def inject_overlap(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    stage3: Path = typer.Option(Path("data/stage3"), "--stage3"),
    output: Path = typer.Option(Path("data/stage4"), "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    seed: int = typer.Option(13, "--seed"),
) -> None:
    """Create Stage 4 overlapped dialogue from Stage 3."""
    cfg = _load(config)
    result = run_inject_overlaps(cfg, stage3, output, limit=limit, seed=seed)
    typer.echo(f"inject-overlap result: {result}")


@app.command()
def tokenize_mimi(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    stage: Path = typer.Option(Path("data/stage3"), "--stage"),
    output: Path = typer.Option(Path("data/tokens/stage3_mimi.jsonl"), "--output", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    mock: bool = typer.Option(False, "--mock"),
) -> None:
    """Tokenize audio with Mimi."""
    cfg = _load(config)
    result = run_tokenize_mimi(cfg, stage, output, limit=limit, mock=mock)
    typer.echo(f"tokenize-mimi result: {result}")


@app.command()
def assemble_shards(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    tokens: Path = typer.Option(Path("data/tokens/stage3_mimi.jsonl"), "--tokens"),
    output: Path = typer.Option(Path("data/shards/stage3_train.jsonl"), "--output", "-o"),
) -> None:
    """Assemble model-ready JSONL training shards."""
    _load(config)
    result = run_assemble_training_shards(tokens, output)
    typer.echo(f"assemble-shards result: {result}")


@app.command()
def validate(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    subset: str = typer.Option("stage3", "--subset"),
    path: Path = typer.Option(Path("data/stage3"), "--path"),
) -> None:
    """Validate generated dataset artifacts."""
    cfg = _load(config)
    if subset == "scripts":
        report = validate_scripts(cfg, path)
    elif subset == "stage3":
        report = validate_stage_dir(cfg, path)
    elif subset == "stage4":
        report = validate_stage_dir(cfg, path, require_overlaps=True)
    elif subset == "shards":
        report = validate_shards(path)
    else:
        raise typer.BadParameter("subset must be scripts, stage3, stage4, or shards")
    typer.echo(report.as_dict())
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def upload_hf(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    subset: str = typer.Option("all", "--subset"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Upload dataset artifacts to Hugging Face."""
    cfg = _load(config)
    result = upload_to_hf(cfg, data_dir, subset=subset, dry_run=dry_run)
    typer.echo(result)


if __name__ == "__main__":
    app()
