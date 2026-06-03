from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from convo_ds.config import load_config
from convo_ds.scripts import generate_scripts as run_generate_scripts
from convo_ds.stage3 import synthesize_stage3 as run_synthesize_stage3
from convo_ds.validation import validate_scripts, validate_shards, validate_stage_dir

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
def inject_overlap(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Create Stage 4 overlapped dialogue from Stage 3."""
    _load(config)
    typer.echo("inject-overlap is not implemented yet.")


@app.command()
def tokenize_mimi(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Tokenize audio with Mimi."""
    _load(config)
    typer.echo("tokenize-mimi is not implemented yet.")


@app.command()
def assemble_shards(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Assemble model-ready JSONL training shards."""
    _load(config)
    typer.echo("assemble-shards is not implemented yet.")


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
    subset: str = typer.Option("all", "--subset"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Upload dataset artifacts to Hugging Face."""
    _load(config)
    typer.echo(f"upload-hf subset={subset} dry_run={dry_run}; not implemented yet.")


if __name__ == "__main__":
    app()
