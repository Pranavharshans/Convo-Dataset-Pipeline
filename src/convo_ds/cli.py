from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from convo_ds.config import load_config

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
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate text dialogue scripts from an OpenAI-compatible API."""
    _load(config)
    typer.echo("generate-scripts is not implemented yet." if not dry_run else "generate-scripts dry run ok.")


@app.command()
def synth_stage3(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Synthesize clean Stage 3 dialogue audio."""
    _load(config)
    typer.echo("synth-stage3 is not implemented yet.")


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
def validate(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Validate generated dataset artifacts."""
    _load(config)
    typer.echo("validate is not implemented yet.")


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
