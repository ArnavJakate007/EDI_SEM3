"""FastAPI service backing the comparison dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sattsr import __version__
from sattsr.infer.pipeline import read_manifest

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ANIMATION_KINDS = frozenset({"original", "interpolated"})


def _safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Resolve a run directory, refusing anything that escapes `runs_dir`."""
    if not RUN_ID_RE.match(run_id) or run_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid run id")
    root = runs_dir.resolve()
    candidate = (root / run_id).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=400, detail="invalid run id")
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return candidate


def _require_file(path: Path, message: str) -> Path:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=message)
    return path


def create_app(runs_dir: str | Path, *, static_dir: Path | None = None) -> FastAPI:
    """Build the API over a directory of inference runs."""
    runs_root = Path(runs_dir)
    static_root = Path(static_dir) if static_dir else Path(__file__).parent / "static"

    app = FastAPI(title="sattsr dashboard", version=__version__)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, object]]:
        if not runs_root.is_dir():
            return []
        summaries: list[dict[str, object]] = []
        for entry in sorted(runs_root.iterdir()):
            if not (entry / "manifest.json").is_file():
                continue
            manifest = read_manifest(entry)
            summaries.append(
                {
                    "run_id": manifest.run_id,
                    "sensor": manifest.sensor,
                    "created_at": manifest.created_at.isoformat(),
                    "factor": manifest.factor,
                    "n_original": manifest.n_original,
                    "n_synthetic": manifest.n_synthetic,
                    "input_cadence_minutes": manifest.input_cadence_minutes,
                    "output_cadence_minutes": manifest.output_cadence_minutes,
                    "has_report": (entry / "report.json").is_file(),
                }
            )
        summaries.sort(key=lambda s: str(s["created_at"]), reverse=True)
        return summaries

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, object]:
        run_dir = _safe_run_dir(runs_root, run_id)
        _require_file(run_dir / "manifest.json", "run has no manifest")
        return read_manifest(run_dir).to_dict()

    @app.get("/api/runs/{run_id}/frames/{index}.png")
    def frame_png(run_id: str, index: int) -> FileResponse:
        run_dir = _safe_run_dir(runs_root, run_id)
        if index < 0:
            raise HTTPException(status_code=404, detail="no such frame")
        path = _require_file(run_dir / "frames" / f"{index:03d}.png", "no such frame")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/runs/{run_id}/animations/{kind}.gif")
    def animation(run_id: str, kind: str) -> FileResponse:
        if kind not in ANIMATION_KINDS:
            raise HTTPException(status_code=404, detail=f"unknown animation: {kind}")
        run_dir = _safe_run_dir(runs_root, run_id)
        path = _require_file(run_dir / "animations" / f"{kind}.gif", "animation not generated")
        return FileResponse(path, media_type="image/gif")

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str) -> JSONResponse:
        run_dir = _safe_run_dir(runs_root, run_id)
        path = _require_file(run_dir / "report.json", "no report for this run")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/runs/{run_id}/download")
    def download(run_id: str) -> FileResponse:
        run_dir = _safe_run_dir(runs_root, run_id)
        path = _require_file(run_dir / "output.nc", "no NetCDF product for this run")
        return FileResponse(path, media_type="application/x-netcdf", filename=f"{run_id}.nc")

    # Mounted last on purpose: FastAPI matches routes in order, so mounting "/" first
    # would swallow every /api/... request.
    if static_root.is_dir():
        app.mount("/", StaticFiles(directory=static_root, html=True), name="static")

    return app
