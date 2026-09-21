"""
Bounding-box annotation tool, multi-project.

The server is no longer bound to one dataset. It is bound to a ROOT, and every
request names the project it applies to; the browser holds that choice in two
dropdowns. See ConfigSam3.py for the directory layout.

    project-scoped   annotations, flags     chosen by the project dropdown
    run-scoped       predictions, exemplars chosen by the run dropdown

That split is the whole design. Annotations are irreplaceable human work and a
fact about the dataset, so they must not be filed under a run name that might
be renamed. Predictions belong to exactly one SAM 3 configuration and are
meaningless without it.

Exemplars are run-scoped: the run that produced a given set of predictions has
to be reproducible from exactly the boxes it was prompted with, in order. One
consequence is that you pick exemplars for a run that does not exist yet, so
the run dropdown can create a name (POST /api/run) and the directories appear
on the first write.

Annotation format (one file per image, in <project>/annotations/):
    <class_id> <cx> <cy> <w> <h> <provenance> <quality> <score>

Fields 1-5 are standard YOLO (normalised, class first). Fields 6-8 are the
extra flags; `cut -d' ' -f1-5` gives you strict YOLO. Human boxes carry a
score of 1.0 - they are not predictions, so there is nothing to be uncertain
about, and a fixed 1.0 keeps every line the same shape.

SAM3 predictions are READ ONLY, from <project>/runs/<run>/predictions/, in the
same 8-field format written by the SAM 3 script. This tool never writes there:
predictions are a regenerable artifact of a particular run, annotations are not.

A prediction can be nudged around in the browser and then approved, at which
point it is COPIED into the annotations as a new line carrying its own
provenance token:

    sam3_asis       approved with the geometry SAM3 produced
    sam3_adjusted   approved after the annotator moved or resized it

The prediction file itself is untouched either way, so a rerun of the
prediction script stays reproducible and the annotation file records what a
human actually signed off on. The score column keeps SAM3's confidence for
these lines rather than being flattened to 1.0: the box is now verified (that
is what `quality` says), but the confidence at which a human accepted it is
worth keeping for later analysis of the threshold.

Approved predictions are not marked as such in the predictions directory -
nothing there is writable - so on load the client hides any prediction that
overlaps an existing sam3_* annotation, which is what keeps an approved box
from showing up twice.

Picking an exemplar ALSO writes the box into the annotations as an ordinary
human line. It is a real verified instance and should be trained on; the
manifest, not a provenance token, is what records that it was also used as a
prompt. That keeps `cut -d' ' -f1-5` on the annotations strictly a label set,
with no prompt boxes hiding in it.

Deliberately NOT in the manifest: pad, tile_max_edge and anything else the
consumer decides. They belong to the run config.

Completion flags (one file per image, in <project>/flags/): a single character,
0 or 1. 1 means "every instance in this image is boxed" - so an image with flag
1 and zero boxes is a confirmed negative, which is a useful training example.
A missing flag file means the image has not been signed off.

Run:
    python Annotator.py --config Sam3N10.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from pydantic import BaseModel, field_validator
from importlib.resources import files

from weedloop.config import IMAGE_SUFFIXES, Config, Project, Run, load_config

# --------------------------------------------------------------------------
# Format constants. Everything path-shaped now lives in ConfigSam3.
# --------------------------------------------------------------------------

EXEMPLAR_SCHEMA = 1       # bump when the manifest layout changes incompatibly

CLASS_ID = 0
PROVENANCE = "human"
QUALITY = "verified"
HUMAN_SCORE = 1.0         # written on every human box; see the module docstring

# Provenance vocabulary. The file format is space separated, so these tokens
# must not contain whitespace - hence sam3_adjusted rather than "sam3 adjusted".
PROV_HUMAN = "human"
PROV_SAM3 = "sam3"                    # a raw prediction; never written here
PROV_SAM3_ASIS = "sam3_asis"          # approved unchanged
PROV_SAM3_ADJUSTED = "sam3_adjusted"  # approved after being moved or resized

# What the browser is allowed to PUT into the annotations. PROV_SAM3 is
# deliberately absent: an unreviewed prediction must never land in there, and
# a client bug that tried would get a 422 rather than quietly corrupting the
# training labels.
WRITABLE_PROVENANCE = frozenset({PROV_HUMAN, PROV_SAM3_ASIS, PROV_SAM3_ADJUSTED})
WRITABLE_QUALITY = frozenset({"verified", "unverified"})

# --------------------------------------------------------------------------

app = FastAPI(title="annotator")


@app.middleware("http")
async def strip_proxy_prefix(request, call_next):
    """OOD proxies to /node/<host>/<port>/... and passes the whole path through,
    so the app sees a prefix none of its routes know about."""
    path = request.scope["path"]
    prefix = f"/node/{socket.gethostname()}/{request.app.state.cfg.annotator.port}"
    if path.startswith(prefix):
        request.scope["path"] = path[len(prefix):] or "/"
    return await call_next(request)


# Image dimensions, keyed by (project, filename). The project half matters:
# two datasets may both contain image001.jpg, and a cache keyed on the bare
# name would hand one project the other's width and height, silently writing
# every box at the wrong scale.
_dimensions: dict[tuple[str, str], tuple[int, int]] = {}


# --------------------------------------------------------------------------
# Per-request resolution
# --------------------------------------------------------------------------


def get_cfg(request: Request) -> Config:
    """The config lives on the app, so a handler can ask for it as a
    dependency rather than reaching for a module global."""
    return request.app.state.cfg


CfgDep = Annotated[Config, Depends(get_cfg)]


def get_project(
    cfg: CfgDep,
    project: Annotated[str, Query(description="project (dataset) name")],
) -> Project:
    """Resolve ?project= to a Project, 404 on anything unopenable.

    A missing project is a client-side staleness problem - a bookmark, or a
    dropdown filled before someone renamed a folder - so it must not take the
    server down the way the old startup sys.exit did.
    """
    try:
        return cfg.project(project)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


ProjectDep = Annotated[Project, Depends(get_project)]


def get_run(
    project: ProjectDep,
    run: Annotated[str | None, Query(description="run name")] = None,
) -> Run | None:
    """Resolve ?run= within the already-resolved project.

    Optional: annotating needs no run at all. Absent means "show no
    predictions and no exemplars", which is the honest state when a project
    has never been through SAM 3.

    The run need not exist on disk - exemplars are picked before the run does.
    """
    if not run:
        return None
    try:
        return project.run(run)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


RunDep = Annotated[Run | None, Depends(get_run)]


def image_path(project: Project, name: str) -> Path:
    """Resolve a name to a file inside the project's images/, refusing anything
    outside it.

    Checked by resolving and comparing the parent rather than by membership in
    a scanned list: the list is now per project and would have to be rebuilt or
    invalidated on every switch, and resolve() also survives images/ being a
    symlink into a read-only share.
    """
    p = (project.image_dir / name).resolve()
    if p.parent != project.image_dir.resolve() or not p.is_file():
        raise HTTPException(404, f"unknown image: {name}")
    if p.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(404, f"not an image: {name}")
    return p


def dimensions(project: Project, name: str) -> tuple[int, int]:
    """Width and height read from the file itself, never from metadata."""
    key = (project.name, name)
    if key not in _dimensions:
        with Image.open(image_path(project, name)) as im:
            _dimensions[key] = im.size
    return _dimensions[key]


def annotation_path(project: Project, name: str) -> Path:
    return project.annotations_dir / (Path(name).stem + ".txt")


def flag_path(project: Project, name: str) -> Path:
    return project.flags_dir / (Path(name).stem + ".txt")


def prediction_path(run: Run, name: str) -> Path:
    return run.predictions_dir / (Path(name).stem + ".txt")


# --------------------------------------------------------------------------
# Boxes
# --------------------------------------------------------------------------


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    provenance: str = PROVENANCE
    quality: str = QUALITY
    score: float = HUMAN_SCORE


class Annotation(BaseModel):
    """What the browser PUTs back. Validated; Box on its own is not.

    Box is also the parse target for files on disk, where old or hand-edited
    lines may carry tokens this version does not know about. Rejecting those on
    read would make a file unopenable; rejecting them on write is what actually
    protects the annotations.
    """

    boxes: list[Box]
    complete: bool

    @field_validator("boxes")
    @classmethod
    def _writable_tokens(cls, boxes: list[Box]) -> list[Box]:
        for i, b in enumerate(boxes):
            if b.provenance not in WRITABLE_PROVENANCE:
                raise ValueError(
                    f"box {i}: provenance {b.provenance!r} may not be written "
                    f"to the annotations; allowed: {sorted(WRITABLE_PROVENANCE)}"
                )
            if b.quality not in WRITABLE_QUALITY:
                raise ValueError(
                    f"box {i}: quality {b.quality!r} not in "
                    f"{sorted(WRITABLE_QUALITY)}"
                )
        return boxes


def read_boxes(
    project: Project,
    path: Path,
    name: str,
    default_provenance: str,
    default_quality: str = QUALITY,
) -> list[Box]:
    """Parse one YOLO-plus-flags file into pixel-space boxes.

    Used for both annotations and predictions: the two share a format, so they
    share a parser. Fields 6-8 are optional, which keeps files written before
    the score column was added readable - the defaults differ per caller
    because an unlabelled annotation line is a verified human box while an
    unlabelled prediction line is neither.
    """
    if not path.is_file():
        return []
    width, height = dimensions(project, name)
    boxes: list[Box] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            print(f"  skipping {path.name}:{lineno} - expected 5+ fields")
            continue
        try:
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            print(f"  skipping {path.name}:{lineno} - non-numeric coordinates")
            continue
        try:
            score = float(parts[7]) if len(parts) > 7 else HUMAN_SCORE
        except ValueError:
            score = HUMAN_SCORE
        boxes.append(Box(
            x1=(cx - w / 2) * width,
            y1=(cy - h / 2) * height,
            x2=(cx + w / 2) * width,
            y2=(cy + h / 2) * height,
            provenance=parts[5] if len(parts) > 5 else default_provenance,
            quality=parts[6] if len(parts) > 6 else default_quality,
            score=score,
        ))
    return boxes


def read_annotation(project: Project, name: str) -> list[Box]:
    return read_boxes(project, annotation_path(project, name), name,
                      PROV_HUMAN, "verified")


def read_predictions(project: Project, run: Run | None, name: str) -> list[Box]:
    if run is None:
        return []
    return read_boxes(project, prediction_path(run, name), name,
                      PROV_SAM3, "unverified")


def write_annotation(project: Project, name: str, boxes: list[Box]) -> None:
    """Serialise pixel-space boxes back to normalised YOLO plus flags.

    Written to a temporary file and moved into place, so an interrupted write
    leaves the previous version intact rather than a truncated one.
    """
    width, height = dimensions(project, name)
    lines = []
    for b in boxes:
        x1, x2 = sorted((max(0.0, b.x1), min(float(width), b.x2)))
        y1, y2 = sorted((max(0.0, b.y1), min(float(height), b.y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(
            f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
            f"{b.provenance} {b.quality} {b.score:.4f}"
        )
    project.annotations_dir.mkdir(parents=True, exist_ok=True)
    path = annotation_path(project, name)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
    tmp.replace(path)


# --------------------------------------------------------------------------
# Exemplar sets. One JSON manifest per run; see the module docstring.
# --------------------------------------------------------------------------


class ExemplarEntry(BaseModel):
    """One picked box. Normalised, so the set survives a resize of the images."""

    id: int
    image: str
    cx: float
    cy: float
    w: float
    h: float
    picked: str


class ExemplarSet(BaseModel):
    """The whole manifest, and the schema the SAM3 script reads.

    `image_dir` is what makes a set self-describing: the stems inside it only
    mean something relative to one image folder, and a run that points the set
    at a different dataset should fail rather than silently index the wrong
    pictures. Under the project layout a manifest can no longer be reached from
    the wrong project by accident, but the field still catches a set copied
    between projects by hand.

    `exemplars` is ordered by pick, so "the first n" is a stable, reproducible
    subset of a longer set.
    """

    version: int = EXEMPLAR_SCHEMA
    name: str
    project: str = ""
    image_dir: str
    class_id: int = CLASS_ID
    created: str
    updated: str
    exemplars: list[ExemplarEntry] = []


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_exemplar_set(project: Project, run: Run) -> ExemplarSet:
    path = run.exemplar_manifest
    if not path.is_file():
        stamp = now()
        return ExemplarSet(
            name=run.name, project=project.name,
            image_dir=str(project.image_dir),
            created=stamp, updated=stamp,
        )
    data = json.loads(path.read_text())
    if data.get("version") != EXEMPLAR_SCHEMA:
        raise HTTPException(
            409,
            f"{path.name} is schema {data.get('version')}, this tool writes "
            f"{EXEMPLAR_SCHEMA}",
        )
    return ExemplarSet.model_validate(data)


def write_exemplar_set(run: Run, s: ExemplarSet) -> None:
    run.exemplar_dir.mkdir(parents=True, exist_ok=True)
    path = run.exemplar_manifest
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s.model_dump(), indent=2) + "\n")
    tmp.replace(path)


def exemplar_boxes(
    project: Project, run: Run | None, name: str, s: ExemplarSet | None = None
) -> list[dict]:
    """This image's exemplars, in pixel space, for drawing."""
    if run is None:
        return []
    s = s if s is not None else read_exemplar_set(project, run)
    width, height = dimensions(project, name)
    out = []
    for e in s.exemplars:
        if e.image != name:
            continue
        out.append({
            "id": e.id,
            "x1": (e.cx - e.w / 2) * width,
            "y1": (e.cy - e.h / 2) * height,
            "x2": (e.cx + e.w / 2) * width,
            "y2": (e.cy + e.h / 2) * height,
        })
    return out


def read_flag(project: Project, name: str) -> bool:
    path = flag_path(project, name)
    return path.is_file() and path.read_text().strip() == "1"


def write_flag(project: Project, name: str, complete: bool) -> None:
    project.flags_dir.mkdir(parents=True, exist_ok=True)
    path = flag_path(project, name)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text("1" if complete else "0")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (files("weedloop.annotator") / "static" / "index.html").read_text()


@app.get("/api/projects")
def list_projects(cfg: CfgDep) -> dict:
    """Everything the two dropdowns need, in one call.

    `default_project` comes from the YAML so a fresh browser lands on whatever
    the SAM 3 script is pointed at, which is nearly always the dataset you want.

    `run.name` is deliberately NOT sent. It names the run the next batch job
    writes to, which is a different question from the run you are looking at,
    and a run that does not exist yet is reachable through POST /api/run.
    """
    out = []
    for name in cfg.projects():
        try:
            p = cfg.project(name)
        except (FileNotFoundError, ValueError):
            continue        # a folder that lost its images/ between calls
        # Runs carry their mtime and prediction count so the client can order
        # them and label them without a second round trip. Sorted newest first:
        # list_runs() is alphabetical, which puts sam3_n10 ahead of sam3_n9 and
        # makes "the latest run" whichever name happens to sort highest.
        runs = []
        for run_name in p.list_runs():
            r = p.run(run_name)
            runs.append({
                "name": run_name,
                "modified": r.dir.stat().st_mtime,
                "predictions": (sum(1 for _ in r.predictions_dir.glob("*.txt"))
                                if r.predictions_dir.is_dir() else 0),
            })
        runs.sort(key=lambda d: d["modified"], reverse=True)
        out.append({
            "name": name,
            "runs": runs,
            "images": len(p.image_names()),
        })
    return {
        "projects": out,
        "default_project": cfg.paths.dataset,
        "root": str(cfg.paths.root),
    }


@app.post("/api/run")
def create_run(project: ProjectDep, run: str) -> dict:
    """Make a run directory so exemplars can be picked for a run SAM 3 has not
    executed yet. Idempotent; creating an existing run reports created=False.

    run.json exists because run_manifest.json does not appear until SAM 3
    actually executes. Without it a run made here is indistinguishable from a
    directory someone created by hand, and there would be no record of when the
    exemplar set was started.
    """
    try:
        r = project.run(run)          # _safe_name rejects separators/traversal
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    existed = r.exists
    r.mkdirs()
    if not existed:
        (r.dir / "run.json").write_text(json.dumps({
            "name": r.name,
            "project": project.name,
            "created": now(),
            "created_by": "annotator",
        }, indent=2) + "\n")
    return {"project": project.name, "run": r.name, "created": not existed}


@app.delete("/api/run")
def delete_run(project: ProjectDep, run: str, confirm: str = "") -> dict:
    """Delete a run and everything inside it.

    Safe to expose because of where runs sit: annotations/ and flags/ are
    project-scoped, above runs/, so nothing here can reach the irreplaceable
    half. What goes is predictions, overlays, composites, tiles and the
    exemplar manifest - all regenerable, except the manifest, which is why the
    client makes you type the name when a set has picks in it.

    `confirm` must repeat the run name. The UI already asks, but a DELETE is
    one stray bookmark or curl away otherwise, and this is not recoverable.
    """
    try:
        r = project.run(run)          # _safe_name rejects separators/traversal
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if confirm != r.name:
        raise HTTPException(422, "confirm must repeat the run name")
    if not r.exists:
        raise HTTPException(404, f"no such run: {r.name}")
    if r.dir.is_symlink():
        raise HTTPException(
            422, "the run directory is a symlink; remove it by hand so this "
                 "cannot delete whatever it points at")

    # Belt and braces over _safe_name: the resolved directory has to sit
    # directly inside this project's runs/, or nothing is removed.
    resolved = r.dir.resolve()
    if resolved.parent != project.runs_dir.resolve():
        raise HTTPException(422, f"refusing to delete outside runs/: {resolved}")

    n_files = sum(1 for f in r.dir.rglob("*") if f.is_file())
    n_exemplars = 0
    if r.exemplar_manifest.is_file():
        try:
            n_exemplars = len(json.loads(r.exemplar_manifest.read_text())
                              .get("exemplars", []))
        except (ValueError, OSError):
            n_exemplars = -1          # unreadable; report rather than guess

    shutil.rmtree(r.dir)
    print(f"deleted run {project.name}/{r.name}: {n_files} file(s), "
          f"{n_exemplars} exemplar(s)")
    return {"run": r.name, "files": n_files, "exemplars": n_exemplars}


@app.get("/api/images")
def list_images(project: ProjectDep) -> list[dict]:
    """Every image with just enough state to draw the progress ribbon.

    This walks two files per image. It is the first thing that will need an
    index when a project gets large - see the note at the bottom of the file.
    """
    out = []
    for name in project.image_names():
        path = annotation_path(project, name)
        n = 0
        if path.is_file():
            n = len([ln for ln in path.read_text().splitlines() if ln.strip()])
        out.append({"name": name, "boxes": n, "complete": read_flag(project, name)})
    return out


@app.get("/api/image/{name}")
def get_image(project: ProjectDep, name: str) -> FileResponse:
    return FileResponse(image_path(project, name))


@app.get("/api/annotation/{name}")
def get_annotation(project: ProjectDep, run: RunDep, name: str) -> dict:
    """Annotations and predictions in one response.

    They are kept as separate lists all the way to the browser: the client
    renders them differently, filters only the predictions, and PUTs only the
    annotations back. Merging them here would make it far too easy to write
    predictions into the annotations by accident.

    Approving a prediction is exactly a move from the second list to the first,
    performed in the client and made durable by the next PUT.

    Exemplars are a third list for the same reason, and a stranger one: each
    one has a twin in `boxes`, because picking an exemplar also writes an
    ordinary human annotation. The client draws the exemplar and hides the
    twin, so the box appears once and its role is visible.
    """
    width, height = dimensions(project, name)
    return {
        "width": width,
        "height": height,
        "boxes": [b.model_dump() for b in read_annotation(project, name)],
        "predictions": [b.model_dump() for b in read_predictions(project, run, name)],
        "exemplars": exemplar_boxes(project, run, name),
        "complete": read_flag(project, name),
    }


@app.put("/api/annotation/{name}")
def put_annotation(project: ProjectDep, name: str, annotation: Annotation) -> dict:
    """No run parameter: annotations and flags are project-scoped, and a run
    must never be able to influence where hand labels land."""
    image_path(project, name)  # validates the name
    write_annotation(project, name, annotation.boxes)
    write_flag(project, name, annotation.complete)
    return {"saved": len(annotation.boxes)}


@app.get("/api/exemplars")
def get_exemplar_set(project: ProjectDep, run: RunDep) -> dict:
    """Set-level state for the header: which set is being filled, and how full."""
    if run is None:
        return {"name": None, "count": 0, "image_dir": None, "path": None}
    s = read_exemplar_set(project, run)
    return {
        "name": s.name,
        "count": len(s.exemplars),
        "image_dir": s.image_dir,
        "path": str(run.exemplar_manifest),
    }


@app.post("/api/exemplar/{name}")
def add_exemplar(project: ProjectDep, run: RunDep, name: str, box: Box) -> dict:
    """Append one picked box to the manifest.

    The client posts here first and only writes the annotation once this
    returns, so a rejected pick leaves nothing behind in either place.
    """
    if run is None:
        raise HTTPException(422, "pick a run before picking exemplars")
    image_path(project, name)
    width, height = dimensions(project, name)
    x1, x2 = sorted((max(0.0, box.x1), min(float(width), box.x2)))
    y1, y2 = sorted((max(0.0, box.y1), min(float(height), box.y2)))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise HTTPException(422, "exemplar is smaller than a pixel")

    s = read_exemplar_set(project, run)
    if s.exemplars and s.image_dir != str(project.image_dir):
        raise HTTPException(
            409,
            f"set {s.name!r} was picked against {s.image_dir}; refusing to mix "
            f"in boxes from {project.image_dir}",
        )
    s.image_dir = str(project.image_dir)
    s.project = project.name
    entry = ExemplarEntry(
        id=max((e.id for e in s.exemplars), default=0) + 1,
        image=name,
        cx=(x1 + x2) / 2 / width,
        cy=(y1 + y2) / 2 / height,
        w=(x2 - x1) / width,
        h=(y2 - y1) / height,
        picked=now(),
    )
    s.exemplars.append(entry)
    s.updated = entry.picked
    write_exemplar_set(run, s)
    return {"id": entry.id, "set": s.name, "count": len(s.exemplars)}


@app.delete("/api/exemplar/{exemplar_id}")
def drop_exemplar(project: ProjectDep, run: RunDep, exemplar_id: int) -> dict:
    """Un-pick a box. The annotation it created stays; it is still a real plant."""
    if run is None:
        raise HTTPException(422, "no run selected")
    s = read_exemplar_set(project, run)
    kept = [e for e in s.exemplars if e.id != exemplar_id]
    if len(kept) == len(s.exemplars):
        raise HTTPException(404, f"no exemplar with id {exemplar_id}")
    s.exemplars = kept
    s.updated = now()
    write_exemplar_set(run, s)
    return {"set": s.name, "count": len(s.exemplars)}


# --------------------------------------------------------------------------


def main(cfg: Config):
    app.state.cfg = cfg

    names = cfg.projects()
    print(f"root       =  {cfg.paths.root}")
    if not names:
        print("[warn] no projects found - a project is a folder with an images/ "
              "subdirectory")
    for name in names:
        p = cfg.project(name)
        runs = p.list_runs()
        mark = " <- default" if name == cfg.paths.dataset else ""
        print(f"  {name}: {len(p.image_names())} images, "
              f"{len(runs)} run(s){mark}")
        for r in runs:
            run = p.run(r)
            n_pred = (sum(1 for _ in run.predictions_dir.glob('*.txt'))
                      if run.predictions_dir.is_dir() else 0)
            print(f"      {r}: {n_pred} prediction file(s)")

    if cfg.paths.dataset not in names:
        print(f"[warn] paths.dataset {cfg.paths.dataset!r} is not a project "
              f"under root; the UI will open on whatever you pick instead")

    port = cfg.annotator.port
    print(f"open: https://ondemand.gamarello.agsad.admin.ch"
          f"/node/{socket.gethostname()}/{port}/")
    # One worker on purpose. Any index added later is a single-writer design,
    # and $HOME is NFS.
    uvicorn.run(app, host=cfg.annotator.host, port=port, log_level="warning")


def cli() -> None:
    ap = argparse.ArgumentParser(
        prog="weedloop-annotate",
        description="Serve the annotation UI for any project under paths.root.",
    )
    ap.add_argument("--config", required=True, type=Path,
                    help="path to the YAML settings file")
    args = ap.parse_args()
    main(load_config(args.config))


if __name__ == "__main__":
    cli()
