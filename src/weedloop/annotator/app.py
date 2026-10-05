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
    <class> <cx> <cy> <w> <h> <provenance> <quality> <score>

The class is its NAME, as set in the UI - one word, no spaces - not a number
that means something only through a mapping elsewhere. Fields 2-5 are YOLO
geometry (normalised); fields 6-8 are the extra flags. Older files with
numeric classes still read: "0" is simply a class named "0". Turning this into
a standard format (numeric YOLO, COCO) is an export step, not the storage.
Renaming a class rewrites every annotation file that uses it, after copying
the annotations folder to <project>/backups/. Human boxes carry a
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

Event log (<project>/events/events.jsonl): one JSON object per line, append
only, never rewritten. Every decision made in the editor lands here - boxes
drawn, moved and deleted, predictions approved and REJECTED, exemplars picked
and dropped, sign-offs, undo and redo, and the active time spent per image
visit. The annotation files say what the labels are; the log says how they got
that way, which is what timing statistics, per-run acceptance rates and
review diffs are computed from. The server stamps each line with its own clock
and the OS user it runs as, so the log already has an actor column for when
the tool stops being single-user. Undo does not delete an entry: it appends a
history.undo line naming the entries it reverses.

The project's class list - names, their order (hotkeys 1-9 follow it) and
how each is drawn - lives in <project>/settings.json.

Comments (<project>/comments/<stem>.json): notes pinned to a point on an
image, with author, time and a resolved flag.

Personal settings - input bindings, pointer speeds, and the look of
predictions, exemplars, the cursor and the crosshair - apply to every project
and live in $XDG_CONFIG_HOME/weedloop/settings.json (normally
~/.config/weedloop/). Bindings are stored only where they differ from the
defaults, which are defined in the client next to the commands themselves.

Coverage (<project>/coverage/<stem>.json): how long each part of an image
has been looked at, as milliseconds per cell of a coarse grid over it. The
browser counts while the view is zoomed in far enough and the annotator is
active, and sends what it counted; the server adds it up. This is what the
minimap colours, and what "a full pass was done" can later be checked against.

Stages (<project>/stages.json): where each image is in the project's workflow
- by default No stage, Annotating, In review, Signed off; the list is per
project, in settings.json, each stage with a fixed id, a name, a colour and
instructions for whoever works on it. The LAST stage is "signed off": the
workflow completed. An image there with zero boxes is a confirmed negative, a
useful training example. One file for the project, so a
bulk change is one atomic write; an image without an entry is in the first
stage. Every change goes into the event log, with who made it.

Image flags (<project>/image_flags.json): what is wrong with an image, any
number of motion_blur, out_of_focus, occluded, no_object, not_qualified and
other - with a note saying what "other" is. Saved with the image, like its
stage, and logged.

Comments can be a point, a box drawn on the image (kept with the comments,
never in the annotation file), or attached to a box of yours, a sam3 box or
an exemplar - stored with that box's position, so it follows the box.

It replaces the older flags/ folder (one file per image, 0 or 1): the first
time a project is opened, each flag 1 becomes the last stage. flags/ is
left as it was, and no longer read after that.

Run:
    python Annotator.py --config Sam3N10.yaml
"""

from __future__ import annotations

import argparse
import secrets
import shlex
import signal
import subprocess
import sys
import getpass
import hashlib
import io
import json
import os
import re
import shutil
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from importlib.resources import files

from weedloop.config import (CLASS_NAME_MAX, IMAGE_SUFFIXES, Config, Project, Run, RunSettings,
                             class_token, config_for_run, load_config, read_run_settings,
                             valid_class_name, write_config, write_run_settings)

# --------------------------------------------------------------------------
# Format constants. Everything path-shaped now lives in ConfigSam3.
# --------------------------------------------------------------------------

EXEMPLAR_SCHEMA = 1       # bump when the manifest layout changes incompatibly

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

# Thumbnail edge lengths the grid may ask for. A request is snapped UP to the
# next one, so the on-disk cache holds at most this many copies per image
# however the browser happens to be sized.
THUMB_SIZES = (256, 512)
THUMB_HEADERS = {"Cache-Control": "private, max-age=300"}

# The event vocabulary. Closed on purpose: a typo in the client should be a
# 422 now, not an event type that silently never shows up in any statistic.
EVENT_TYPES = frozenset({
    "image.visit", "image.signoff",
    "box.draw", "box.edit", "box.delete",
    "pred.approve", "pred.adjust", "pred.reject",
    "exemplar.pick", "exemplar.drop",
    "box.class", "box.paste", "image.stage", "image.flags", "run.launch", "run.cancel",
    "comment.add", "comment.edit", "comment.delete", "class.rename",
    "history.undo", "history.redo",
})
MAX_EVENT_DATA = 16_384            # bytes of JSON in one event's data
MAX_EVENT_BATCH = 500
MAX_VISIT_MS = 24 * 3600 * 1000    # anything longer is a clock bug, not work
ACTOR = getpass.getuser()          # OOD runs the app as the person using it

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


# The access token. The OOD login covers the proxy, not the port: anyone who
# can reach the node can open http://<node>:<port>/ directly. So every request
# must carry the token - once in the link printed at start-up, from then on as
# a cookie. Kept in the personal config folder, so restarts keep it valid;
# WEEDLOOP_TOKEN overrides it (tests).
TOKEN_FILE = "token"


def access_token() -> str:
    env = os.environ.get("WEEDLOOP_TOKEN", "").strip()
    if env:
        return env
    path = settings_path().parent / TOKEN_FILE
    try:
        token = path.read_text().strip()
        if len(token) >= 20:
            return token
    except OSError:
        pass
    token = secrets.token_urlsafe(24)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token + "\n")
        path.chmod(0o600)
    except OSError as exc:
        print(f"[warn] could not keep the access token in {path} ({exc}); "
              f"it changes with every start")
    return token


NO_TOKEN_PAGE = """<!doctype html><meta charset="utf-8"><title>annotator</title>
<body style="font: 15px/1.5 system-ui; background: #14161a; color: #e6e6e6; padding: 48px">
<h2 style="font-weight: 600">This annotator needs its access link</h2>
<p>Open the link <code>weedloop-annotate</code> printed when it started - the one ending in
<code>?token=&hellip;</code>. Your browser remembers it after that.</p></body>"""


@app.middleware("http")
async def require_token(request, call_next):
    token = getattr(request.app.state, "token", None)
    if not token:
        return await call_next(request)
    name = f"weedloop_{request.app.state.cfg.annotator.port}"
    given = request.query_params.get("token", "")
    if given and secrets.compare_digest(given, token):
        response = await call_next(request)
        response.set_cookie(name, token, httponly=True, samesite="strict",
                            max_age=90 * 24 * 3600)
        return response
    if secrets.compare_digest(request.cookies.get(name, ""), token):
        return await call_next(request)
    return HTMLResponse(NO_TOKEN_PAGE, status_code=401)


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


DEFAULT_CLASS = "0"        # what an old file without names calls its one class

# What a class name may be, and how a stored class field reads, are shared with
# the SAM 3 script through weedloop.config: valid_class_name(), class_token().


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    cls: str = DEFAULT_CLASS
    provenance: str = PROVENANCE
    quality: str = QUALITY
    score: float = HUMAN_SCORE

    @field_validator("cls")
    @classmethod
    def _class_name(cls, v: str) -> str:
        if not valid_class_name(v):
            raise ValueError(f"not a class name (one word, at most {CLASS_NAME_MAX} "
                             f"characters): {v!r}")
        return v


class Annotation(BaseModel):
    """What the browser PUTs back. Validated; Box on its own is not.

    Box is also the parse target for files on disk, where old or hand-edited
    lines may carry tokens this version does not know about. Rejecting those on
    read would make a file unopenable; rejecting them on write is what actually
    protects the annotations.
    """

    boxes: list[Box]
    # The image's stage, when it changed; None leaves it as it is.
    stage: str | None = None
    # Its flags and the note for "other"; None leaves them as they are.
    flags: list[str] | None = None
    flag_note: str | None = Field(default=None, max_length=500)

    @field_validator("flags")
    @classmethod
    def _known_flags(cls, flags: list[str] | None) -> list[str] | None:
        if flags is not None:
            unknown = [f for f in flags if f not in IMAGE_FLAGS]
            if unknown:
                raise ValueError(f"unknown image flags {unknown}; known: {list(IMAGE_FLAGS)}")
        return flags

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
        # The class is kept, not assumed: a file must come back out with the
        # classes it went in with.
        cls = class_token(parts[0])
        if cls is None:
            print(f"  skipping {path.name}:{lineno} - not a class name {parts[0]!r}")
            continue
        try:
            score = float(parts[7]) if len(parts) > 7 else HUMAN_SCORE
        except ValueError:
            score = HUMAN_SCORE
        boxes.append(Box(
            cls=cls,
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
            f"{b.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
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
    # The class of the box that was picked. The SAM 3 script uses only the
    # exemplars of the class its run predicts. None: picked before this was
    # recorded.
    cls: str | None = None


class ExemplarSet(BaseModel):
    """The whole manifest, and the schema the SAM3 script reads.

    `image_dir` is what makes a set self-describing: the stems inside it only
    mean something relative to one image folder, and a run that points the set
    at a different dataset should fail rather than silently index the wrong
    pictures. Under the project layout a manifest can no longer be reached from
    the wrong project by accident, but the field still catches a set copied
    between projects by hand.

    `exemplars` is ordered by pick, so "the first n" is a stable, reproducible
    subset of a longer set. Each entry records its class; the set-level numeric
    class_id of earlier versions is gone (old files still read: the field is
    ignored, and dropped on the next write).
    """

    version: int = EXEMPLAR_SCHEMA
    name: str
    project: str = ""
    image_dir: str
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
            "cls": e.cls,
        })
    return out


def stage_defs(project: Project) -> list[dict]:
    """The project's stages, from settings.json, or the defaults. Tolerant:
    a file it cannot read gives the defaults, never an error here."""
    try:
        raw = json.loads(project_settings_path(project).read_text(encoding="utf-8"))
        stages = raw.get("stages") if isinstance(raw, dict) else None
        if stages:
            return [StageDef.model_validate(st).model_dump() for st in stages]
    except (OSError, ValueError):
        pass
    return [dict(st) for st in DEFAULT_STAGES]


def stages_path(project: Project) -> Path:
    return project.annotations_dir.parent / "stages.json"


_stage_cache: dict[str, dict[str, dict]] = {}   # stages.json path -> {image: {stage, t, by}}
_stage_lock = threading.Lock()


def _write_stages(path: Path, entries: dict[str, dict]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"version": 1, "images": entries}, indent=0,
                              ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def image_stages(project: Project) -> dict[str, dict]:
    """{image: {stage, t, by}} for the project, read once and then kept.

    Without a stages.json the project still has the old flags/ folder: each
    flag 1 becomes the first done stage (Signed off, by default) and the file
    is written, once. flags/ itself is left as it was."""
    path = stages_path(project)
    key = str(path)
    with _stage_lock:
        if key in _stage_cache:
            return _stage_cache[key]
        entries: dict[str, dict] = {}
        if path.is_file():
            try:
                entries = json.loads(path.read_text(encoding="utf-8")).get("images") or {}
            except (OSError, ValueError) as exc:
                raise HTTPException(500, f"{path.name} could not be read: {exc}") from exc
        else:
            done = stage_defs(project)[-1]["id"]       # the last stage: signed off
            if project.flags_dir.is_dir():
                by_stem = {Path(n).stem: n for n in project.image_names()}
                for f in project.flags_dir.glob("*.txt"):
                    try:
                        if f.stem in by_stem and f.read_text().strip() == "1":
                            entries[by_stem[f.stem]] = {"stage": done, "t": now(),
                                                        "by": "flags/"}
                    except OSError:
                        continue
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_stages(path, entries)
        _stage_cache[key] = entries
        return entries


IMAGE_FLAGS = ("motion_blur", "out_of_focus", "occluded", "no_object", "not_qualified", "other")
_flag_cache: dict[str, dict[str, dict]] = {}   # image_flags.json path -> {image: {flags, note, t, by}}
_flag_lock = threading.Lock()


def image_flags_path(project: Project) -> Path:
    return project.annotations_dir.parent / "image_flags.json"


def image_flag_entries(project: Project) -> dict[str, dict]:
    path = image_flags_path(project)
    with _flag_lock:
        if str(path) not in _flag_cache:
            entries: dict[str, dict] = {}
            if path.is_file():
                try:
                    entries = json.loads(path.read_text(encoding="utf-8")).get("images") or {}
                except (OSError, ValueError) as exc:
                    raise HTTPException(500, f"{path.name} could not be read: {exc}") from exc
            _flag_cache[str(path)] = entries
        return _flag_cache[str(path)]


def flags_of(entries: dict[str, dict], name: str) -> tuple[list[str], str]:
    e = entries.get(name) or {}
    return [f for f in e.get("flags", []) if f in IMAGE_FLAGS], e.get("note", "")


def set_image_flags(project: Project, name: str, flags: list[str], note: str) -> None:
    """Set an image's flags (in the fixed order) and the note for "other", and
    log the change. An image with no flags left has no entry."""
    flags = [f for f in IMAGE_FLAGS if f in set(flags)]
    note = note.strip() if "other" in flags else ""
    entries = image_flag_entries(project)
    before = flags_of(entries, name)
    if before == (flags, note):
        return
    with _flag_lock:
        if flags:
            entries[name] = {"flags": flags, "note": note, "t": now(), "by": ACTOR}
        else:
            entries.pop(name, None)
        path = image_flags_path(project)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"version": 1, "images": entries}, indent=0,
                                  ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    log_server_event(project, "image.flags", name,
                     {"from": before[0], "to": flags, "note": note})


def stage_of(entries: dict[str, dict], name: str, defs: list[dict]) -> str:
    """An image's stage id. No entry, or a stage the project no longer has
    (it was removed from the list): the first stage."""
    sid = (entries.get(name) or {}).get("stage")
    return sid if any(st["id"] == sid for st in defs) else defs[0]["id"]


def set_stages(project: Project, names: list[str], stage: str) -> list[tuple[str, str]]:
    """Move images to a stage, in one write, and log each move. Returns
    [(image, previous stage)] for the images that actually moved."""
    defs = stage_defs(project)
    if not any(st["id"] == stage for st in defs):
        raise HTTPException(422, f"no stage {stage!r} in this project")
    entries = image_stages(project)
    moved = []
    with _stage_lock:
        for name in names:
            before = stage_of(entries, name, defs)
            if before != stage or name not in entries:
                entries[name] = {"stage": stage, "t": now(), "by": ACTOR}
                if before != stage:
                    moved.append((name, before))
        if moved or names:
            _write_stages(stages_path(project), entries)
    if moved:
        stamp, ct = now(), int(datetime.now(timezone.utc).timestamp() * 1000)
        event_log(project).append([{
            "t": stamp, "actor": ACTOR, "project": project.name, "id": uuid.uuid4().hex,
            "type": "image.stage", "ct": ct, "session": "server", "image": name,
            "run": None, "data": {"from": before, "to": stage, "bulk": len(names) > 1},
        } for name, before in moved])
    return moved


# --------------------------------------------------------------------------
# Event log. See the module docstring.
# --------------------------------------------------------------------------


class Event(BaseModel):
    """One event as the browser sends it. The server adds time and actor."""

    id: str = Field(min_length=1, max_length=64)
    type: str
    ct: int = Field(ge=0)                       # client clock, ms since epoch
    session: str = Field(min_length=1, max_length=64)
    image: str | None = Field(default=None, max_length=512)
    run: str | None = Field(default=None, max_length=256)
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in EVENT_TYPES:
            raise ValueError(f"unknown event type {v!r}")
        return v

    @model_validator(mode="after")
    def _bounded(self) -> "Event":
        if len(json.dumps(self.data)) > MAX_EVENT_DATA:
            raise ValueError(f"event {self.id}: data larger than {MAX_EVENT_DATA} bytes")
        if self.type == "image.visit":
            # Visit times are summed into totals, so they are the one payload
            # that has to be numeric and sane rather than merely present.
            for key in ("active_ms", "wall_ms"):
                try:
                    v = int(self.data.get(key, 0))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"event {self.id}: {key} is not a number") from exc
                self.data[key] = min(max(v, 0), MAX_VISIT_MS)
        return self


class EventBatch(BaseModel):
    events: list[Event] = Field(max_length=MAX_EVENT_BATCH)


class EventLog:
    """The append-only log of one project, plus totals derived from it.

    Appends are serialised by a lock because sync endpoints run in a thread
    pool; with one uvicorn worker that makes this the single writer, which is
    what an NFS-backed file needs. The derived totals are a cache: built by one
    pass over the file the first time they are asked for, then kept current by
    folding in each append. The file is the truth; delete nothing from it.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._active: dict[str, int] | None = None     # image -> active ms

    def append(self, records: list[dict]) -> None:
        text = "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n"
                       for r in records)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A crash mid-write can leave a last line without its newline;
            # appending straight after it would fuse two events into one
            # unparseable line. Start on a fresh line instead.
            if self.path.is_file() and self.path.stat().st_size:
                with self.path.open("rb") as f:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        text = "\n" + text
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text)
            if self._active is not None:
                for r in records:
                    self._fold(r)

    def active_ms(self, image: str) -> int:
        with self._lock:
            if self._active is None:
                self._active = {}
                for r in self._scan():
                    self._fold(r)
            return self._active.get(image, 0)

    def _scan(self):
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    print(f"  skipping {self.path.name}:{lineno} - not JSON")

    def _fold(self, r: dict) -> None:
        if r.get("type") != "image.visit" or not r.get("image"):
            return
        try:
            ms = int(r["data"]["active_ms"])
        except (KeyError, TypeError, ValueError):
            return
        self._active[r["image"]] = self._active.get(r["image"], 0) + max(ms, 0)


# --------------------------------------------------------------------------
# Coverage. See the module docstring.
# --------------------------------------------------------------------------

MAX_COVERAGE_CELLS = 128             # per side
MAX_CELL_MS = 24 * 3600 * 1000


class CoverageDelta(BaseModel):
    """Milliseconds looked at, per cell, since the browser last sent any."""

    gw: int = Field(ge=1, le=MAX_COVERAGE_CELLS)
    gh: int = Field(ge=1, le=MAX_COVERAGE_CELLS)
    ms: list[float]

    @model_validator(mode="after")
    def _shape(self) -> "CoverageDelta":
        if len(self.ms) != self.gw * self.gh:
            raise ValueError(f"{len(self.ms)} cells for a {self.gw} x {self.gh} grid")
        if any(not 0 <= v <= MAX_CELL_MS for v in self.ms):
            raise ValueError("cell times must be between 0 and 24 h")
        return self


def coverage_path(project: Project, name: str) -> Path:
    return project.annotations_dir.parent / "coverage" / (Path(name).stem + ".json")


def read_coverage(project: Project, name: str) -> dict | None:
    """The stored grid, or None. A damaged file reads as no coverage rather
    than as an error: it is a record of attention, not of labels."""
    path = coverage_path(project, name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        gw, gh, ms = int(data["gw"]), int(data["gh"]), data["ms"]
        if len(ms) != gw * gh:
            raise ValueError("cell count does not match the grid")
        return {"gw": gw, "gh": gh, "ms": [float(v) for v in ms]}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"  ignoring {path.name}: {exc}")
        return None


_coverage_lock = threading.Lock()


_event_logs: dict[str, EventLog] = {}
_event_logs_lock = threading.Lock()


def event_log(project: Project) -> EventLog:
    """One EventLog per file, so every request shares its lock and cache."""
    path = project.annotations_dir.parent / "events" / "events.jsonl"
    with _event_logs_lock:
        log = _event_logs.get(str(path))
        if log is None:
            log = _event_logs[str(path)] = EventLog(path)
        return log


# --------------------------------------------------------------------------
# Personal settings: input bindings and pointer speeds
# --------------------------------------------------------------------------

SETTINGS_SCHEMA = 1
_COMMAND_ID = re.compile(r"^[a-z][A-Za-z0-9]*(\.[A-Za-z0-9]+)+$")
MAX_BOUND_COMMANDS = 256
MAX_INPUTS_PER_COMMAND = 8
MAX_INPUT_LEN = 48
SPEED_MIN, SPEED_MAX = 0.1, 3.0


def home_dir() -> Path:
    """The user's home, even when $HOME is empty.

    A SLURM job session can start with $HOME unset or empty, and an empty
    $HOME makes "~" resolve to the working directory - settings would land
    wherever the server happened to be started. The password database knows
    better, so it is asked whenever the environment does not say.
    """
    home = os.environ.get("HOME", "").strip()
    if home:
        return Path(home)
    try:
        import pwd                  # POSIX only; Windows falls through
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):
        return Path.home()


def config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg and os.path.isabs(xdg):  # the spec says to ignore a relative one
        return Path(xdg) / "weedloop"
    return home_dir() / ".config" / "weedloop"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def legacy_keybindings_path() -> Path:
    return config_dir() / "keybindings.json"


def display_path(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(home_dir()))
    except ValueError:
        return str(p)


class PointerSettings(BaseModel):
    """How far the image moves per pixel of drag and how much one wheel notch
    zooms (multipliers of the built-in behaviour), and how boxes are drawn."""

    pan_gain: float = Field(default=1.0, ge=SPEED_MIN, le=SPEED_MAX)
    zoom_speed: float = Field(default=1.0, ge=SPEED_MIN, le=SPEED_MAX)
    # "clicks": click to start a box, click again to finish it, no button held
    # in between. "drag": press, drag, release.
    draw: Literal["clicks", "drag"] = "clicks"
    key_pan: float = Field(default=700, ge=100, le=4000)    # screen px/s, arrow keys


class MinimapSettings(BaseModel):
    """The minimap's look and place, and from what zoom looking at a part of
    the image counts: at least `min_zoom` times the fit-to-screen zoom."""

    show: bool = True
    size: int = Field(default=220, ge=100, le=480)          # width, screen px
    opacity: float = Field(default=0.85, ge=0.2, le=1.0)
    right: float = Field(default=12, ge=0, le=10000)        # top-right corner, px
    top: float = Field(default=12, ge=0, le=10000)          # from the view's
    min_zoom: float = Field(default=2.0, ge=1.0, le=8.0)


class ScanSettings(BaseModel):
    """Automatic scanning: how fast the view slides (screen px per second,
    one of the levels the client offers), how long it holds the image's first
    and last view, how much two rows overlap, how long after the last touch a
    paused scan goes on, and whether it moves on to the next image."""

    speed: float = Field(default=130, ge=10, le=2000)
    edge_s: float = Field(default=1.0, ge=0, le=10)
    overlap: float = Field(default=0.2, ge=0, le=0.5)
    resume_s: float = Field(default=2.0, ge=0.5, le=10)
    next_image: bool = False


class StripSettings(BaseModel):
    """The strip of next images under the editor: shown or not, and how tall
    its thumbnails are, in screen px."""

    show: bool = True
    height: int = Field(default=84, ge=56, le=200)


class PersonalSettings(BaseModel):
    """What the settings dialog saves.

    `bindings` holds overrides only - a command id mapped to its full list of
    inputs, an empty list meaning "nothing bound". Commands you never touched
    are absent and keep following the defaults, including when a later
    version changes them. Ids are checked for shape only: a file written by a
    newer version may name commands this one does not have, and those are
    kept rather than dropped.
    """

    version: int = SETTINGS_SCHEMA
    bindings: dict[str, list[str]] = Field(default_factory=dict)
    pointer: PointerSettings = Field(default_factory=PointerSettings)
    styles: "PersonalStyles" = Field(default_factory=lambda: PersonalStyles())
    minimap: MinimapSettings = Field(default_factory=MinimapSettings)
    scan: ScanSettings = Field(default_factory=ScanSettings)
    strip: StripSettings = Field(default_factory=StripSettings)

    @field_validator("version")
    @classmethod
    def _known_version(cls, v: int) -> int:
        if v != SETTINGS_SCHEMA:
            raise ValueError(f"settings schema {v}; this version reads {SETTINGS_SCHEMA}")
        return v

    @field_validator("bindings")
    @classmethod
    def _sane(cls, bindings: dict[str, list[str]]) -> dict[str, list[str]]:
        if len(bindings) > MAX_BOUND_COMMANDS:
            raise ValueError(f"more than {MAX_BOUND_COMMANDS} commands")
        for cmd, inputs in bindings.items():
            if not _COMMAND_ID.match(cmd):
                raise ValueError(f"not a command id: {cmd!r}")
            if len(inputs) > MAX_INPUTS_PER_COMMAND:
                raise ValueError(f"{cmd}: more than {MAX_INPUTS_PER_COMMAND} inputs")
            for i in inputs:
                if not 0 < len(i) <= MAX_INPUT_LEN or any(ord(ch) < 32 for ch in i):
                    raise ValueError(f"{cmd}: not an input name: {i!r}")
        return bindings


def from_legacy_keybindings(path: Path) -> PersonalSettings:
    """Read keybindings.json, the first format: {"overrides": ..., "pan":
    "middle" | "right" | "either"}. Panning is a command now, so the pan
    choice becomes that command's binding."""
    data = json.loads(path.read_text(encoding="utf-8"))
    bindings = dict(data.get("overrides") or {})
    pan = {"right": ["MouseRight"],
           "either": ["MouseMiddle", "MouseRight"]}.get(data.get("pan"))
    if pan and "view.pan" not in bindings:
        bindings["view.pan"] = pan
    return PersonalSettings(bindings=bindings)


_settings_lock = threading.Lock()


# --------------------------------------------------------------------------
# Box, cursor and crosshair styles
# --------------------------------------------------------------------------

_COLOR = r"^#[0-9a-fA-F]{6}$"


class LineStyle(BaseModel):
    """A stroke, in screen pixels: the same on screen at any zoom."""

    color: str = Field(pattern=_COLOR)
    width: float = Field(ge=0.5, le=12)
    opacity: float = Field(ge=0.05, le=1)
    line: Literal["solid", "dashed", "dotted"]


class BoxStyle(LineStyle):
    fill: float = Field(ge=0, le=1)          # opacity of the fill, in the same colour


class CrosshairStyle(LineStyle):
    show: bool


class CursorStyle(LineStyle):
    show: bool                               # false: the system crosshair cursor
    size: float = Field(ge=4, le=80)


class PersonalStyles(BaseModel):
    """Overrides only; a style that is absent follows the default in the client.

    Unknown keys are kept, not dropped, so a file from a newer version survives
    being saved by this one."""

    model_config = ConfigDict(extra="allow")

    prediction: BoxStyle | None = None
    approved: BoxStyle | None = None
    exemplar: BoxStyle | None = None
    cursor: CursorStyle | None = None
    crosshair: CrosshairStyle | None = None


PROJECT_SETTINGS_SCHEMA = 2

# The stages a project starts with. The ids are what stages.json and the
# events store: renaming a stage keeps its id, so nothing needs rewriting.
DEFAULT_STAGES = [
    {"id": "none", "name": "No stage", "color": "#4a5460", "done": False},
    {"id": "annotating", "name": "Annotating", "color": "#8a7fd4", "done": False},
    {"id": "review", "name": "In review", "color": "#e0a13a", "done": False},
    {"id": "signed_off", "name": "Signed off", "color": "#4bd6a4", "done": True},
]
_STAGE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class StageDef(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=40)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    # What to do in this stage, shown to whoever works on an image in it.
    instructions: str = Field(default="", max_length=4000)
    # Read from older files and ignored: the last stage is the finished one.
    done: bool = False

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _STAGE_ID.match(v):
            raise ValueError(f"a stage id is lower-case letters, digits, _ and -: {v!r}")
        return v

    @field_validator("name")
    @classmethod
    def _printable(cls, v: str) -> str:
        if any(ord(ch) < 32 for ch in v) or not v.strip():
            raise ValueError("a stage name is printable and not blank")
        return v.strip()


class ClassDef(BaseModel):
    name: str
    style: BoxStyle | None = None        # None: the palette default

    @field_validator("name")
    @classmethod
    def _one_word(cls, v: str) -> str:
        if not valid_class_name(v):
            raise ValueError(f"class names are one word, at most {CLASS_NAME_MAX} "
                             f"characters: {v!r}")
        return v


class ProjectSettings(BaseModel):
    """What belongs to one project rather than to one person: its classes, in
    order (hotkeys 1-9 follow the order), and how each is drawn."""

    version: int = PROJECT_SETTINGS_SCHEMA
    classes: list[ClassDef] = Field(default_factory=list)
    # The workflow, in order; the first stage is where an image starts. None:
    # the defaults. A workflow canvas will later connect these by id.
    stages: list[StageDef] | None = None

    @field_validator("stages")
    @classmethod
    def _stage_list(cls, stages: list[StageDef] | None) -> list[StageDef] | None:
        if stages is None:
            return None
        if not stages:
            raise ValueError("a project needs at least one stage")
        ids = [st.id for st in stages]
        if len(set(ids)) != len(ids):
            raise ValueError("stage ids must be unique")
        names = [st.name.lower() for st in stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        return stages

    @field_validator("version")
    @classmethod
    def _known_version(cls, v: int) -> int:
        if v != PROJECT_SETTINGS_SCHEMA:
            raise ValueError(f"project settings schema {v}; this version reads "
                             f"{PROJECT_SETTINGS_SCHEMA}")
        return v

    @field_validator("classes")
    @classmethod
    def _unique(cls, classes: list[ClassDef]) -> list[ClassDef]:
        if len(classes) > 1000:
            raise ValueError("more than 1000 classes")
        names = [c.name for c in classes]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"class names must be unique: {sorted(dup)}")
        return classes


def project_settings_path(project: Project) -> Path:
    return project.annotations_dir.parent / "settings.json"


def write_settings_file(path: Path, model: BaseModel, parse) -> None:
    """Atomic write, keeping aside a file that exists but cannot be read - it
    may hold hand edits worth recovering. Callers hold the settings lock."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            parse(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            path.replace(path.with_name(f"{path.stem}.unreadable-{stamp}.json"))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(model.model_dump_json(indent=2, exclude_none=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


# Class names that occur in a project's annotations, so the class list can
# show every class there is, not only those in images opened so far. One scan
# per project and server run; every save adds its classes to the result.
_seen_classes: dict[str, set[str]] = {}
_seen_lock = threading.Lock()

# Whole-project rewrites (a class renamed) and single saves must not
# interleave, or a save could land between a file's read and its rewrite.
_annotation_lock = threading.Lock()


def class_counts(project: Project) -> dict[str, int]:
    """Boxes per class name over all annotation files. A full scan."""
    counts: dict[str, int] = {}
    if project.annotations_dir.is_dir():
        for f in project.annotations_dir.glob("*.txt"):
            try:
                text = f.read_text()
            except OSError:
                continue
            for line in text.splitlines():
                head = line.split(maxsplit=1)
                if len(head) < 2:
                    continue
                name = class_token(head[0])
                if name is not None:
                    counts[name] = counts.get(name, 0) + 1
    return counts


def seen_classes(project: Project) -> set[str]:
    key = str(project.annotations_dir)
    with _seen_lock:
        if key not in _seen_classes:
            _seen_classes[key] = set(class_counts(project))
        return set(_seen_classes[key])


def note_classes(project: Project, boxes: list[Box]) -> None:
    with _seen_lock:
        known = _seen_classes.get(str(project.annotations_dir))
        if known is not None:
            known.update(b.cls for b in boxes)


def backup_annotations(project: Project, reason: str) -> Path | None:
    """Copy the whole annotations folder aside before a rewrite that touches
    many files. They are small text files; the hand work in them is not."""
    src = project.annotations_dir
    if not src.is_dir():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = project.annotations_dir.parent / "backups" / f"annotations-{stamp}-{reason}"
    shutil.copytree(src, dst)
    return dst


def rename_classes_in_files(project: Project, mapping: dict[str, str]) -> tuple[int, int]:
    """Rewrite the class field of every annotation line whose class is a key
    of `mapping`. Only that field changes; the rest of each line is kept
    byte for byte. Returns (files changed, lines changed)."""
    files = lines = 0
    if not project.annotations_dir.is_dir():
        return 0, 0
    for f in sorted(project.annotations_dir.glob("*.txt")):
        try:
            text = f.read_text()
        except OSError:
            continue
        out, changed = [], 0
        for line in text.splitlines():
            head = line.split(maxsplit=1)
            name = class_token(head[0]) if head else None
            if name in mapping and len(head) == 2:
                out.append(f"{mapping[name]} {head[1]}")
                changed += 1
            else:
                out.append(line)
        if changed:
            tmp = f.with_name(f".{f.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""))
            tmp.replace(f)
            files += 1
            lines += changed
    return files, lines


def migrate_project_settings_v1(project: Project, data: dict) -> tuple[ProjectSettings, int]:
    """v1 kept a display name per numeric class id; since v2 the annotation
    files carry the name itself. Ids that had a name are renamed in every
    annotation file (after a backup); ids without one keep their number as
    their name. Spaces in old names become underscores - a name is one word."""
    classes: list[ClassDef] = []
    mapping: dict[str, str] = {}
    for key in sorted(data.get("classes") or {}, key=lambda k: (not k.isdigit(), k.zfill(8))):
        entry = (data["classes"].get(key) or {})
        name = re.sub(r"\s+", "_", (entry.get("name") or "").strip()) or key
        if not valid_class_name(name) or name in [c.name for c in classes]:
            name = key
        style = entry.get("style")
        try:
            style = BoxStyle.model_validate(style) if style else None
        except ValueError:
            style = None
        classes.append(ClassDef(name=name, style=style))
        if name != key:
            mapping[key] = name
    files = 0
    if mapping:
        with _annotation_lock:
            backup_annotations(project, "before-class-names")
            files, _ = rename_classes_in_files(project, mapping)
        with _seen_lock:
            _seen_classes.pop(str(project.annotations_dir), None)
    return ProjectSettings(classes=classes), files


# PersonalSettings names PersonalStyles before it is defined; resolve it now
# rather than on first use.
PersonalSettings.model_rebuild()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (files("weedloop.annotator") / "static" / "index.html").read_text()


@app.get("/api/projects")
def list_projects(cfg: CfgDep) -> dict:
    """Everything the project page and the run dropdown need, in one call.

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
        names = p.image_names()
        defs = stage_defs(p)
        entries = image_stages(p)
        counts = {st["id"]: 0 for st in defs}
        for n in names:
            counts[stage_of(entries, n, defs)] += 1
        out.append({
            "name": name,
            "runs": runs,
            "images": len(names),
            "boxes": total_boxes(p),
            "stages": defs,
            "stage_counts": counts,
            "covers": spread(names, 3),
        })
    return {
        "projects": out,
        "default_project": cfg.paths.dataset,
        "root": str(cfg.paths.root),
    }


_box_totals: dict[str, tuple[float, int]] = {}


def total_boxes(project: Project) -> int:
    """Boxes in all of a project's annotation files, for the projects page.
    Every save replaces its file through a rename, which touches the folder's
    mtime - so the count is redone only when something was saved."""
    d = project.annotations_dir
    try:
        mtime = d.stat().st_mtime
    except OSError:
        return 0
    hit = _box_totals.get(str(d))
    if hit and hit[0] == mtime:
        return hit[1]
    n = 0
    for f in d.glob("*.txt"):
        try:
            n += sum(1 for ln in f.read_text().splitlines() if ln.strip())
        except OSError:
            continue
    _box_totals[str(d)] = (mtime, n)
    return n


def spread(names: list[str], k: int) -> list[str]:
    """Up to k names spread evenly from first to last, for a project card that
    shows the set rather than its first few frames."""
    if len(names) <= k:
        return list(names)
    return [names[round(i * (len(names) - 1) / (k - 1))] for i in range(k)]


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


# --------------------------------------------------------------------------
# A run's settings, and launching it. See config.RunSettings: the YAML holds
# the defaults, each run its own copy (runs/<run>/settings.yaml); a launch
# writes the complete config (config.yaml) and starts the SAM 3 script with it
# - in this session (launcher.mode: local) or as a Slurm job (slurm). The
# launch is recorded in job.json; the script reports into progress.json and
# logs/<job>.log, which is all the status below reads, so a restarted
# annotator still sees a run it started.
# --------------------------------------------------------------------------

def _need_run(run: Run | None) -> Run:
    if run is None:
        raise HTTPException(422, "no run selected")
    return run


def _ensure_run_dir(project: Project, r: Run) -> None:
    existed = r.exists
    r.mkdirs()
    if not existed:
        (r.dir / "run.json").write_text(json.dumps({
            "name": r.name, "project": project.name,
            "created": now(), "created_by": "annotator"}, indent=2) + "\n")


@app.get("/api/run/settings")
def get_run_settings(project: ProjectDep, run: RunDep, cfg: CfgDep, defaults: bool = False) -> dict:
    """The run's settings - its own if saved, else the YAML's - or, with
    defaults=true, the YAML's whatever the run has."""
    r = _need_run(run)
    try:
        settings, own = ((RunSettings.from_config(cfg), False) if defaults
                         else read_run_settings(cfg, r))
    except (ValueError, OSError) as exc:
        raise HTTPException(422, f"{r.settings_path.name} could not be read: {exc}") from exc
    return {"settings": settings.model_dump(mode="json"), "own": own,
            "classes_known": project.class_names(), "gt": project.gt_dir is not None,
            "launcher": cfg.launcher.mode}


@app.put("/api/run/settings")
def put_run_settings(project: ProjectDep, run: RunDep, settings: RunSettings) -> dict:
    r = _need_run(run)
    _ensure_run_dir(project, r)
    write_run_settings(r, settings)
    return {"settings": settings.model_dump(mode="json"), "own": True}


@app.post("/api/run/duplicate")
def duplicate_run(project: ProjectDep, run: RunDep, cfg: CfgDep, to: str) -> dict:
    """A new run with this one's settings and exemplars - a sweep is a
    duplicate with one setting changed. Nothing of the outputs is copied."""
    src = _need_run(run)
    try:
        dst = project.run(to)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if dst.exists:
        raise HTTPException(409, f"there is already a run called {dst.name}")
    settings, _ = read_run_settings(cfg, src)
    _ensure_run_dir(project, dst)
    write_run_settings(dst, settings)
    copied = 0
    if src.exemplar_manifest.is_file():
        s = read_exemplar_set(project, src)
        s.name, s.created, s.updated = dst.name, now(), now()
        write_exemplar_set(dst, s)
        copied = len(s.exemplars)
    return {"run": dst.name, "exemplars": copied}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


_procs: dict[str, subprocess.Popen] = {}       # launches of this server: job id -> process
_launch_lock = threading.Lock()


def _pid_alive(pid: int, marker: str) -> bool:
    """Is the process still there, and still the one launched (not a new
    process that got the same id)?"""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return marker in cmd
    except OSError:
        return True


def _run_cmd(args: list[str], env=None, timeout=20) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _slurm_state(job_id: str) -> str | None:
    out = _run_cmd(["squeue", "-h", "-j", job_id, "-o", "%T"])
    if out and out.returncode == 0 and out.stdout.strip():
        return out.stdout.split()[0]
    out = _run_cmd(["sacct", "-n", "-X", "-P", "-j", job_id, "-o", "State"])
    if out and out.returncode == 0 and out.stdout.strip():
        return out.stdout.split()[0]
    return None


def job_state(run: Run, job: dict) -> str:
    """queued, running, cancelling, done, failed, cancelled - or stopped: it
    ended without saying how (killed, or the session it ran in ended)."""
    progress = _read_json(run.progress_path) or {}
    cancelled = bool(job.get("cancelled"))
    if job.get("mode") == "slurm":
        st = _slurm_state(str(job.get("slurm_id", "")))
        if st in ("PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"):
            return "cancelling" if cancelled else "queued"
        if st in ("RUNNING", "COMPLETING", "STAGE_OUT"):
            return "cancelling" if cancelled else "running"
        if st and st.startswith("CANCELLED"):
            return "cancelled"
        if st in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"):
            return "failed"
    else:
        p = _procs.get(job.get("id", ""))
        alive = p.poll() is None if p else _pid_alive(int(job.get("pid", 0)), str(run.config_path))
        if alive:
            return "cancelling" if cancelled else "running"
    if cancelled:
        return "cancelled"
    phase = progress.get("phase")
    if phase == "done":
        return "done"
    if phase == "failed":
        return "failed"
    return "stopped"


def _log_tail(path: Path, n: int = 80) -> list[str]:
    """The end of a launch's output. Progress bars rewrite their line with
    carriage returns; each rewrite counts as a line here."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64_000))
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    lines = [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    # Of a run of progress-bar rewrites ("images:  25%|##  | 1/4 [...]"), only
    # the latest is worth showing.
    bar = re.compile(r"^(.*?):\s*\d+%\|")
    out: list[str] = []
    for ln in lines:
        m, prev = bar.match(ln), bar.match(out[-1]) if out else None
        if m and prev and m.group(1) == prev.group(1):
            out[-1] = ln
        else:
            out.append(ln)
    return out[-n:]


_alloc_cache: dict[str, tuple[float, dict | None]] = {}


def allocation() -> dict | None:
    """The Slurm allocation this annotator runs in (the salloc session), and
    how long it has left - a local launch ends with it."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    hit = _alloc_cache.get(job)
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    info = {"job": job, "left": None, "left_s": None}
    out = _run_cmd(["squeue", "-h", "-j", job, "-o", "%L"])
    if out and out.returncode == 0 and out.stdout.strip():
        left = out.stdout.split()[0]
        info["left"] = left
        m = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", left)
        if m:
            d, hh, mm, ss = (int(v or 0) for v in m.groups())
            info["left_s"] = ((d * 24 + hh) * 60 + mm) * 60 + ss
    _alloc_cache[job] = (time.time(), info)
    return info


def run_status(run: Run, cfg: Config) -> dict:
    job = _read_json(run.job_path)
    state = job_state(run, job) if job else None
    return {
        "job": job, "state": state,
        "progress": _read_json(run.progress_path) if job else None,
        "log": _log_tail(run.logs_dir / job["log"]) if job and job.get("log") else [],
        "mode": cfg.launcher.mode,
        "allocation": allocation() if cfg.launcher.mode == "local" else None,
    }


@app.get("/api/run/job")
def get_run_job(run: RunDep, cfg: CfgDep) -> dict:
    return run_status(_need_run(run), cfg)


def preflight(project: Project, run: Run, settings: RunSettings) -> tuple[list[str], list[str]]:
    """What stops a launch, and what is only worth a warning."""
    problems, warnings = [], []
    classes = settings.inference.run_classes()
    if not classes:
        problems.append("no class to predict: add one in the run's settings")
    if settings.exemplars.source == "manifest":
        entries = read_exemplar_set(project, run).exemplars if run.exemplar_manifest.is_file() else []
        for c in classes:
            n = sum(1 for e in entries
                    if e.cls == c.name or (len(classes) == 1 and e.cls is None))
            if not n:
                problems.append(f"no exemplars picked as {c.name}: pick some (e, then a box), "
                                f"or sample gt/ boxes instead")
    elif project.gt_dir is None:
        problems.append(f"exemplars from gt/, but {project.name} has no gt/ folder")
    known = project.class_names()
    if known is not None:
        for c in classes:
            if c.name not in known:
                warnings.append(f"{c.name} is not on the project's class list: approved "
                                f"predictions would take the class being drawn with")
    return problems, warnings


def slurm_script(cfg: Config, run: Run, cmd: list[str], log: Path) -> str:
    sc = cfg.launcher.slurm
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"weedloop-{run.name}")
    lines = ["#!/bin/bash", f"#SBATCH --job-name={name}", f"#SBATCH --output={log}",
             f"#SBATCH --cpus-per-task={sc.cpus}", f"#SBATCH --mem={sc.mem}",
             f"#SBATCH --time={sc.time}"]
    if sc.partition:
        lines.append(f"#SBATCH --partition={sc.partition}")
    if sc.gres:
        lines.append(f"#SBATCH --gres={sc.gres}")
    if sc.account:
        lines.append(f"#SBATCH --account={sc.account}")
    lines += [f"#SBATCH {x}" for x in sc.extra]
    lines += [
        "",
        "# Written by the annotator for this launch; see the run's config.yaml.",
        "set -eo pipefail",
        f'export HOME="${{HOME:-{Path.home()}}}"        # batch jobs can start without one',
        "unset SLURM_MEM_PER_CPU",
        "export PYTHONUNBUFFERED=1",
        *sc.setup,
        "exec " + " ".join(shlex.quote(c) for c in cmd),
        "",
    ]
    return "\n".join(lines)


@app.post("/api/run/launch")
def launch_run(project: ProjectDep, run: RunDep, cfg: CfgDep, confirm: bool = False) -> dict:
    """Start the run. Refused while it is already running, when something
    would make it fail at once, and - unless confirmed - when it would
    overwrite predictions it already has."""
    r = _need_run(run)
    with _launch_lock:
        old = _read_json(r.job_path)
        if old and job_state(r, old) in ("queued", "running", "cancelling"):
            raise HTTPException(409, f"{r.name} is still running")
        try:
            settings, _ = read_run_settings(cfg, r)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, f"{r.settings_path.name} could not be read: {exc}") from exc
        problems, warnings = preflight(project, r, settings)
        if problems:
            raise HTTPException(422, problems)
        n_pred = sum(1 for _ in r.predictions_dir.glob("*.txt")) if r.predictions_dir.is_dir() else 0
        if settings.overwrite and n_pred and not confirm:
            raise HTTPException(409, {"confirm": f"{r.name} has predictions for {n_pred} "
                                                f"image(s); launching replaces them"})
        _ensure_run_dir(project, r)
        write_config(config_for_run(cfg, project, r, settings), r.config_path,
                     note=f"launched {now()} by {ACTOR}")
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        r.logs_dir.mkdir(parents=True, exist_ok=True)
        log = r.logs_dir / f"{job_id}.log"
        _write_json(r.progress_path, {"phase": "queued", "done": 0, "total": 0,
                                      "updated": time.time()})
        cmd = [*(cfg.launcher.command or [sys.executable, "-m", "weedloop.predictor.predict"]),
               "--config", str(r.config_path)]
        job = {"id": job_id, "mode": cfg.launcher.mode, "log": log.name, "started": now(),
               "by": ACTOR, "command": cmd}
        if cfg.launcher.mode == "local":
            with log.open("ab") as out:
                p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, cwd=str(r.dir),
                                     start_new_session=True,      # survives the annotator
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
            _procs[job_id] = p
            job["pid"] = p.pid
        else:
            script = r.dir / "job.sh"
            script.write_text(slurm_script(cfg, r, cmd, log))
            # sbatch from inside a Slurm session would take that session's
            # settings with it (memory, cpus, ...): the job gets a clean slate.
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("SLURM_", "SBATCH_", "SALLOC_", "SRUN_"))}
            out = _run_cmd(["sbatch", "--parsable", str(script)], env=env, timeout=60)
            if out is None or out.returncode != 0:
                msg = out.stderr.strip() if out else "sbatch could not be run"
                _write_json(r.progress_path, {"phase": "failed", "message": msg, "updated": time.time()})
                raise HTTPException(502, f"sbatch refused the job: {msg}")
            job["slurm_id"] = out.stdout.strip().split(";")[0]
        _write_json(r.job_path, job)
    log_server_event(project, "run.launch", None,
                     {"run": r.name, "job": job_id, "mode": job["mode"],
                      "slurm_id": job.get("slurm_id"),
                      "classes": [c.name for c in settings.inference.run_classes()]})
    return {**run_status(r, cfg), "warnings": warnings}


@app.post("/api/run/cancel")
def cancel_run(project: ProjectDep, run: RunDep, cfg: CfgDep) -> dict:
    r = _need_run(run)
    job = _read_json(r.job_path)
    if not job or job_state(r, job) not in ("queued", "running"):
        raise HTTPException(409, f"{r.name} is not running")
    if job["mode"] == "slurm":
        out = _run_cmd(["scancel", str(job["slurm_id"])])
        if out is None or out.returncode != 0:
            raise HTTPException(502, f"scancel failed: {out.stderr.strip() if out else 'not found'}")
    else:
        try:
            os.killpg(int(job["pid"]), signal.SIGTERM)   # its own session: the whole run
        except (ProcessLookupError, PermissionError):
            pass
    job["cancelled"] = now()
    _write_json(r.job_path, job)
    log_server_event(project, "run.cancel", None, {"run": r.name, "job": job["id"]})
    time.sleep(0.3)
    return run_status(r, cfg)


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
    job = _read_json(r.job_path) if r.exists else None
    if job and job_state(r, job) in ("queued", "running", "cancelling"):
        raise HTTPException(409, f"{r.name} is running; cancel it first")
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
def list_images(project: ProjectDep, run: str | None = None, min_score: float = 0.0) -> list[dict]:
    """Every image with the state the grid, the image strip and the grid's
    content filter and sort need: file size, boxes in total and per class, the stage, open
    comments - and, for `run`, its SAM 3 boxes scoring at least `min_score`
    (what the run proposed, before any approval or rejection) and its
    exemplars.

    This walks every annotation file, and the run's prediction files. It is
    the first thing that will need an index when a project gets large - see
    the note at the bottom of the file.
    """
    entries, defs = image_stages(project), stage_defs(project)
    flag_entries = image_flag_entries(project)
    r = project.run(run) if run else None

    preds: dict[str, int] = {}
    if r is not None and r.predictions_dir.is_dir():
        for f in r.predictions_dir.glob("*.txt"):
            n = 0
            try:
                for line in f.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        score = float(parts[7]) if len(parts) >= 8 else 1.0
                    except ValueError:
                        score = 1.0
                    if score >= min_score:
                        n += 1
            except OSError:
                continue
            preds[f.stem] = n

    exems: dict[str, int] = {}
    if r is not None:
        try:
            for e in read_exemplar_set(project, r).exemplars:
                exems[e.image] = exems.get(e.image, 0) + 1
        except (HTTPException, ValueError, OSError) as exc:
            print(f"  exemplar counts unavailable for {r.name}: {exc}")

    open_comments: dict[str, int] = {}
    cdir = project.annotations_dir.parent / "comments"
    if cdir.is_dir():
        for f in cdir.glob("*.json"):
            try:
                items = json.loads(f.read_text(encoding="utf-8")).get("comments", [])
            except (OSError, ValueError):
                continue
            n = sum(1 for c in items if isinstance(c, dict) and not c.get("resolved"))
            if n:
                open_comments[f.stem] = n

    out = []
    for name in project.image_names():
        stem = Path(name).stem
        path = annotation_path(project, name)
        n, classes = 0, {}
        if path.is_file():
            for line in path.read_text().splitlines():
                head = line.split(maxsplit=1)
                if not head:
                    continue
                n += 1
                cls = class_token(head[0])
                if cls is not None:
                    classes[cls] = classes.get(cls, 0) + 1
        try:
            size = (project.image_dir / name).stat().st_size
        except OSError:
            size = 0
        out.append({
            "name": name, "bytes": size, "boxes": n, "classes": classes,
            "stage": stage_of(entries, name, defs),
            "flags": flags_of(flag_entries, name)[0], "flag_note": flags_of(flag_entries, name)[1],
            "preds": preds.get(stem, 0), "exems": exems.get(name, 0),
            "comments": open_comments.get(stem, 0),
        })
    return out


@app.get("/api/image/{name}")
def get_image(project: ProjectDep, name: str) -> FileResponse:
    return FileResponse(image_path(project, name))


def thumb_path(project: Project, name: str, size: int) -> Path:
    """Cache location for one thumbnail.

    Beside annotations/, never inside images/: images/ may be a symlink into a
    read-only share. Keyed on the full file name rather than the stem, because
    the cache must not care whether two images happen to share one.
    """
    return project.annotations_dir.parent / ".thumbs" / str(size) / (name + ".jpg")


def render_thumb(src: Path, size: int) -> bytes:
    with Image.open(src) as im:
        # draft() lets the JPEG decoder skip straight to a reduced scale,
        # which is most of the cost on large field photos. No-op otherwise.
        im.draft("RGB", (size, size))
        # Browsers apply EXIF orientation to <img>, so the thumbnail must too
        # or the grid would show a photo rotated relative to the editor.
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82, optimize=True)
        return buf.getvalue()


@app.get("/api/thumb/{name}")
def get_thumb(
    project: ProjectDep,
    name: str,
    size: Annotated[int, Query(ge=16, le=4096)] = THUMB_SIZES[0],
) -> Response:
    """A downscaled JPEG for the grid, cached on disk.

    The cache is best effort. A stale entry is detected by mtime and rebuilt;
    a project folder that cannot be written to still gets its thumbnails, it
    just renders them on every request.
    """
    src = image_path(project, name)
    size = next((s for s in THUMB_SIZES if s >= size), THUMB_SIZES[-1])
    cached = thumb_path(project, name, size)
    try:
        if cached.is_file() and cached.stat().st_mtime >= src.stat().st_mtime:
            return FileResponse(cached, media_type="image/jpeg",
                                headers=THUMB_HEADERS)
    except OSError:
        pass

    try:
        data = render_thumb(src, size)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(415, f"cannot decode {name}: {exc}") from exc

    # The endpoint is sync, so FastAPI runs it in a thread pool and two tiles
    # for the same image can race. A per-thread temp name plus an atomic
    # replace means the loser simply overwrites with identical bytes.
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(
            f".{cached.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(cached)
    except OSError:
        pass
    return Response(data, media_type="image/jpeg", headers=THUMB_HEADERS)


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
        "stage": stage_of(image_stages(project), name, stage_defs(project)),
        "flags": flags_of(image_flag_entries(project), name)[0],
        "flag_note": flags_of(image_flag_entries(project), name)[1],
        "active_ms": event_log(project).active_ms(name),
        "coverage": read_coverage(project, name),
        "comments": read_comments(project, name),
    }


@app.put("/api/annotation/{name}")
def put_annotation(project: ProjectDep, name: str, annotation: Annotation) -> dict:
    """No run parameter: annotations and stages are project-scoped, and a run
    must never be able to influence where hand labels land."""
    image_path(project, name)  # validates the name
    with _annotation_lock:
        write_annotation(project, name, annotation.boxes)
    if annotation.stage is not None:
        defs = stage_defs(project)
        if stage_of(image_stages(project), name, defs) != annotation.stage:
            set_stages(project, [name], annotation.stage)
    if annotation.flags is not None:
        set_image_flags(project, name, annotation.flags, annotation.flag_note or "")
    note_classes(project, annotation.boxes)
    return {"saved": len(annotation.boxes)}


@app.post("/api/coverage/{name}")
def post_coverage(project: ProjectDep, name: str, delta: CoverageDelta) -> dict:
    """Add a batch of looked-at time to an image's grid. A stored grid of
    another shape is replaced rather than mixed with - its cells would not
    mean the same parts of the image."""
    image_path(project, name)                   # validates the name
    path = coverage_path(project, name)
    with _coverage_lock:
        cur = read_coverage(project, name)
        if cur and cur["gw"] == delta.gw and cur["gh"] == delta.gh:
            ms = [min(a + b, MAX_CELL_MS) for a, b in zip(cur["ms"], delta.ms)]
        else:
            ms = list(delta.ms)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"gw": delta.gw, "gh": delta.gh,
                                   "ms": [round(v) for v in ms]},
                                  separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    return {"cells": len(ms)}


@app.post("/api/events")
def post_events(project: ProjectDep, batch: EventBatch) -> dict:
    """Append a batch of events from the browser.

    All or nothing: a batch naming an image this project does not have is
    refused whole, so a stale tab cannot half-write. The client drops a batch
    the server refused (it would be refused again) and retries one the server
    never answered.
    """
    stamp = now()
    records = []
    for ev in batch.events:
        if ev.image is not None:
            image_path(project, ev.image)       # 404 on anything unknown
        records.append({"t": stamp, "actor": ACTOR, "project": project.name,
                        **ev.model_dump()})
    if records:
        event_log(project).append(records)
    return {"written": len(records)}


@app.get("/api/settings")
def get_settings() -> dict:
    """The saved personal settings, or the defaults. A file that cannot be
    read is reported, not fatal: the dialog falls back to the defaults and
    says so. Without settings.json, an older keybindings.json is carried over."""
    path, legacy = settings_path(), legacy_keybindings_path()
    warning = note = None
    settings = PersonalSettings()
    source = path if path.is_file() else legacy if legacy.is_file() else None
    if source is not None:
        try:
            if source == path:
                settings = PersonalSettings.model_validate_json(
                    path.read_text(encoding="utf-8"))
            else:
                settings = from_legacy_keybindings(legacy)
                note = (f"Shortcuts carried over from {display_path(legacy)}; they "
                        f"are saved to {display_path(path)} on your next change.")
        except (ValueError, OSError) as exc:
            first = str(exc).splitlines()[0]
            warning = (f"{display_path(source)} could not be read ({first}); using "
                       f"the defaults. The file is kept aside on your next change.")
    return {**settings.model_dump(exclude_none=True), "path": display_path(path),
            "warning": warning, "note": note}


@app.put("/api/settings")
def put_settings(settings: PersonalSettings) -> dict:
    """Replace the saved settings: written to a temporary file and moved into
    place. A file that exists but cannot be read is renamed rather than
    overwritten - it may hold hand edits worth recovering - and a carried-over
    keybindings.json is renamed once its contents are safely in settings.json."""
    path, legacy = settings_path(), legacy_keybindings_path()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with _settings_lock:
        write_settings_file(path, settings, PersonalSettings.model_validate_json)
        if legacy.is_file():
            legacy.replace(legacy.with_name(f"{legacy.stem}.carried-over-{stamp}.json"))
    return {"saved": True, "path": display_path(path)}


@app.get("/api/project-settings")
def get_project_settings(project: ProjectDep) -> dict:
    """The class list and styles, plus every class name the annotations use.
    A version-1 file is migrated on the way (see migrate_project_settings_v1).
    An unreadable file is reported, not fatal."""
    path = project_settings_path(project)
    warning = note = None
    ps = ProjectSettings()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("version", 1) == 1:
                with _settings_lock:
                    ps, files = migrate_project_settings_v1(project, raw)
                    write_settings_file(path, ps, lambda _t: None)
                note = ("Class names are now written into the annotation files themselves"
                        + (f" ({files} files updated; the old ones are in backups/)." if files else "."))
            else:
                ps = ProjectSettings.model_validate(raw)
        except (ValueError, OSError) as exc:
            first = str(exc).splitlines()[0]
            warning = (f"{display_path(path)} could not be read ({first}); using the "
                       f"default class list. The file is kept aside on your next change.")
    return {**ps.model_dump(exclude_none=True),
            "stages": stage_defs(project) if ps.stages is None else [st.model_dump() for st in ps.stages],
            "seen": sorted(seen_classes(project)),
            "path": display_path(path), "warning": warning, "note": note}


@app.put("/api/project-settings")
def put_project_settings(project: ProjectDep, ps: ProjectSettings) -> dict:
    path = project_settings_path(project)
    with _settings_lock:
        write_settings_file(path, ps, ProjectSettings.model_validate_json)
    return {"saved": True, "path": display_path(path)}


class StageChange(BaseModel):
    images: list[str] = Field(min_length=1, max_length=100_000)
    stage: str


@app.post("/api/stages")
def change_stages(project: ProjectDep, req: StageChange) -> dict:
    """Move many images to one stage at once - the grid's selection. One write,
    one log entry per image that moved."""
    known = set(project.image_names())
    unknown = [n for n in req.images if n not in known]
    if unknown:
        raise HTTPException(404, f"{len(unknown)} unknown image(s), e.g. {unknown[0]!r}")
    moved = set_stages(project, req.images, req.stage)
    return {"moved": len(moved), "stage": req.stage}


@app.get("/api/classes/counts")
def get_class_counts(project: ProjectDep) -> dict:
    """Boxes per class, from a scan of every annotation file."""
    return {"counts": class_counts(project)}


class ClassRename(BaseModel):
    old: str
    new: str
    merge: bool = False                  # the new name is taken: join the two

    @field_validator("old", "new")
    @classmethod
    def _one_word(cls, v: str) -> str:
        if not valid_class_name(v):
            raise ValueError(f"class names are one word, at most {CLASS_NAME_MAX} "
                             f"characters: {v!r}")
        return v


@app.post("/api/classes/rename")
def rename_class(project: ProjectDep, req: ClassRename) -> dict:
    """Rename a class everywhere it is used: every annotation file (after a
    backup of the folder) and the class list. Renaming onto a name that is
    already a class merges the two, and only when asked to."""
    if req.old == req.new:
        return {"files": 0, "boxes": 0}
    path = project_settings_path(project)
    with _settings_lock:
        ps = ProjectSettings()
        if path.is_file():
            ps = ProjectSettings.model_validate_json(path.read_text(encoding="utf-8"))
        names = {c.name for c in ps.classes} | seen_classes(project)
        if req.new in names and not req.merge:
            raise HTTPException(409, f"{req.new!r} is already a class; merging needs merge=true")
        with _annotation_lock:
            backup = backup_annotations(project, f"rename-{req.old}")
            files, boxes = rename_classes_in_files(project, {req.old: req.new})
        old = next((c for c in ps.classes if c.name == req.old), None)
        target = next((c for c in ps.classes if c.name == req.new), None)
        if old is not None and target is None:
            old.name = req.new                     # keeps its place and style
        elif old is not None:
            ps.classes.remove(old)                 # merged: the target's style wins
        elif target is None:
            ps.classes.append(ClassDef(name=req.new))
        write_settings_file(path, ps, ProjectSettings.model_validate_json)
    with _seen_lock:
        _seen_classes.pop(str(project.annotations_dir), None)
    log_server_event(project, "class.rename", None,
                     {"from": req.old, "to": req.new, "merge": req.merge,
                      "files": files, "boxes": boxes})
    return {"files": files, "boxes": boxes,
            "backup": display_path(backup) if backup else None,
            **ps.model_dump(exclude_none=True)}


# --------------------------------------------------------------------------
# Comments. See the module docstring.
# --------------------------------------------------------------------------

class CommentIn(BaseModel):
    x: float = Field(ge=0, le=1)         # normalised position of its pin
    y: float = Field(ge=0, le=1)
    text: str = Field(min_length=1, max_length=4000)
    # point: at x, y. region: a box drawn for the comment, never an annotation.
    # box / pred / exem: on a box of yours, a sam3 box or an exemplar.
    kind: Literal["point", "region", "box", "pred", "exem"] = "point"
    box: list[float] | None = None       # normalised x1, y1, x2, y2
    cls: str | None = Field(default=None, max_length=64)   # the class of the box commented on

    @model_validator(mode="after")
    def _box_for_kind(self) -> "CommentIn":
        if self.kind == "point":
            self.box = None
            return self
        b = self.box
        if not b or len(b) != 4 or not all(0 <= v <= 1 for v in b) or b[0] >= b[2] or b[1] >= b[3]:
            raise ValueError("a comment on an area or a box needs box: [x1, y1, x2, y2], normalised")
        return self


class CommentPatch(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=4000)
    resolved: bool | None = None
    # A comment on a box follows the box: its new place, normalised.
    box: list[float] | None = None

    @field_validator("box")
    @classmethod
    def _box(cls, b: list[float] | None) -> list[float] | None:
        if b is not None and (len(b) != 4 or not all(0 <= v <= 1 for v in b)
                              or b[0] >= b[2] or b[1] >= b[3]):
            raise ValueError("box is [x1, y1, x2, y2], normalised")
        return b


_comments_lock = threading.Lock()


def comments_path(project: Project, name: str) -> Path:
    return project.annotations_dir.parent / "comments" / (Path(name).stem + ".json")


def read_comments(project: Project, name: str) -> list[dict]:
    path = comments_path(project, name)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [c for c in data.get("comments", []) if isinstance(c, dict) and "id" in c]
    except (ValueError, OSError) as exc:
        print(f"  ignoring {path.name}: {exc}")
        return []


def write_comments(project: Project, name: str, comments: list[dict]) -> None:
    path = comments_path(project, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"image": name, "comments": comments}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def log_server_event(project: Project, type_: str, image: str | None, data: dict) -> None:
    """An event the server itself originates, in the same shape as the
    browser's, so the log reads as one stream."""
    event_log(project).append([{
        "t": now(), "actor": ACTOR, "project": project.name,
        "id": uuid.uuid4().hex, "type": type_,
        "ct": int(datetime.now(timezone.utc).timestamp() * 1000),
        "session": "server", "image": image, "run": None, "data": data,
    }])


@app.post("/api/comments/{name}")
def add_comment(project: ProjectDep, name: str, c: CommentIn) -> dict:
    image_path(project, name)
    item = {"id": uuid.uuid4().hex[:12], "x": c.x, "y": c.y, "text": c.text,
            "kind": c.kind, "box": c.box, "cls": c.cls,
            "author": ACTOR, "created": now(), "updated": None, "resolved": False}
    with _comments_lock:
        comments = read_comments(project, name)
        comments.append(item)
        write_comments(project, name, comments)
    log_server_event(project, "comment.add", name, {"id": item["id"], "x": c.x, "y": c.y,
                                                    "kind": c.kind, "box": c.box,
                                                    "text": c.text})
    return item


@app.patch("/api/comments/{name}/{cid}")
def edit_comment(project: ProjectDep, name: str, cid: str, patch: CommentPatch) -> dict:
    image_path(project, name)
    with _comments_lock:
        comments = read_comments(project, name)
        item = next((c for c in comments if c["id"] == cid), None)
        if item is None:
            raise HTTPException(404, f"no comment {cid} on {name}")
        if patch.text is not None:
            item["text"] = patch.text
        if patch.resolved is not None:
            item["resolved"] = patch.resolved
        if patch.box is not None:
            item["box"] = patch.box
            item["x"], item["y"] = patch.box[2], patch.box[1]    # the pin: top right
        if patch.text is not None or patch.resolved is not None:
            item["updated"] = now()
        write_comments(project, name, comments)
    log_server_event(project, "comment.edit", name,
                     {"id": cid, **patch.model_dump(exclude_none=True)})
    return item


@app.delete("/api/comments/{name}/{cid}")
def delete_comment(project: ProjectDep, name: str, cid: str) -> dict:
    image_path(project, name)
    with _comments_lock:
        comments = read_comments(project, name)
        kept = [c for c in comments if c["id"] != cid]
        if len(kept) == len(comments):
            raise HTTPException(404, f"no comment {cid} on {name}")
        write_comments(project, name, kept)
    log_server_event(project, "comment.delete", name, {"id": cid})
    return {"deleted": cid}


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


@app.get("/api/run-exemplars")
def get_run_exemplars(project: ProjectDep, run: RunDep, request: Request) -> dict:
    """The run's exemplars, for the header's exemplar window: every one picked
    for it, in pick order, and - when the run has been made - which it used
    (from run_manifest.json; a run that sampled gt/ boxes used those)."""
    if run is None:
        return {"run": None, "picked": [], "last_run": None, "wanted": []}
    picked = [e.model_dump() for e in read_exemplar_set(project, run).exemplars]
    try:
        settings, _ = read_run_settings(request.app.state.cfg, run)
        wanted = [{"name": c.name, "n": c.n or settings.exemplars.n}
                  for c in settings.inference.run_classes()]
        source = settings.exemplars.source
    except (ValueError, OSError):
        wanted, source = [], None
    last = None
    try:
        m = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        when = datetime.fromtimestamp(run.manifest_path.stat().st_mtime, timezone.utc)
        last = {
            "when": when.isoformat(timespec="seconds"),
            "source": m.get("exemplar_source"),
            "class_name": m.get("class_name"),
            "used": [{"image": e.get("source"), "cls": e.get("class") or m.get("class_name"),
                      "cx": b[0], "cy": b[1], "w": b[2], "h": b[3]}
                     for e in m.get("exemplars", []) if len(b := e.get("box_cxcywh_norm") or []) == 4],
        }
    except (OSError, ValueError):
        pass                                 # not run yet, or a manifest it cannot read
    return {"run": run.name, "picked": picked, "last_run": last, "wanted": wanted, "source": source}


CROP_SIZES = (96, 160, 320)


@app.get("/api/crop/{name}")
def get_crop(
    project: ProjectDep,
    name: str,
    cx: Annotated[float, Query(ge=0, le=1)],
    cy: Annotated[float, Query(ge=0, le=1)],
    w: Annotated[float, Query(gt=0, le=1)],
    h: Annotated[float, Query(gt=0, le=1)],
    pad: Annotated[float, Query(ge=0, le=2)] = 0.25,
    size: Annotated[int, Query(ge=16, le=1024)] = 160,
) -> Response:
    """A box cut out of its image - with `pad` of its size around it, as the
    SAM 3 script pastes exemplars - scaled to `size` on its long edge. For
    showing exemplars; cached next to the thumbnails."""
    src = image_path(project, name)
    size = next((v for v in CROP_SIZES if v >= size), CROP_SIZES[-1])
    key = f"{cx:.5f},{cy:.5f},{w:.5f},{h:.5f},{pad:.3f},{size}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    cached = project.dir / ".thumbs" / "crops" / f"{Path(name).stem}-{digest}.jpg"
    try:
        if cached.is_file() and cached.stat().st_mtime >= src.stat().st_mtime:
            return FileResponse(cached, media_type="image/jpeg", headers=THUMB_HEADERS)
    except OSError:
        pass
    try:
        with Image.open(src) as raw:
            # Upright, as the editor shows it and the boxes were drawn on it.
            im = ImageOps.exif_transpose(raw) or raw
            W, H = im.size
            px, py = w * pad * W, h * pad * H
            box = (max(0, round((cx - w / 2) * W - px)), max(0, round((cy - h / 2) * H - py)),
                   min(W, round((cx + w / 2) * W + px)), min(H, round((cy + h / 2) * H + py)))
            tile = im.convert("RGB").crop(box)
        tile.thumbnail((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        tile.save(buf, "JPEG", quality=88)
        data = buf.getvalue()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(415, f"cannot decode {name}: {exc}") from exc
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(f".{cached.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(cached)
    except OSError:
        pass
    return Response(data, media_type="image/jpeg", headers=THUMB_HEADERS)


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
        cls=box.cls,
    )
    s.exemplars.append(entry)
    s.updated = entry.picked
    write_exemplar_set(run, s)
    return {"id": entry.id, "set": s.name, "count": len(s.exemplars)}


class ExemplarOrder(BaseModel):
    ids: list[int]


@app.post("/api/exemplars/order")
def order_exemplars(project: ProjectDep, run: RunDep, order: ExemplarOrder) -> dict:
    """Put the exemplars in this order - "the first n" of a class are the ones
    a run uses. Ids not listed keep their order, after the listed ones."""
    r = _need_run(run)
    s = read_exemplar_set(project, r)
    by_id = {e.id: e for e in s.exemplars}
    first = [by_id[i] for i in order.ids if i in by_id]
    rest = [e for e in s.exemplars if e.id not in set(order.ids)]
    s.exemplars = first + rest
    s.updated = now()
    write_exemplar_set(r, s)
    return {"count": len(s.exemplars)}


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
    app.state.token = access_token() if cfg.annotator.require_token else None
    query = f"?token={app.state.token}" if app.state.token else ""
    print(f"open: https://ondemand.gamarello.agsad.admin.ch"
          f"/node/{socket.gethostname()}/{port}/{query}")
    if app.state.token:
        print("      (the token is remembered by the browser after the first visit)")
    else:
        print("[warn] annotator.require_token is off: anyone who can reach this "
              "node's port can use the annotator, and launch runs")
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