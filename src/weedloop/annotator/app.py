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

Class names and class box styles belong to one project, because class ids
mean something different in each: they live in <project>/settings.json.

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

Completion flags (one file per image, in <project>/flags/): a single character,
0 or 1. 1 means "every instance in this image is boxed" - so an image with flag
1 and zero boxes is a confirmed negative, which is a useful training example.
A missing flag file means the image has not been signed off.

Run:
    python Annotator.py --config Sam3N10.yaml
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import re
import shutil
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


MAX_CLASS_ID = 9999


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int = Field(default=CLASS_ID, ge=0, le=MAX_CLASS_ID)
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
        # The class id is kept, not assumed: a file from a multi-class dataset
        # must come back out with the ids it went in with. "3.0" is accepted,
        # since some exporters write class ids as floats.
        try:
            cls = int(float(parts[0]))
        except ValueError:
            cls = -1
        if not 0 <= cls <= MAX_CLASS_ID:
            print(f"  skipping {path.name}:{lineno} - bad class id {parts[0]!r}")
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


class MinimapSettings(BaseModel):
    """The minimap's look and place, and what counts as having looked at a
    part of the image: at least `min_zoom` times the fit-to-screen zoom, for
    `seen_s` seconds in total."""

    show: bool = True
    size: int = Field(default=220, ge=100, le=480)          # width, screen px
    opacity: float = Field(default=0.85, ge=0.2, le=1.0)
    right: float = Field(default=12, ge=0, le=10000)        # top-right corner, px
    top: float = Field(default=12, ge=0, le=10000)          # from the view's
    seen_s: float = Field(default=1.5, ge=0.5, le=10)
    min_zoom: float = Field(default=2.0, ge=1.0, le=8.0)


class ScanSettings(BaseModel):
    """Automatic scanning: how fast the view slides (screen px per second), how
    much two rows overlap, how long after the last touch it resumes, and
    whether it skips rows already seen or moves on to the next image."""

    speed: float = Field(default=130, ge=10, le=2000)
    overlap: float = Field(default=0.2, ge=0, le=0.5)
    resume_s: float = Field(default=2.0, ge=0.5, le=10)
    skip_seen: bool = False
    next_image: bool = False


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


class ClassEntry(BaseModel):
    name: str | None = Field(default=None, max_length=64)   # None: shows as "class 3"
    style: BoxStyle | None = None                            # None: the palette default

    @field_validator("name")
    @classmethod
    def _printable(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if any(ord(ch) < 32 for ch in v):
            raise ValueError("control characters in a class name")
        return v.strip() or None


class ProjectSettings(BaseModel):
    """What belongs to one project rather than to one person: class names and
    how each class is drawn. Keys are class ids, as strings (JSON keys)."""

    version: int = SETTINGS_SCHEMA
    classes: dict[str, ClassEntry] = Field(default_factory=dict)

    @field_validator("classes")
    @classmethod
    def _class_ids(cls, classes: dict[str, ClassEntry]) -> dict[str, ClassEntry]:
        if len(classes) > 1000:
            raise ValueError("more than 1000 classes")
        for key in classes:
            if not key.isdigit() or int(key) > MAX_CLASS_ID or str(int(key)) != key:
                raise ValueError(f"not a class id: {key!r}")
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


# Class ids that occur in a project's annotations, so the settings dialog can
# list every class there is, not only those in images opened so far. One scan
# per project and server run; every save adds its classes to the result.
_seen_classes: dict[str, set[int]] = {}
_seen_lock = threading.Lock()


def seen_classes(project: Project) -> set[int]:
    key = str(project.annotations_dir)
    with _seen_lock:
        if key not in _seen_classes:
            ids: set[int] = set()
            if project.annotations_dir.is_dir():
                for f in project.annotations_dir.glob("*.txt"):
                    try:
                        text = f.read_text()
                    except OSError:
                        continue
                    for line in text.splitlines():
                        head = line.split(maxsplit=1)
                        if not head:
                            continue
                        try:
                            cls = int(float(head[0]))
                        except ValueError:
                            continue
                        if 0 <= cls <= MAX_CLASS_ID:
                            ids.add(cls)
            _seen_classes[key] = ids
        return set(_seen_classes[key])


def note_classes(project: Project, boxes: list[Box]) -> None:
    with _seen_lock:
        known = _seen_classes.get(str(project.annotations_dir))
        if known is not None:
            known.update(b.cls for b in boxes)


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
        signed, started = progress(p, names)
        out.append({
            "name": name,
            "runs": runs,
            "images": len(names),
            "signed_off": signed,
            "in_progress": started,
            "cover": names[0] if names else None,
        })
    return {
        "projects": out,
        "default_project": cfg.paths.dataset,
        "root": str(cfg.paths.root),
    }


def progress(project: Project, names: list[str]) -> tuple[int, int]:
    """(signed off, in progress) for the project card.

    Counted by stem against the current image list, so a flag or annotation
    left behind by an image that has since been removed does not inflate
    either number. "In progress" means boxes exist but no sign-off - the same
    three states the ribbon and the grid colour by.

    Annotated is judged by file size rather than by reading each file: an
    empty annotation file is exactly what an image with every box deleted
    leaves behind.
    """
    stems = {Path(n).stem for n in names}
    signed: set[str] = set()
    if project.flags_dir.is_dir():
        for f in project.flags_dir.glob("*.txt"):
            if f.stem not in stems:
                continue
            try:
                if f.read_text().strip() == "1":
                    signed.add(f.stem)
            except OSError:
                pass
    boxed: set[str] = set()
    if project.annotations_dir.is_dir():
        for f in project.annotations_dir.glob("*.txt"):
            try:
                if f.stem in stems and f.stat().st_size > 0:
                    boxed.add(f.stem)
            except OSError:
                pass
    return len(signed), len(boxed - signed)


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
        "complete": read_flag(project, name),
        "active_ms": event_log(project).active_ms(name),
        "coverage": read_coverage(project, name),
    }


@app.put("/api/annotation/{name}")
def put_annotation(project: ProjectDep, name: str, annotation: Annotation) -> dict:
    """No run parameter: annotations and flags are project-scoped, and a run
    must never be able to influence where hand labels land."""
    image_path(project, name)  # validates the name
    write_annotation(project, name, annotation.boxes)
    write_flag(project, name, annotation.complete)
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
    """Class names and styles for this project, plus every class id its
    annotations use. An unreadable file is reported, not fatal."""
    path = project_settings_path(project)
    warning = None
    ps = ProjectSettings()
    if path.is_file():
        try:
            ps = ProjectSettings.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            first = str(exc).splitlines()[0]
            warning = (f"{display_path(path)} could not be read ({first}); using the "
                       f"default class styles. The file is kept aside on your next change.")
    return {**ps.model_dump(exclude_none=True),
            "seen": sorted(seen_classes(project) | {CLASS_ID}),
            "path": display_path(path), "warning": warning}


@app.put("/api/project-settings")
def put_project_settings(project: ProjectDep, ps: ProjectSettings) -> dict:
    path = project_settings_path(project)
    with _settings_lock:
        write_settings_file(path, ps, ProjectSettings.model_validate_json)
    return {"saved": True, "path": display_path(path)}


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