"""Typed config and path resolution for the SAM 3 run and the annotation tool.

One schema, one YAML file, read by both programs. They have to agree about
where images, annotations, predictions and the exemplar manifest live, and a
single source is the only way to guarantee they do.

Both programs validate the WHOLE file even though each reads only part of it.
That keeps extra="forbid" useful everywhere: a typo in a section you are not
using still fails at startup rather than lying dormant.

--- Layout -------------------------------------------------------------------
The YAML no longer carries per-dataset paths. It carries the ROOT that every
dataset lives under, and the name of the one dataset the SAM 3 script targets
on its next run. Everything else is derived:

    <root>/
      <project>/                       one dataset; the annotator can open any
        images/                        the dataset's pictures
        annotations/                   human labels        PROJECT-scoped
        flags/                         sign-off state      PROJECT-scoped
        gt/                            optional, read-only shipped labels
        runs/
          <run>/                       one SAM 3 configuration  RUN-scoped
            predictions/
            overlays/
            composites/
            exemplar_tiles/            the crops actually pasted
            exemplars/<run>.json       the manifest that prompted this run
            run_manifest.json

The annotator keeps more project-scoped files next to annotations/ - its
class list and styles (settings.json), comments/, coverage/, events/ (the
append-only log), backups/ (annotations copied aside before a class rename)
and .thumbs/. None of them is read by the SAM 3 script except settings.json,
whose class list it checks inference.class_name against.

--- Classes -------------------------------------------------------------------
A class is a NAME, the first field of every annotation and prediction line:
    <class> <cx> <cy> <w> <h> <provenance> <quality> <score>
There is no numeric id with a mapping kept elsewhere. A name is one word
(no whitespace), because the line is split on whitespace. Older files with
numeric classes still read - "0" is a class called "0" - and some exporters
write "2.0" for 2, which reads as "2". Both programs use valid_class_name()
and class_token() below, so they cannot disagree about what a class is.

Annotations and flags sit above `runs/` on purpose: they are facts about the
dataset, not about one SAM 3 configuration, and putting them under a run name
would orphan hours of irreplaceable work the moment a run is renamed.
Predictions are the opposite - regenerable, and meaningless without the run
that produced them.

`runs/` is a namespace rather than a flat sibling so that a run called
"annotations" cannot collide with the annotations directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

CLASS_NAME_MAX = 64


def valid_class_name(name: str) -> bool:
    """One word: it is the first field of a whitespace-separated line."""
    return (0 < len(name) <= CLASS_NAME_MAX
            and not any(ch.isspace() or ord(ch) < 32 for ch in name))


def class_token(raw: str) -> str | None:
    """The class of an annotation or prediction line, as stored. Integer ids
    that some exporters write as floats ("2.0") read as "2", so an old numeric
    file does not split one class into two. None if it cannot be a class."""
    tok = raw.strip()
    if "." in tok:
        try:
            f = float(tok)
            if f.is_integer():
                tok = str(int(f))
        except ValueError:
            pass
    return tok if valid_class_name(tok) else None


def _expand(p: Path) -> Path:
    """Expand ~ and $VARS in a path before resolving."""
    return Path(os.path.expandvars(str(p))).expanduser()


def _safe_name(kind: str, name: str) -> str:
    """Reject anything that could escape the directory it will be joined to.

    Project and run names arrive from a query parameter, so this is a security
    boundary, not a tidiness check. Spaces are allowed - "Rumex Project" is a
    legitimate name - but separators and traversal are not.
    """
    if not name or name.strip() != name:
        raise ValueError(f"{kind} name may not be empty or padded: {name!r}")
    if name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"{kind} name may not contain a path separator: {name!r}")
    return name


# --------------------------------------------------------------------------
# Resolved locations. Plain classes, not models: they are resolved per request
# from a name the caller supplies, not validated once at load.
# --------------------------------------------------------------------------


class Run:
    """Everything scoped to one SAM 3 configuration.

    A Run may not exist on disk yet. That is deliberate: exemplars are picked
    BEFORE the run that consumes them exists, so the annotator has to be able
    to address a run it is about to create. Use mkdirs() at the point of the
    first write, and `exists` to decide what to show.
    """

    def __init__(self, project: "Project", name: str):
        self.project = project
        self.name = _safe_name("run", name)
        self.dir = project.runs_dir / self.name

    @property
    def exists(self) -> bool:
        return self.dir.is_dir()

    @property
    def predictions_dir(self) -> Path:
        return self.dir / "predictions"

    @property
    def overlays_dir(self) -> Path:
        return self.dir / "overlays"

    @property
    def composites_dir(self) -> Path:
        return self.dir / "composites"

    @property
    def tiles_dir(self) -> Path:
        """The exemplar CROPS this run pasted - images, not the manifest.

        Kept apart from exemplars/ so the directory holding the manifest stays
        one file per set and is not silently filled with PNGs.
        """
        return self.dir / "exemplar_tiles"

    @property
    def exemplar_dir(self) -> Path:
        return self.dir / "exemplars"

    @property
    def exemplar_manifest(self) -> Path:
        """One manifest per run, named after it, so the annotator and the SAM 3
        script compute the same path from the same two values."""
        return self.exemplar_dir / f"{self.name}.json"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "run_manifest.json"

    def mkdirs(self) -> None:
        for d in (self.predictions_dir, self.overlays_dir, self.composites_dir,
                  self.tiles_dir, self.exemplar_dir):
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Run({self.project.name!r}, {self.name!r})"


class Project:
    """One dataset, and every directory belonging to it.

    Raises FileNotFoundError when the project or its images/ directory is
    missing, so a bad name from the YAML or from a query parameter fails with
    a message that names the path rather than producing an empty listing.
    """

    def __init__(self, root: Path, name: str):
        self.root = root
        self.name = _safe_name("project", name)
        self.dir = root / self.name
        if not self.dir.is_dir():
            raise FileNotFoundError(f"no such project: {self.dir}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"project {self.name!r} has no images/ directory: {self.image_dir}"
            )

    # -- project-scoped: survives any run being renamed or deleted ----------

    @property
    def image_dir(self) -> Path:
        return self.dir / "images"

    @property
    def annotations_dir(self) -> Path:
        return self.dir / "annotations"

    @property
    def flags_dir(self) -> Path:
        return self.dir / "flags"

    @property
    def gt_dir(self) -> Path | None:
        """The dataset's own shipped labels: optional, read-only, often absent.

        Deliberately separate from annotations/, which is human work from the
        tool and irreplaceable. Conflating them would let a path mistake point
        the tool's writes at a published label set.
        """
        d = self.dir / "gt"
        return d if d.is_dir() else None

    @property
    def runs_dir(self) -> Path:
        return self.dir / "runs"

    @property
    def settings_path(self) -> Path:
        """The annotator's project settings: the class list, in order."""
        return self.dir / "settings.json"

    def class_names(self) -> list[str] | None:
        """The project's class list as the annotator keeps it, or None when
        there is no list to check against: no file yet, a file the annotator
        has not migrated to class names (version 1 - opening the project in
        the annotator once does that), or one that cannot be read."""
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("version") != 2 or not isinstance(data.get("classes"), list):
            return None
        return [c["name"] for c in data["classes"] if isinstance(c, dict) and "name" in c]

    # -- runs ---------------------------------------------------------------

    def run(self, name: str) -> Run:
        return Run(self, name)

    def list_runs(self) -> list[str]:
        if not self.runs_dir.is_dir():
            return []
        return sorted(p.name for p in self.runs_dir.iterdir() if p.is_dir())

    # -- images -------------------------------------------------------------

    def image_names(self, exts: set[str] | None = None) -> list[str]:
        exts = exts or IMAGE_SUFFIXES
        return sorted(
            p.name for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

    def mkdirs(self) -> None:
        """Only the project-scoped writable directories. Runs make their own."""
        for d in (self.annotations_dir, self.flags_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- discovery ----------------------------------------------------------

    @staticmethod
    def discover(root: Path) -> list[str]:
        """Every valid project under root, by name.

        A directory without images/ is not a project - it is a stray folder,
        and listing it would put a broken entry in the dropdown.
        """
        if not root.is_dir():
            return []
        return sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / "images").is_dir()
        )

    def __repr__(self) -> str:
        return f"Project({self.name!r})"


# --------------------------------------------------------------------------
# The YAML schema
# --------------------------------------------------------------------------


class Base(BaseModel):
    # A key the schema does not know about is a typo, and typos should be loud.
    model_config = ConfigDict(extra="forbid")


class PathsCfg(Base):
    """Where everything lives, and which dataset SAM 3 targets next.

    `dataset` is read by the SAM 3 script only. The annotator ignores it except
    as the entry preselected in its dropdown: it can open any project under
    root, so binding it to one would defeat the point.
    """

    root: Path              # the folder every project lives under
    dataset: str            # SAM 3's target project; annotator's default

    @field_validator("root", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        v = _expand(v)
        if not v.is_absolute():
            raise ValueError(f"must be an absolute path, got {v}")
        if not v.is_dir():
            raise ValueError(f"paths.root is not a directory: {v}")
        return v

    @field_validator("dataset", mode="after")
    @classmethod
    def _name(cls, v: str) -> str:
        return _safe_name("dataset", v)


class AnnotatorCfg(Base):
    """The annotation server itself.

    Behind the Open OnDemand proxy the port is assigned per session, so
    Annotator.py prefers $HOST and $PORT when they are set.
    """

    host: str = "0.0.0.0"          # 127.0.0.1 is unreachable through the proxy
    port: int = Field(8000, ge=1, le=65535)


class RunCfg(Base):
    name: str = "sam3_run"
    overwrite: bool = True

    @field_validator("name", mode="after")
    @classmethod
    def _name(cls, v: str) -> str:
        return _safe_name("run", v)


class ExemplarCfg(Base):
    # random_gt: sample n ground-truth boxes, an oracle and an upper bound.
    # manifest:  the boxes a human actually picked in the annotation tool.
    # The comparison between the two is the experiment.
    #
    # The manifest path is derived from the run, so a sweep over pad or n has
    # to reuse one run's manifest or copy it forward deliberately.
    source: Literal["random_gt", "manifest"] = "random_gt"
    n: int = Field(10, ge=1)
    seed: int | None = 0             # ignored when source is manifest
    pad: float = Field(0.25, ge=0.0)

    # Applied AFTER the tile is scaled to match the target, so it only bounds
    # canvas size. If it binds, build_tiles warns: the exemplar has stopped
    # matching the target's apparent scale.
    tile_max_edge: int = Field(1024, ge=1)


class StripCfg(Base):
    side: Literal["auto", "right", "bottom"] = "auto"
    margin: int = Field(8, ge=0)
    fill: tuple[int, int, int] = (114, 114, 114)


class InferenceCfg(Base):
    text_prompt: str | None = None
    # The class every prediction line of this run is written as - a name from
    # the project's class list, so that approving a prediction in the
    # annotator keeps it. Required by the SAM 3 script, ignored by the
    # annotator, hence optional here: the annotator validates this file too.
    class_name: str | None = None
    conf_threshold: float = Field(0.1, ge=0.0, le=1.0)
    max_edge: int = Field(2048, ge=1)
    fast_prompt_append: bool = True

    @model_validator(mode="before")
    @classmethod
    def _no_class_id(cls, data: Any) -> Any:
        # class_id did two jobs and is split in two; say so rather than let
        # extra="forbid" report a bare "extra input".
        if isinstance(data, dict) and "class_id" in data:
            raise ValueError(
                "inference.class_id is gone: classes are names now. Set "
                "inference.class_name to the class the predictions belong to "
                "(e.g. rumex), and dataset.gt_class to filter the gt/ labels "
                "(null = every class)")
        return data

    @field_validator("class_name", mode="after")
    @classmethod
    def _one_word(cls, v: str | None) -> str | None:
        if v is not None and not valid_class_name(v):
            raise ValueError(f"a class name is one word, at most {CLASS_NAME_MAX} "
                             f"characters: {v!r}")
        return v


class DatasetCfg(Base):
    num_images: int | None = None
    image_seed: int = 0
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    # Which class of the shipped gt/ labels counts, compared as a class token:
    # 0, "0" and "0.0" are the same class, and a named one works too.
    # null = every class.
    gt_class: str | None = None

    @field_validator("gt_class", mode="before")
    @classmethod
    def _as_token(cls, v: Any) -> str | None:
        if v is None:
            return None
        tok = class_token(str(v))
        if tok is None:
            raise ValueError(f"not a class: {v!r}")
        return tok


class PredictionCfg(Base):
    provenance: str = "sam3"
    quality: str = "unverified"


class DebugCfg(Base):
    save_overlays: int = Field(20, ge=0)
    save_composites: int = Field(5, ge=0)
    mask_color: tuple[int, int, int] = (255, 45, 45)
    mask_alpha: float = Field(0.45, ge=0.0, le=1.0)
    mask_edge_alpha: float = Field(0.95, ge=0.0, le=1.0)
    draw_pred_boxes: bool = True
    draw_gt_boxes: bool = True


class Config(Base):
    paths: PathsCfg
    annotator: AnnotatorCfg = AnnotatorCfg()
    run: RunCfg = RunCfg()
    exemplars: ExemplarCfg = ExemplarCfg()
    strip: StripCfg = StripCfg()
    inference: InferenceCfg = InferenceCfg()
    dataset: DatasetCfg = DatasetCfg()
    predictions: PredictionCfg = PredictionCfg()
    debug: DebugCfg = DebugCfg()

    # Resolution happens in these methods, not in a validator: load_config()
    # must succeed for the annotator even when paths.dataset names a project
    # that is missing, because the annotator can open any of the others.

    def project(self, name: str | None = None) -> Project:
        """Resolve a project by name, defaulting to the YAML's dataset."""
        return Project(self.paths.root, name or self.paths.dataset)

    def active_run(self, project: Project | None = None) -> Run:
        """The run named in the YAML, in the given project (default: dataset)."""
        return (project or self.project()).run(self.run.name)

    def projects(self) -> list[str]:
        return Project.discover(self.paths.root)


def load_config(path: str | Path) -> Config:
    """Read and validate one YAML file.

    safe_load, never load: plain yaml.load can instantiate arbitrary Python
    objects named in the file, which turns a config into an execution vector.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"config not found: {path.resolve()}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return Config.model_validate(raw)