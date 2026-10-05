"""
=============================================================================
Rumex / haldennord09 -- SAM 3 few-shot segmentation via pasted exemplar tiles
=============================================================================

Ten exemplar boxes are sampled once across the whole dataset. For every target
image, the exemplar CROPS are pasted into a strip beside the image, forming one
composite canvas:

    +-----------------------------+-------+
    |                             | tile1 |
    |        target image         | tile2 |
    |          (W x H)            | tile3 |
    |                             |  ...  |
    +-----------------------------+-------+

SAM 3 is then prompted with boxes over the pasted tiles. Because those boxes
now sit inside the image currently loaded in the state, the coordinate-only
limitation of `add_geometric_prompt` no longer bites: the exemplars are
embedded from the canvas's own backbone features, i.e. from the exemplar
pixels themselves. Predictions falling in the strip are discarded, the rest
are cropped back to the target region and rendered as red overlays.

This is deliberately a hack -- see "Known distortions" below. It exists to
answer, cheaply and today, whether few-shot Rumex exemplars are worth the
deeper `img_ids` engineering at all.

--- Why this works when the coordinate route did not -------------------------
`add_geometric_prompt` stores raw normalised coordinates in
`state["geometric_prompt"]`; the exemplar is only turned into features inside
`model.forward_grounding(backbone_out=state["backbone_out"], ...)`, against
whatever image is loaded. Replaying image A's coordinates on image B samples
image B. Pasting image A's pixels INTO the canvas makes the coordinates point
at the right appearance again.

--- Known distortions (read before trusting the output) ----------------------
* Resolution. The processor resizes the canvas to 1008x1008 regardless of
  aspect, so the strip steals pixels from the target. With STRIP_SIDE = "auto"
  the strip goes on the long edge, which costs least. Expect the target to
  keep roughly W/(W+strip_w) of its former detail -- printed at startup.
* Seams. Pasted tiles introduce hard edges and a background fill that do not
  occur in real field images; the backbone sees them.
* Self-detection. The tiles are Rumex, so SAM finds them. Anything whose
  predicted box centre lands in the strip is dropped.
* The exemplars are ground truth -- tight, correct, chosen by someone who
  already knew where the plants were. Results are an upper bound.

--- Coordinate spaces --------------------------------------------------------
Four spaces are in play; only one hop is lossy and needs undoing.
  original file   W0 x H0   -- what the annotation tool and any downstream
                               training see
  resized target  W1 x H1   -- after load_image()'s MAX_EDGE thumbnail
  canvas          Wc x Hc   -- target pasted at (0, 0), so target and canvas
                               coordinates coincide inside the target region;
                               there is no offset to subtract
  model internal  1008x1008 -- the processor rescales, but state["boxes"]
                               comes back in canvas pixels already
segment_canvas() therefore returns xyxy pixels in the RESIZED TARGET space.
write_yolo_predictions() scales that back to W0 x H0 and normalises.

Outputs
-------
    <OUTPUT_DIR>/overlays/<stem>_overlay.png     first SAVE_OVERLAYS images only:
                                                 target + red masks
    <OUTPUT_DIR>/predictions/<stem>.txt          YOLO + provenance/quality/score,
                                                 normalised against the ORIGINAL
                                                 image size
    <OUTPUT_DIR>/composites/<stem>_canvas.png    first SAVE_COMPOSITES canvases
                                                 with prompt boxes drawn, for
                                                 sanity-checking the layout
    <OUTPUT_DIR>/exemplars/tile_XX.png           the 10 crops actually used
    <OUTPUT_DIR>/run_manifest.json

Prediction line format (8 fields, space separated):
    <class> <cx> <cy> <w> <h> <provenance> <quality> <score>
The class is a NAME, exactly as in the annotation files, not a number that
needs a mapping to mean anything. A run predicts one class or several
(inference.classes): each is prompted on its own - its own exemplar tiles, its
own text - and all of them are written to the same file per image. Each
should be a class on the project's list in the annotator, so that approving
a prediction keeps its class; the run warns when one is not.

How the classes are prompted is one choice per run (inference.prompt_mode):
exemplar tiles pasted in a strip beside the image; the exemplar boxes that lie
on the image itself, where they are; or the text prompt alone.

Masks (predictions.save_masks) go next to each prediction file as
<stem>.masks.json: one entry per prediction line, in the same order - its
bounding box in the mask's pixel space and the runs inside it.

Progress goes to <run>/progress.json as the run goes (phase, images done,
total), which is how the annotator shows a run it launched. Fields 2-5 are YOLO geometry,
normalised; fields 6-8 are the extra flags the annotation tool reads.
Converting to a standard format (numeric YOLO, COCO) is an export step.

Launch (interactive, Gamarello):
    unset SLURM_MEM_PER_CPU
    salloc --mem 32G -c 4 --gres=gpu:1 -p gpu --time=01:00:00
    export HOME=/home/f89605564
    agrosoft load
    module load python/3.12.12 cuda/12.9.1
    source $HOME/envs/SamSeg/bin/activate
    python3 $HOME/scripts/sam3_rumex_haldennord09.py
=============================================================================
"""

import argparse
from weedloop.config import class_token, load_config

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ============================== CONFIG ======================================



# =============================================================================


# --------------------------- device / model ---------------------------------

def build_model_and_processor(cfg):
    """CUDA only. Earlier versions carried CPU and MPS paths plus a helper that
    chased down tensors SAM3 caches as plain attributes (which model.to() misses,
    producing 'all tensors on the same device' errors off CUDA). None of that is
    reachable on Gamarello, so it is gone; restore from git if you ever need it.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible - request a GPU allocation")
    device = torch.device("cuda")

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"using device: {device} ({torch.cuda.get_device_name(0)})")

    sam3_root = Path(sam3.__file__).parent.parent
    bpe_path = sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe_path), device=device, enable_inst_interactivity=True
    )
    # build_sam3_image_model's `if device == "cuda"` doesn't match a torch.device
    model = model.to(device)
    processor = Sam3Processor(model, device=device)
    processor.set_confidence_threshold(cfg.inference.conf_threshold)
    return model, processor


# ------------------------------ dataset -------------------------------------

def read_yolo_boxes(txt_path: Path, cfg):
    """(class, normalised cxcywh) for every box of one gt/ file. Classes are
    class tokens, so a numeric file (0, 0.0) and a named one (rumex) both work;
    which class counts is decided per run class (gt_wanted)."""
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = class_token(parts[0])
        if cls is None:
            continue
        try:
            boxes.append((cls, tuple(float(v) for v in parts[1:5])))
        except ValueError:
            continue
    return boxes


def gt_wanted(c, classes, cfg):
    """The gt/ class that stands for run class `c`: its own gt_class; for a
    one-class run, dataset.gt_class (None: every gt box, as before classes
    were names); otherwise the class's own name."""
    if c.gt_class is not None:
        return c.gt_class
    if len(classes) == 1:
        return cfg.dataset.gt_class
    return c.name


def of_class(boxes, want):
    return [b for cls, b in boxes if want is None or cls == want]


def collect_pairs(image_dir: Path, anno_dir: Path, cfg):
    exts = {e.lower() for e in tuple(cfg.dataset.image_exts)}
    return [
        (p, read_yolo_boxes(anno_dir / f"{p.stem}.txt", cfg) if anno_dir else [])
        for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in exts
    ]


def load_image(path: Path, cfg):
    """Returns (possibly downscaled image, original (width, height)).

    The original size is what every downstream consumer -- the annotation tool,
    any YOLO training run -- works in, so it has to survive the thumbnail.
    """
    im = Image.open(path).convert("RGB")
    orig_wh = im.size
    if cfg.inference.max_edge and max(im.size) > cfg.inference.max_edge:
        im.thumbnail((cfg.inference.max_edge, cfg.inference.max_edge), Image.LANCZOS)
    return im, orig_wh


def norm_to_xyxy_px(box, w, h):
    cx, cy, bw, bh = box
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


# ---------------------------- exemplar tiles --------------------------------

class Tile:
    """A cropped exemplar plus the annotation box's position inside the crop."""

    def __init__(self, image: Image.Image, inner_xyxy, source: str, box):
        self.image = image
        self.inner = inner_xyxy  # xyxy px, relative to the crop
        self.source = source
        self.box = box  # the original normalised cxcywh, for the manifest

    @property
    def size(self):
        return self.image.size

def read_manifest(run, image_dir: Path):
    path = run.exemplar_manifest
    if not path.is_file():
        raise SystemExit(f"exemplar manifest not found: {path}")
    data = json.loads(path.read_text())
    if data.get("image_dir") != str(image_dir):
        raise SystemExit(
            f"manifest was picked against {data.get('image_dir')}, "
            f"but this run reads {image_dir}; the stems would index the wrong "
            f"pictures"
        )
    return data


def select_from_manifest(data, c, n, single, image_dir: Path):
    """The first n boxes picked as class c, in pick order - stable, so n=5
    against a 10-box set is reproducibly the first five. Boxes picked before
    exemplars recorded their class count for a one-class run only: with
    several classes nobody can tell whose they are."""
    entries = data.get("exemplars", [])
    mine = [e for e in entries if e.get("cls") == c.name or (single and e.get("cls") is None)]
    unlabelled = sum(1 for e in mine if e.get("cls") is None)
    if unlabelled:
        print(f"[note] {unlabelled} exemplar(s) were picked before exemplars recorded "
              f"their class; they are used as {c.name}")
    if not mine:
        raise SystemExit(f"no exemplars picked as {c.name!r} for this run")
    if len(mine) < n:
        print(f"[warn] {len(mine)} exemplar(s) of {c.name}, n={n}; using all of them")
    return [(image_dir / e["image"], (e["cx"], e["cy"], e["w"], e["h"])) for e in mine[:n]]


def select_random_gt(pairs, want, n, seed, gt_dir, name):
    flat = [(p, box) for p, boxes in pairs for box in of_class(boxes, want)]
    if not flat:
        raise SystemExit(f"no gt/ boxes of class {want!r} under {gt_dir} for {name}")
    draw = random.Random(seed).sample(flat, min(n, len(flat)))
    if len(draw) < n:
        print(f"[warn] only {len(draw)} gt/ boxes of {want!r} exist for {name}; using all of them")
    return draw

def build_tiles(picked, pad, max_edge, target_max_edge):
    """Crops each picked box, downscaled by the SAME factor its source image
    would receive as a target.

    The target is shrunk to cfg.inference.max_edge before compositing. A crop
    taken at full source resolution therefore lands on the canvas larger than
    the same plant does inside the target - by exactly that factor, ~1.97x for
    4032px originals at max_edge 2048. Matching the factor here keeps apparent
    scale honest.

    An earlier version deliberately did NOT rescale, on the reasoning that
    shrinking the exemplars would change their apparent scale. That had it
    backwards: the target is shrunk, so not shrinking the exemplar is what
    breaks the match.

    `max_edge` (cfg.exemplars.tile_max_edge) is applied afterwards and now only
    bounds canvas size. Set it high enough not to bind, or it silently
    reintroduces a mismatch for large plants.
    """
    tiles = []
    for path, box in picked:
        src = Image.open(path).convert("RGB")
        W, H = src.size
        x1, y1, x2, y2 = norm_to_xyxy_px(box, W, H)
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        cx1, cy1 = max(0.0, x1 - px), max(0.0, y1 - py)
        cx2, cy2 = min(float(W), x2 + px), min(float(H), y2 + py)
        crop = src.crop((int(cx1), int(cy1), int(math.ceil(cx2)), int(math.ceil(cy2))))
        inner = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]

        # Match the downscale this image would get as a target.
        s_target = min(1.0, target_max_edge / max(W, H)) if target_max_edge else 1.0
        if s_target < 1.0:
            crop = crop.resize((max(1, int(crop.width * s_target)),
                                max(1, int(crop.height * s_target))), Image.LANCZOS)
            inner = [v * s_target for v in inner]

        # Then the tile cap, purely to bound canvas size.
        if max_edge and max(crop.size) > max_edge:
            s = max_edge / max(crop.size)
            crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))),
                               Image.LANCZOS)
            inner = [v * s for v in inner]
            print(f"[warn] tile from {path.name} hit tile_max_edge={max_edge} "
                  f"and was shrunk a further {s:.2f}x - its apparent scale no "
                  f"longer matches the target")

        tiles.append(Tile(crop, inner, path.name, box))
    return tiles


def _pack(sizes, limit, margin):
    """Greedy 1-D packing: groups consecutive tiles into lanes no longer than
    `limit`. Returns a list of lists of indices."""
    lanes, cur, cur_len = [], [], margin
    for i, s in enumerate(sizes):
        if cur and cur_len + s + margin > limit:
            lanes.append(cur)
            cur, cur_len = [], margin
        cur.append(i)
        cur_len += s + margin
    if cur:
        lanes.append(cur)
    return lanes


def compose(image: Image.Image, tiles, side, margin, cfg):
    """Returns (canvas, prompt_boxes_xyxy_on_canvas, target_region_wh).
    The target always sits at the canvas origin, so the target region is
    simply (0, 0, W, H)."""
    W, H = image.size
    if side == "auto":
        side = "right" if W >= H else "bottom"

    if side == "right":
        lanes = _pack([t.image.height for t in tiles], H, margin)
        lane_w = [max(tiles[i].image.width for i in lane) for lane in lanes]
        strip_w = sum(lane_w) + margin * (len(lanes) + 1)
        strip_h = max(sum(tiles[i].image.height + margin for i in lane) + margin
                      for lane in lanes)
        canvas = Image.new("RGB", (W + strip_w, max(H, strip_h)), tuple(cfg.strip.fill))
        canvas.paste(image, (0, 0))
        boxes, x = [], W + margin
        for lane, lw in zip(lanes, lane_w):
            y = margin
            for i in lane:
                t = tiles[i]
                canvas.paste(t.image, (x, y))
                boxes.append([x + t.inner[0], y + t.inner[1], x + t.inner[2], y + t.inner[3]])
                y += t.image.height + margin
            x += lw + margin
    else:  # bottom
        lanes = _pack([t.image.width for t in tiles], W, margin)
        lane_h = [max(tiles[i].image.height for i in lane) for lane in lanes]
        strip_h = sum(lane_h) + margin * (len(lanes) + 1)
        strip_w = max(sum(tiles[i].image.width + margin for i in lane) + margin
                      for lane in lanes)
        canvas = Image.new("RGB", (max(W, strip_w), H + strip_h), tuple(cfg.strip.fill))
        canvas.paste(image, (0, 0))
        boxes, y = [], H + margin
        for lane, lh in zip(lanes, lane_h):
            x = margin
            for i in lane:
                t = tiles[i]
                canvas.paste(t.image, (x, y))
                boxes.append([x + t.inner[0], y + t.inner[1], x + t.inner[2], y + t.inner[3]])
                x += t.image.width + margin
            y += lh + margin

    return canvas, boxes, (W, H)


def xyxy_px_to_norm_cxcywh(box, w, h):
    x1, y1, x2, y2 = box
    return [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h]


# ------------------------------ inference -----------------------------------

def append_prompts(processor, state, boxes_norm, cfg):
    """Every add_geometric_prompt() call runs a full forward_grounding, so
    prompting with N tiles costs N forwards. Only the final state matters, so
    the middle boxes are appended straight to the prompt object -- the same
    thing add_geometric_prompt does internally -- leaving 2 forwards total.
    Set cfg.inference.fast_prompt_append = False to use the plain loop."""
    if not boxes_norm:
        return state
    if not cfg.inference.fast_prompt_append or len(boxes_norm) <= 2:
        for box in boxes_norm:
            state = processor.add_geometric_prompt(box=list(box), label=True, state=state)
        return state

    state = processor.add_geometric_prompt(box=list(boxes_norm[0]), label=True, state=state)
    try:
        for box in boxes_norm[1:-1]:
            b = torch.tensor(box, device=processor.device, dtype=torch.float32).view(1, 1, 4)
            lab = torch.tensor([True], device=processor.device, dtype=torch.bool).view(1, 1)
            state["geometric_prompt"].append_boxes(b, lab)
    except Exception as exc:
        print(f"[warn] fast prompt append failed ({type(exc).__name__}: {exc}); "
              f"falling back to one call per box")
        for box in boxes_norm[1:-1]:
            state = processor.add_geometric_prompt(box=list(box), label=True, state=state)
    return processor.add_geometric_prompt(box=list(boxes_norm[-1]), label=True, state=state)


def segment_canvas(processor, canvas, prompt_boxes_px, target_wh, cfg, want_masks=False, text=None):
    """Runs the prompted canvas and keeps only what falls in the target region.

    Returned boxes are xyxy pixels in the RESIZED TARGET space (W1 x H1), not
    the original file's space -- see write_yolo_predictions().

    Masks are only copied off the GPU when `want_masks` is set. At canvas
    resolution with tens of detections that copy is hundreds of MB per image
    and dominated the loop; nothing but overlay rendering needs it.
    """
    cw, ch = canvas.size
    tw, th = target_wh

    state = processor.set_image(canvas)
    text = cfg.inference.text_prompt if text is None else text
    if text:
        state = processor.set_text_prompt(prompt=text, state=state)
    boxes_norm = [xyxy_px_to_norm_cxcywh(b, cw, ch) for b in prompt_boxes_px]
    state = append_prompts(processor, state, boxes_norm, cfg)

    masks = state.get("masks")
    if masks is None or masks.shape[0] == 0:
        return None, np.zeros((0, 4)), np.zeros((0,)), 0

    boxes_np = state["boxes"].detach().float().cpu().numpy()  # xyxy px on canvas
    scores_np = state["scores"].detach().float().cpu().numpy().reshape(-1)

    cx = (boxes_np[:, 0] + boxes_np[:, 2]) / 2
    cy = (boxes_np[:, 1] + boxes_np[:, 3]) / 2
    keep = (cx < tw) & (cy < th)

    n_tile_hits = int((~keep).sum())
    boxes_np = boxes_np[keep]
    scores_np = scores_np[keep]
    if boxes_np.size:
        boxes_np[:, [0, 2]] = boxes_np[:, [0, 2]].clip(0, tw)
        boxes_np[:, [1, 3]] = boxes_np[:, [1, 3]].clip(0, th)

    masks_np = None
    if want_masks:
        masks_np = masks.squeeze(1).detach().cpu().numpy().astype(bool)[keep][:, :th, :tw]
    return masks_np, boxes_np, scores_np, n_tile_hits


# ----------------------------- predictions ----------------------------------

def to_yolo_lines(boxes_px, scores, resized_wh, orig_wh, cfg, class_name=None, keep_index=False):
    """Scales resized-target xyxy pixels back to the ORIGINAL image and
    normalises to YOLO cxcywh.

    The x and y factors are computed separately: PIL's thumbnail() rounds each
    axis independently, so W0/W1 and H0/H1 are not exactly equal and using one
    factor for both introduces a small systematic stretch.
    """
    w1, h1 = resized_wh
    w0, h0 = orig_wh
    if w1 <= 0 or h1 <= 0:
        return ([], []) if keep_index else []
    sx, sy = w0 / w1, h0 / h1

    lines, kept = [], []
    for k, (box, score) in enumerate(zip(np.asarray(boxes_px).reshape(-1, 4), np.asarray(scores).reshape(-1))):
        x1, x2 = sorted((float(box[0]) * sx, float(box[2]) * sx))
        y1, y2 = sorted((float(box[1]) * sy, float(box[3]) * sy))
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w0), x2), min(float(h0), y2)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        lines.append(
            f"{class_name or cfg.inference.class_name} "
            f"{((x1 + x2) / 2) / w0:.6f} {((y1 + y2) / 2) / h0:.6f} "
            f"{(x2 - x1) / w0:.6f} {(y2 - y1) / h0:.6f} "
            f"{cfg.predictions.provenance} {cfg.predictions.quality} {float(score):.4f}"
        )
        kept.append(k)
    return (lines, kept) if keep_index else lines


def encode_mask(mask: np.ndarray):
    """A boolean mask as {"box": [x0, y0, w, h], "rle": [start, length, ...]}:
    its bounding box, and the runs of set pixels inside it, row by row. Small
    for plant-shaped masks, and quick to paint back in a browser. None for an
    empty mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    flat = mask[y0:y1, x0:x1].reshape(-1).astype(np.int8)
    d = np.diff(np.concatenate(([0], flat, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    rle = np.empty(starts.size * 2, dtype=np.int64)
    rle[0::2], rle[1::2] = starts, ends - starts
    return {"box": [x0, y0, x1 - x0, y1 - y0], "rle": rle.tolist()}


def write_masks(path: Path, size, masks) -> None:
    """<stem>.masks.json: the masks of a prediction file, line for line, in the
    resized image's pixel space (size)."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps({"version": 1, "size": list(size), "masks": masks},
                              separators=(",", ":")))
    tmp.replace(path)


def write_lines(path: Path, lines) -> int:
    """Writes one prediction file - every class of the run - via a temporary
    name, so an interrupted run leaves the previous version, not half of it."""
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
    tmp.replace(path)
    return len(lines)


def write_yolo_predictions(path: Path, boxes_px, scores, resized_wh, orig_wh, cfg) -> int:
    lines = to_yolo_lines(boxes_px, scores, resized_wh, orig_wh, cfg)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
    tmp.replace(path)
    return len(lines)


# ------------------------------ rendering -----------------------------------

def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Pure-numpy 1px boundary (no scipy dependency)."""
    edge = np.zeros_like(mask)
    edge[1:, :] |= mask[1:, :] ^ mask[:-1, :]
    edge[:-1, :] |= mask[:-1, :] ^ mask[1:, :]
    edge[:, 1:] |= mask[:, 1:] ^ mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] ^ mask[:, 1:]
    return edge & mask


def render_overlay(image, masks, pred_boxes, gt_boxes, cfg):
    w, h = image.size
    base = image.convert("RGBA")

    if masks is not None and masks.size:
        union = masks.any(axis=0)
        alpha = np.where(union, int(cfg.debug.mask_alpha * 255), 0).astype(np.uint8)
        edges = np.zeros_like(union)
        for m in masks:
            edges |= mask_outline(m)
        alpha[edges] = int(cfg.debug.mask_edge_alpha * 255)
        wash = Image.new("RGBA", (w, h), tuple(cfg.debug.mask_color) + (0,))
        wash.putalpha(Image.fromarray(alpha, mode="L"))
        base = Image.alpha_composite(base, wash)

    out = base.convert("RGB")
    draw = ImageDraw.Draw(out)
    lw = max(2, min(w, h) // 300)
    if cfg.debug.draw_pred_boxes:
        for box in pred_boxes:
            draw.rectangle([float(v) for v in box], outline=tuple(cfg.debug.mask_color), width=lw)
    if cfg.debug.draw_gt_boxes:
        for box in gt_boxes:            # already the class's own
            draw.rectangle(norm_to_xyxy_px(box, w, h), outline="yellow", width=max(1, lw // 2))
    return out


def save_canvas_debug(canvas, prompt_boxes, path: Path):
    im = canvas.copy()
    draw = ImageDraw.Draw(im)
    lw = max(2, min(im.size) // 300)
    for box in prompt_boxes:
        draw.rectangle([float(v) for v in box], outline="lime", width=lw)
    im.save(path)


# ------------------------------- progress -----------------------------------

class Progress:
    """<run>/progress.json, rewritten as the run goes: what the annotator shows
    of a run it launched. Small, atomic, at most a few times a second."""

    def __init__(self, path: Path):
        self.path = path
        self.state = {"phase": "starting", "done": 0, "total": 0, "class": None,
                      "started": time.time(), "updated": time.time(), "message": None}
        self._last = 0.0
        self.write(force=True)

    def update(self, force=False, **kw):
        self.state.update(kw)
        self.write(force)

    def write(self, force=False):
        now_ = time.time()
        if not force and now_ - self._last < 0.5:
            return
        self._last = now_
        self.state["updated"] = now_
        try:
            tmp = self.path.with_name(f".{self.path.name}.tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.path)
        except OSError:
            pass                    # progress is a nicety; the run goes on


# --------------------------------- main --------------------------------------

def main(cfg):
    # One project, named by paths.dataset; one run, named by run.name. Both
    # raise with the offending path rather than producing an empty listing.
    project = cfg.project()                # FileNotFoundError names the path
    run = cfg.active_run(project)
    run.mkdirs()
    progress = Progress(run.progress_path)
    try:
        _main(cfg, project, run, progress)
    except BaseException as exc:           # SystemExit and Ctrl-C too: say how it ended
        msg = str(exc) if not isinstance(exc, KeyboardInterrupt) else "interrupted"
        if not (isinstance(exc, SystemExit) and exc.code in (0, None)):
            progress.update(force=True, phase="failed", message=msg or type(exc).__name__)
        raise


def _main(cfg, project, run, progress):
    # Every prediction line is written as its class, so there has to be one -
    # and each should be on the project's list, or an approved prediction
    # would not keep it (the annotator gives it the class being drawn with).
    classes = cfg.inference.run_classes()
    if not classes:
        raise SystemExit("no class to predict: set inference.classes (or class_name)")
    known = project.class_names()
    if known is None:
        print(f"[note] {project.name} has no class list yet (or the annotator has "
              f"not opened it since classes became names); the classes are not "
              f"checked against one")
    else:
        for c in classes:
            if c.name not in known:
                print(f"[warn] {c.name!r} is not on {project.name}'s class list "
                      f"({', '.join(known) or 'empty'}). Predictions are written as it "
                      f"anyway, but approving one in the annotator will give it the class "
                      f"being drawn with. Add it in the annotator, or fix the run's classes.")

    image_dir = project.image_dir
    anno_dir = project.gt_dir              # may be None; collect_pairs handles it
    overlay_dir = run.overlays_dir
    pred_dir = run.predictions_dir

    if cfg.exemplars.source == "random_gt" and anno_dir is None:
        raise SystemExit(
            f"exemplars.source is 'random_gt' but {project.name!r} has no gt/ "
            f"directory; there are no ground-truth boxes to sample from"
        )

    print(f"project     =  {project.name}  ({project.dir})")
    print(f"run         =  {run.name}  ({run.dir})")
    for c in classes:
        text = cfg.inference.text_for(c)
        print(f"class       =  {c.name}" + (f"   text {text!r}" if text else "")
              + (f"   (gt/ class {gt_wanted(c, classes, cfg)})"
                 if cfg.exemplars.source == "random_gt" else ""))

    pairs = collect_pairs(image_dir, anno_dir, cfg)
    n_boxes = sum(len(b) for _, b in pairs)
    print(f"found {len(pairs)} images and {n_boxes} annotated boxes under {image_dir}")
    if not pairs:
        progress.update(force=True, phase="done", message="no images")
        return

    # Each class: what prompts it - strip tiles built once; in-image exemplars
    # looked up per image; or nothing but its text.
    mode = cfg.inference.prompt_mode
    single = len(classes) == 1
    print(f"prompting   =  {mode}")
    manifest_meta = (read_manifest(run, image_dir)
                     if mode != "text" and cfg.exemplars.source == "manifest" else None)
    tile_dir = run.tiles_dir
    tile_dir.mkdir(parents=True, exist_ok=True)
    for f in tile_dir.glob("tile_*.png"):  # this launch's tiles only, not a mix with the last
        f.unlink()
    per_class = []
    for c in classes:
        n = c.n or cfg.exemplars.n
        text = cfg.inference.text_for(c)
        pc = {"c": c, "n": n, "text": text, "gt": gt_wanted(c, classes, cfg), "tiles": [], "by_image": None}
        if mode == "text":
            if not text:
                raise SystemExit(f"prompt_mode is text, but {c.name} has no text prompt")
            print(f"text only {c.name}: {text!r}")
        elif mode == "in_image":
            if manifest_meta is not None:
                pc["by_image"] = {}
                for e in manifest_meta.get("exemplars", []):
                    if e.get("cls") == c.name or (single and e.get("cls") is None):
                        pc["by_image"].setdefault(e["image"], []).append((e["cx"], e["cy"], e["w"], e["h"]))
                if not pc["by_image"] and not text:
                    raise SystemExit(f"no exemplars picked as {c.name!r} on any image, and no text "
                                     f"prompt to fall back on")
                print(f"in-image {c.name}: exemplars on {len(pc['by_image'])} image(s), up to {n} "
                      f"per image" + (", text alone on the others" if text else ", others skipped"))
            else:
                print(f"in-image {c.name}: up to {n} gt/ box(es) of {pc['gt']!r} on each image"
                      + (", text alone where there are none" if text else ""))
        else:
            if manifest_meta is not None:
                picked = select_from_manifest(manifest_meta, c, n, single, image_dir)
            else:
                picked = select_random_gt(pairs, pc["gt"], n, cfg.exemplars.seed, anno_dir, c.name)
            pc["tiles"] = build_tiles(picked, cfg.exemplars.pad, cfg.exemplars.tile_max_edge, cfg.inference.max_edge)
            prefix = "" if single else f"{c.name}_"
            for i, t in enumerate(pc["tiles"]):
                t.image.save(tile_dir / f"tile_{prefix}{i:02d}_{Path(t.source).stem}.png")
            origin = (f"set={manifest_meta['name']}" if manifest_meta else f"seed={cfg.exemplars.seed}")
            print(f"exemplar strip {c.name}: {len(pc['tiles'])} tiles from "
                  f"{len({t.source for t in pc['tiles']})} image(s), "
                  f"source={cfg.exemplars.source}, {origin}")
        per_class.append(pc)

    def image_prompts(pc, img_path, gt_all):
        """In-image mode: the boxes on this image that prompt class pc."""
        if pc["by_image"] is not None:
            return pc["by_image"].get(img_path.name, [])[: pc["n"]]
        mine = sorted(of_class(gt_all, pc["gt"]))
        rng = random.Random(f"{cfg.exemplars.seed}:{img_path.name}:{pc['c'].name}")
        return rng.sample(mine, min(pc["n"], len(mine)))

    targets = pairs
    if cfg.dataset.num_images is not None:
        targets = random.Random(cfg.dataset.image_seed).sample(pairs, min(cfg.dataset.num_images, len(pairs)))
    if not cfg.run.overwrite:
        # Keyed on the prediction file, not the overlay: overlays are only
        # written for the first cfg.debug.save_overlays images, so testing for
        # one would make every later image look unprocessed.
        targets = [(p, b) for p, b in targets
                   if not (pred_dir / f"{p.stem}.txt").exists()]
    print(f"{len(targets)} image(s) queued")
    if not targets:
        print("nothing to do: every image already has predictions (overwrite is off)")
        progress.update(force=True, phase="done", total=0, message="nothing to do")
        return
    progress.update(force=True, phase="loading model", total=len(targets))

    if mode == "strip":
        # report how much resolution the strip costs on the first image
        probe, _ = load_image(targets[0][0], cfg)
        probe_canvas, _, (pw, ph) = compose(probe, per_class[0]["tiles"], cfg.strip.side, cfg.strip.margin, cfg)
        frac = (pw * ph) / (probe_canvas.width * probe_canvas.height)
        print(f"canvas {probe_canvas.width}x{probe_canvas.height} vs target {pw}x{ph} "
              f"-> the target keeps {frac:.0%} of the canvas area "
              f"(the processor rescales the whole canvas to 1008x1008)")

    model, processor = build_model_and_processor(cfg)
    comp_dir = run.composites_dir
    if cfg.debug.save_composites:
        comp_dir.mkdir(parents=True, exist_ok=True)
    save_masks = cfg.predictions.save_masks
    progress.update(force=True, phase="predicting")

    with torch.inference_mode():
        n_done = n_inst = n_tile = n_written = 0
        per_image = []
        t0 = time.time()

        for img_path, gt_all in tqdm(targets, desc="images"):
            image, orig_wh = load_image(img_path, cfg)
            want_masks = save_masks or n_done < cfg.debug.save_overlays
            lines, counts, mask_entries = [], {}, []
            for pc in per_class:
                c, text, gt_want = pc["c"], pc["text"], pc["gt"]
                progress.update(**{"class": c.name})
                used = []
                if mode == "strip":
                    canvas, prompt_boxes, target_wh = compose(image, pc["tiles"], cfg.strip.side, cfg.strip.margin, cfg)
                else:
                    canvas, target_wh, prompt_boxes = image, image.size, []
                    if mode == "in_image":
                        used = image_prompts(pc, img_path, gt_all)
                        prompt_boxes = [norm_to_xyxy_px(b, *image.size) for b in used]
                        if not prompt_boxes and not text:
                            counts[c.name] = {"skipped": "no exemplar on this image and no text prompt"}
                            continue
                tag = "" if single else f"_{c.name}"
                if n_done < cfg.debug.save_composites:
                    save_canvas_debug(canvas, prompt_boxes, comp_dir / f"{img_path.stem}{tag}_canvas.png")
                try:
                    masks, boxes, scores, tile_hits = segment_canvas(
                        processor, canvas, prompt_boxes, target_wh, cfg, want_masks=want_masks, text=text)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    tqdm.write(f"[warn] CUDA OOM on {img_path.name} ({c.name}); skipped "
                               f"(lower cfg.inference.max_edge or cfg.exemplars.tile_max_edge)")
                    continue
                except Exception as exc:
                    tqdm.write(f"[warn] {img_path.name} ({c.name}) failed "
                               f"({type(exc).__name__}: {exc}); skipped")
                    continue
                if want_masks:
                    render_overlay(image, masks, boxes, of_class(gt_all, gt_want), cfg).save(
                        overlay_dir / f"{img_path.stem}{tag}_overlay.png")
                # image.size is the resized target space segment_canvas clipped
                # to, so the two can never drift apart.
                cls_lines, kept = to_yolo_lines(boxes, scores, image.size, orig_wh, cfg,
                                                class_name=c.name, keep_index=True)
                lines += cls_lines
                if save_masks:
                    mask_entries += [encode_mask(masks[k]) if masks is not None else None for k in kept]
                n_inst += int(boxes.shape[0])
                n_tile += tile_hits
                counts[c.name] = {"n_pred": int(boxes.shape[0]), "n_written": len(cls_lines),
                                  "n_dropped_in_strip": tile_hits,
                                  "mean_score": float(scores.mean()) if scores.size else None}
                if mode == "in_image":
                    counts[c.name]["prompts"] = [list(b) for b in used]
            n_lines = write_lines(pred_dir / f"{img_path.stem}.txt", lines)
            mask_path = pred_dir / f"{img_path.stem}.masks.json"
            if save_masks:
                write_masks(mask_path, image.size, mask_entries)
            elif mask_path.exists():
                mask_path.unlink()             # it would describe the old lines

            n_done += 1
            n_written += n_lines
            progress.update(done=n_done)
            per_image.append({
                "image": img_path.name,
                "orig_wh": list(orig_wh),
                "resized_wh": list(image.size),
                "n_gt": len(gt_all),
                "n_written": n_lines,
                "classes": counts,
            })

            if n_done % 25 == 0:
                torch.cuda.empty_cache()

        dt = time.time() - t0
        print(f"\nprocessed {n_done} image(s) x {len(classes)} class(es) "
              f"({n_inst} instances kept, {n_tile} dropped in the strip, "
              f"{dt / max(n_done, 1):.2f}s/image)")
        print(f"wrote {min(n_done, cfg.debug.save_overlays)} overlay(s) to {overlay_dir}")
        print(f"wrote {n_written} prediction line(s) across {n_done} file(s) to {pred_dir}")
        if n_inst and n_written < n_inst:
            print(f"[note] {n_inst - n_written} prediction(s) were dropped as degenerate "
                  f"(under 1px on an axis after scaling back to the original size)")

        all_tiles = [(pc["c"], t) for pc in per_class for t in pc["tiles"]]
        run.manifest_path.write_text(json.dumps({
            "method": "pasted exemplar tiles",
            "n_exemplars": len(all_tiles),
            "exemplar_source": cfg.exemplars.source,
            "exemplar_seed": cfg.exemplars.seed,
            "exemplar_manifest": str(run.exemplar_manifest),
            "exemplar_set_name": manifest_meta["name"] if manifest_meta else None,
            "exemplar_set_updated": manifest_meta["updated"] if manifest_meta else None,
            # every exemplar used, with its class; per class below as well
            "exemplars": [{"source": t.source, "class": c.name, "box_cxcywh_norm": list(t.box),
                           "tile_px": list(t.size)} for c, t in all_tiles],
            "prompt_mode": mode,
            "masks_saved": save_masks,
            "classes": [{"name": pc["c"].name, "text_prompt": pc["text"], "gt_class": pc["gt"],
                         "n_exemplars": (len(pc["tiles"]) if mode == "strip"
                                         else pc["n"] if mode == "in_image" else 0),
                         "exemplars_per": ("run" if mode == "strip" else
                                           "image" if mode == "in_image" else None)}
                        for pc in per_class],
            "class_name": classes[0].name,
            "text_prompt": cfg.inference.text_prompt,
            "gt_class": cfg.dataset.gt_class,
            "strip_side": cfg.strip.side,
            "tile_max_edge": cfg.exemplars.tile_max_edge,
            "exemplar_scale_matched": True,
            "exemplar_pad": cfg.exemplars.pad,
            "keep_rule": "centre",
            "confidence_threshold": cfg.inference.conf_threshold,
            "max_edge": cfg.inference.max_edge,
            "save_overlays": cfg.debug.save_overlays,
            "project": project.name,
            "run": run.name,
            "image_dir": str(image_dir),
            "anno_dir": str(anno_dir) if anno_dir else None,
            "predictions_dir": str(pred_dir),
            "prediction_format": ("class cx cy w h provenance quality score; class is "
                                  "a name (one of the run's classes); cxcywh normalised "
                                  "against the ORIGINAL image size"),
            "prediction_provenance": cfg.predictions.provenance,
            "prediction_quality": cfg.predictions.quality,
            "images_processed": n_done,
            "instances_total": n_inst,
            "prediction_lines_written": n_written,
            "strip_detections_dropped": n_tile,
            "per_image": per_image,
        }, indent=2, default=str))
    progress.update(force=True, phase="done", done=n_done)


def cli() -> None:
    ap = argparse.ArgumentParser(
        prog="weedloop-predict",
        description="Run SAM 3 over a project's images using pasted exemplar tiles.",
    )
    ap.add_argument("--config", required=True, type=Path,
                    help="path to the YAML settings file")
    args = ap.parse_args()
    main(load_config(args.config))


if __name__ == "__main__":
    cli()