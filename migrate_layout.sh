#!/usr/bin/env bash
# Reshape one existing output folder into the project layout.
#
#   old:  <OLD>/annotations  <OLD>/flags  <OLD>/<run>/{predictions,overlays,...}
#   new:  <ROOT>/<PROJECT>/{images,annotations,flags,runs/<run>/...}
#
# Dry run by default: prints what it would do and touches nothing. Add --go to
# execute. Nothing is deleted - directories are MOVED, and the image folder
# becomes a symlink, so the read-only share is never written to and no image
# bytes are copied.
#
#   ./migrate_layout.sh "$HOME/Outputs/TillerCounter" "TillerCounter" \
#       "/agroscope/.../02_for_upload/2023/01"
#   ./migrate_layout.sh ... --go

set -euo pipefail

ROOT="${ROOT:-$HOME/AnnotationProjects}"

OLD="${1:?usage: $0 <old_output_base> <project_name> <image_dir> [--go]}"
PROJECT="${2:?missing project name}"
IMAGES="${3:?missing image dir}"
GO="${4:-}"

DEST="$ROOT/$PROJECT"
run() { if [ "$GO" = "--go" ]; then eval "$@"; else echo "  would: $*"; fi; }

echo "root     $ROOT"
echo "project  $PROJECT  ->  $DEST"
echo "images   $IMAGES"
[ "$GO" = "--go" ] || echo "(dry run - pass --go as the 4th argument to execute)"
echo

[ -d "$OLD" ] || { echo "no such old output base: $OLD" >&2; exit 1; }
[ -d "$IMAGES" ] || { echo "no such image dir: $IMAGES" >&2; exit 1; }
[ -e "$DEST" ] && { echo "destination already exists: $DEST" >&2; exit 1; }

run "mkdir -p '$DEST/runs'"

# images: a pointer, never a copy. Keeps the share read-only and the quota free.
run "ln -s '$IMAGES' '$DEST/images'"

# project-scoped: the irreplaceable half
for d in annotations flags; do
  if [ -d "$OLD/$d" ]; then
    run "mv '$OLD/$d' '$DEST/$d'"
  else
    run "mkdir -p '$DEST/$d'"
  fi
done

# optional shipped ground truth, if you kept any
[ -d "$OLD/gt" ] && run "mv '$OLD/gt' '$DEST/gt'"

# everything else at the top level of OLD that looks like a run
shopt -s nullglob
for p in "$OLD"/*/; do
  name="$(basename "$p")"
  case "$name" in
    annotations|flags|gt) continue ;;
  esac
  echo "run: $name"
  run "mv '$p' '$DEST/runs/$name'"
  # Tiles used to share the exemplars/ directory with the manifest; the run
  # now keeps them apart. Tested against the destination - the source has
  # already moved by this point.
  if [ -d "$p/exemplars" ] || [ -d "$DEST/runs/$name/exemplars" ]; then
    run "mkdir -p '$DEST/runs/$name/exemplar_tiles'"
    run "find '$DEST/runs/$name/exemplars' -maxdepth 1 -name 'tile_*.png' -exec mv {} '$DEST/runs/$name/exemplar_tiles/' ';'"
  fi
done

echo
echo "done. Then set in the YAML:"
echo "  paths:"
echo "    root: $ROOT"
echo "    dataset: $PROJECT"
