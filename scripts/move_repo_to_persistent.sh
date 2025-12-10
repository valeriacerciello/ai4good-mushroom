#!/usr/bin/env bash
# move_repo_to_persistent.sh
# Safely copy this repository to a cluster persistent location and optionally move it
# - Preserves .git
# - Defaults to dry-run; requires --move to delete source
# - Can create a symlink at the original location after moving

set -euo pipefail

PROG_NAME=$(basename "$0")

print_usage(){
  cat <<EOF
Usage: $PROG_NAME [options]

Options:
  -t, --target PATH     Explicit target path to copy into (overrides detection)
  -g, --group NAME      Group or project folder name to place under detected persistent root (default: group)
      --dry-run         Show actions without making changes (default)
      --move            After copying, remove original and replace with symlink to target
      --symlink         Create symlink from original path to new target (only valid with --move)
  -y, --yes             Accept prompts and run non-interactively
  -h, --help            Show this help
EOF
}

# parse args
TARGET=""
GROUP="group"
DRY_RUN=1
DO_MOVE=0
DO_SYMLINK=0
ASSUME_YES=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    -t|--target) TARGET="$2"; shift 2;;
    -g|--group) GROUP="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --move) DRY_RUN=0; DO_MOVE=1; shift;;
    --symlink) DO_SYMLINK=1; shift;;
    -y|--yes) ASSUME_YES=1; shift;;
    -h|--help) print_usage; exit 0;;
    *) echo "Unknown arg: $1"; print_usage; exit 2;;
  esac
done

if [ $DO_SYMLINK -eq 1 ] && [ $DO_MOVE -eq 0 ]; then
  echo "--symlink only makes sense together with --move" >&2
  exit 2
fi

SRC_DIR=$(pwd)
PROJECT_NAME=$(basename "$SRC_DIR")

# detect persistent and scratch like the suggester
PREFER_PERSIST=("/cluster/work" "/work" "/projects" "/project" "/data" "/srv")
found_persist=""
for p in "${PREFER_PERSIST[@]}"; do
  if [ -d "$p" ]; then
    found_persist="$p"
    break
  fi
done

if [ -z "$TARGET" ]; then
  if [ -n "$found_persist" ]; then
    TARGET="$found_persist/$GROUP/${USER}/$PROJECT_NAME"
  else
    echo "No persistent location detected and no --target provided. Aborting." >&2
    exit 3
  fi
fi

echo "Source: $SRC_DIR"
echo "Target: $TARGET"
echo "Operation: ${DRY_RUN:+dry-run}${DRY_RUN:-move}"
if [ $DO_MOVE -eq 1 ]; then
  echo "After copy: will remove source and${DO_SYMLINK:+ create symlink at original location}" 
fi

if [ $ASSUME_YES -eq 0 ]; then
  read -r -p "Proceed with these settings? [y/N] " yn || true
  case "$yn" in
    [Yy]*) ;;
    *) echo "Aborted."; exit 0;;
  esac
fi

# create target parent
mkdir -p "$(dirname "$TARGET")"

RSYNC_OPTS=("-aH" "--info=progress2" "--exclude" "venv" "--exclude" "__pycache__")

echo
echo "Running rsync to copy repository (this will preserve .git)"
echo "rsync ${RSYNC_OPTS[*]} \"$SRC_DIR/\" \"$TARGET/\""

if [ $DRY_RUN -eq 1 ]; then
  rsync --dry-run "${RSYNC_OPTS[@]}" "$SRC_DIR/" "$TARGET/"
  echo
  echo "Dry-run complete. To actually perform the move, re-run with --move --yes" 
  exit 0
fi

# perform actual copy
rsync "${RSYNC_OPTS[@]}" "$SRC_DIR/" "$TARGET/"

echo "Copy completed. Target at: $TARGET"

if [ $DO_MOVE -eq 1 ]; then
  echo "Verifying copied files..."
  src_count=$(find "$SRC_DIR" -mindepth 1 | wc -l || true)
  tgt_count=$(find "$TARGET" -mindepth 1 | wc -l || true)
  echo "Source items: $src_count, Target items: $tgt_count"

  if [ $tgt_count -lt $src_count ]; then
    echo "Warning: target has fewer items than source. Aborting move to avoid data loss." >&2
    exit 5
  fi

  if [ $ASSUME_YES -eq 0 ]; then
    read -r -p "Remove source directory $SRC_DIR and replace with symlink to $TARGET? [y/N] " yn2 || true
    case "$yn2" in
      [Yy]*) : ;;
      *) echo "Skipping removal. Move finished with copy only."; exit 0;;
    esac
  fi

  # move: remove source contents and create symlink
  echo "Removing source directory contents..."
  # move safety: move to a tmp backup location first
  BACKUP="${SRC_DIR}_backup_$(date +%s)"
  mv "$SRC_DIR" "$BACKUP"
  ln -s "$TARGET" "$SRC_DIR"
  echo "Source moved to backup: $BACKUP; symlink created at $SRC_DIR -> $TARGET"

  if [ $DO_SYMLINK -eq 1 ]; then
    echo "Symlink already created at $SRC_DIR -> $TARGET"
  fi
fi

echo "Done. Please verify the repository at the new location and update any job scripts to point to $TARGET"

exit 0
