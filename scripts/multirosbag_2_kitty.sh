#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  extract_all_rosbags.sh -r <SRC_ROOT> -d <DST_ROOT> [options]

Required:
  -r, --root PATH          Source dataset root to scan recursively
  -d, --dst-root PATH      Destination root under which the source's relative
                           folder structure will be preserved
                           (alias: -o, --out-root)

Options:
  -c, --config PATH        Converter config YAML
                           (default: src/core/ros_torch_converter/config/kitti_config/tartan_rgbt.yaml)
  -C, --converter PATH     Converter script
                           (default: src/core/ros_torch_converter/scripts/ros2bag_2_kitti.py)
  -n, --dry-run            Forward '--dry-run' to the converter (converter must support it)
  -f, --overwrite          Overwrite existing outputs (remove destination dir before running)
      --ext EXT            File extension to search for (default: .mcap)
  -h, --help               Show this help

Notes:
- Default behavior: skip directories whose destination already contains files.
- Each directory that contains at least one *EXT file is processed once.
- Relative structure under --root is preserved under --dst-root.
EOF
}

SRC_ROOT=""
DST_ROOT=""
CONFIG="src/core/ros_torch_converter/config/kitti_config/tartan_rgbt.yaml"
CONVERTER="src/core/ros_torch_converter/scripts/ros2bag_2_kitti.py"
DRYRUN=0
OVERWRITE=0
EXT=".mcap"

# ---- parse args (portable) ----
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--root) SRC_ROOT="$2"; shift 2 ;;
    -d|--dst-root|-o|--out-root) DST_ROOT="$2"; shift 2 ;;
    -c|--config) CONFIG="$2"; shift 2 ;;
    -C|--converter) CONVERTER="$2"; shift 2 ;;
    -n|--dry-run) DRYRUN=1; shift ;;
    -f|--overwrite) OVERWRITE=1; shift ;;
    --ext) EXT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
      usage; exit 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
set -- "${args[@]:-}"

# ---- validate paths ----
[[ -n "$SRC_ROOT" ]] || { echo "Error: --root is required"; usage; exit 2; }
[[ -n "$DST_ROOT" ]] || { echo "Error: --dst-root is required"; usage; exit 2; }

SRC_ROOT="$(realpath -m "$SRC_ROOT")"
DST_ROOT="$(realpath -m "$DST_ROOT")"

[[ -d "$SRC_ROOT" ]] || { echo "Error: source root not found: $SRC_ROOT" >&2; exit 2; }
mkdir -p "$DST_ROOT"

# normalize extension
[[ "$EXT" == .* ]] || EXT=".$EXT"

echo "Source root    : $SRC_ROOT"
echo "Dest root      : $DST_ROOT"
echo "Converter      : $CONVERTER"
echo "Config         : $CONFIG"
echo "Dry run (fwd)  : $DRYRUN"
echo "Overwrite      : $OVERWRITE"
echo "Search ext     : $EXT"
echo

[[ -f "$CONVERTER" ]] || echo "Warning: converter not found at $CONVERTER" >&2
[[ -f "$CONFIG" ]] || echo "Warning: config not found at $CONFIG" >&2

# ---- find unique directories that contain at least one *EXT file ----
mapfile -d '' bag_dirs < <(
  find "$SRC_ROOT" -type f -name "*$EXT" -printf '%h\0' | sort -zu
)

if (( ${#bag_dirs[@]} == 0 )); then
  echo "No *$EXT files found under $SRC_ROOT"
  exit 0
fi

for src_dir in "${bag_dirs[@]}"; do
  # compute path relative to SRC_ROOT
  rel="${src_dir#${SRC_ROOT%/}/}"
  rel="${rel#./}"

  dst_dir="$DST_ROOT/$rel"

  if [[ -d "$dst_dir" ]]; then
    if (( OVERWRITE == 1 )); then
      echo "[CLEAN] $dst_dir"
      rm -rf -- "$dst_dir"
    else
      if compgen -G "$dst_dir/*" > /dev/null 2>&1; then
        echo "[SKIP ] $src_dir -> $dst_dir (already contains files)"
        continue
      fi
      # if dir exists but is empty, proceed
    fi
  fi

  echo "[RUN  ] $src_dir -> $dst_dir"

  cmd=( python3 "$CONVERTER"
        --config "$CONFIG"
        --src_dir "$src_dir"
        --dst_dir "$dst_dir" )

  (( DRYRUN == 1 )) && cmd+=( --dryrun )

  # Stream converter output live
  "${cmd[@]}"
done

echo
echo "Done."
