#!/usr/bin/env bash
# Run the cuttings pipeline for every borehole in one go, at the end of the project when we
# need every batch's output regenerated at once.
#
# Usage:
#   scripts/run_all_cuttings.sh <input-root> <output-root> [--config path/to/config.yaml] [extra flags...]
#
# <input-root>/<borehole>/... must contain raw cuttings photos, one folder per borehole.
# Any [extra flags...] (e.g. --mlflow --debug --cache) are forwarded as-is to every invocation.
#
# Each borehole's --cut-type is looked up below. A borehole not listed there is skipped with
# a warning rather than silently defaulting to "full", since a wrong cut-type produces a
# wrong crop for the whole batch. Plain case statements (not associative arrays) on purpose:
# macOS's default /bin/bash (3.2) has no associative array support.

set -euo pipefail

# --- borehole -> --cut-type -------------------------------------------------------------
# "skip" means: don't run the cuttings pipeline for this borehole at all.
cut_type_for() {
  case "$1" in
    Montagny | Montagny-2 | Montagny-2ST) echo "tray" ;;
    Forsthaus-GES-F1 | Forsthaus-GES-F2 | Forsthaus-GES-F3 | Forsthaus-GES-F3A) echo "pebble" ;;
    GEo-01 | GEo-02 | Vinzel-1 | Vinzel-1-Malm | Vinzel-1S) echo "full" ;;
    GVL-1) echo "black_circle" ;;
    Lavey-1) echo "skip" ;;
    *) echo "" ;;
  esac
}

# Boreholes that need dedup_keep=last instead of the config's default ("first") when
# multiple cuttings images share the same depth.
needs_dedup_last() {
  case "$1" in
    GEo-01 | GEo-02) return 0 ;;
    *) return 1 ;;
  esac
}

# --- args --------------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input-root> <output-root> [--config path/to/config.yaml] [extra flags...]" >&2
  exit 1
fi

input_root=$1
output_root=$2
shift 2

config="config.yaml"
extra_args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      config="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

# config.yaml with cuttings.dedup_keep overridden to "last", keeping every other tuned
# value from $config as-is (a hand-written minimal override would fall back to the
# dataclass defaults for everything else instead of $config's tuned values).
dedup_last_config="$tmp_dir/config-dedup-last.yaml"
uv run python -c "
import yaml
with open('$config') as f:
    data = yaml.safe_load(f)
data.setdefault('segmentation', {}).setdefault('cuttings', {})['dedup_keep'] = 'last'
with open('$dedup_last_config', 'w') as f:
    yaml.safe_dump(data, f)
"

for borehole_dir in "$input_root"/*/; do
  [[ -d "$borehole_dir" ]] || continue
  borehole=$(basename "$borehole_dir")
  cut_type=$(cut_type_for "$borehole")

  if [[ -z "$cut_type" ]]; then
    echo "=== cuttings: $borehole -- SKIPPED (no --cut-type mapping, see cut_type_for) ===" >&2
    continue
  fi
  if [[ "$cut_type" == "skip" ]]; then
    echo "=== cuttings: $borehole -- skipped by config ==="
    continue
  fi

  borehole_config="$config"
  if needs_dedup_last "$borehole"; then
    borehole_config="$dedup_last_config"
  fi

  echo "=== cuttings: $borehole (--cut-type $cut_type) ==="
  uv run boreholes-photo-processor-cuttings \
    --input "$borehole_dir" \
    --output "$output_root/$borehole" \
    --config "$borehole_config" \
    --cut-type "$cut_type" \
    "${extra_args[@]+"${extra_args[@]}"}"
done
