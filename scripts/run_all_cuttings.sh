#!/usr/bin/env bash
# Run the cuttings pipeline for every borehole we have cuttings data for, e.g. at the end
# of the project when we need every batch's output regenerated at once.
#
# Usage:
#   scripts/run_all_cuttings.sh <input-root> <output-root> [extra flags...]
#
# <input-root>/<borehole> must contain that borehole's raw cuttings photos.
# Any [extra flags...] (e.g. --mlflow --debug --cache) are forwarded as-is to every run.
#
# This is a plain list of commands, one per borehole, not a generic tool: each borehole's
# --cut-type is specific to its physical setup and is hardcoded below. Add a line for each
# new borehole as it comes in. Lavey-1 has no line here on purpose — it isn't run.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input-root> <output-root> [extra flags...]" >&2
  exit 1
fi

input_root=$1
output_root=$2
shift 2

# GEo-01 and GEo-02 need cuttings.dedup_keep=last instead of config.yaml's default
# ("first") when multiple cuttings images share the same depth. Generated once here,
# from config.yaml itself, so every other tuned value is kept as-is.
dedup_last_config=$(mktemp)
trap 'rm -f "$dedup_last_config"' EXIT
uv run python -c "
import yaml
with open('config.yaml') as f:
    data = yaml.safe_load(f)
data.setdefault('segmentation', {}).setdefault('cuttings', {})['dedup_keep'] = 'last'
with open('$dedup_last_config', 'w') as f:
    yaml.safe_dump(data, f)
"

run() {
  local borehole=$1 cut_type=$2 config=${3:-config.yaml}
  shift 3
  echo "=== cuttings: $borehole (--cut-type $cut_type) ==="
  uv run boreholes-photo-processor-cuttings \
    --input "$input_root/$borehole" \
    --output "$output_root/$borehole" \
    --config "$config" \
    --cut-type "$cut_type" \
    "$@"
}

run Montagny tray "" "$@"
run Montagny-2 tray "" "$@"
run Montagny-2ST tray "" "$@"
run Forsthaus-GES-F1 pebble "" "$@"
run Forsthaus-GES-F2 pebble "" "$@"
run Forsthaus-GES-F3 pebble "" "$@"
run Forsthaus-GES-F3A pebble "" "$@"
run GEo-01 full "$dedup_last_config" "$@"
run GEo-02 full "$dedup_last_config" "$@"
run Vinzel-1 full "" "$@"
run Vinzel-1-Malm full "" "$@"
run Vinzel-1S full "" "$@"
run GVL-1 black_circle "" "$@"
