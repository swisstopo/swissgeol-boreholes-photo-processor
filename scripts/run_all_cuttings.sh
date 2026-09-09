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
# --cut-type (and, where needed, --dedup-keep) is specific to its physical setup / data and
# is hardcoded below. Add a line for each new borehole as it comes in. Lavey-1 has no line
# here on purpose — it isn't run.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input-root> <output-root> [extra flags...]" >&2
  exit 1
fi

input_root=$1
output_root=$2
shift 2

run() {
  local borehole=$1 cut_type=$2
  shift 2
  echo "=== cuttings: $borehole (--cut-type $cut_type) ==="
  uv run boreholes-photo-processor-cuttings \
    --input "$input_root/$borehole" \
    --output "$output_root/$borehole" \
    --cut-type "$cut_type" \
    "$@"
}

run Montagny tray "$@"
run Montagny-2 tray "$@"
run Montagny-2ST tray "$@"
run Forsthaus-GES-F1 pebble "$@"
run Forsthaus-GES-F2 pebble "$@"
run Forsthaus-GES-F3 pebble "$@"
run Forsthaus-GES-F3A pebble "$@"
# GEo-01 and GEo-02 have multiple cuttings images sharing the same depth; keep the last
# (by filename) instead of config.yaml's default ("first").
run GEo-01 full --dedup-keep last "$@"
run GEo-02 full --dedup-keep last "$@"
run Vinzel-1 full "$@"
run Vinzel-1-Malm full "$@"
run Vinzel-1S full "$@"
run GVL-1 black_circle "$@"
