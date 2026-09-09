#!/usr/bin/env bash
# Run the cores pipeline for every borehole in one go, at the end of the project when we
# need every batch's output regenerated at once.
#
# Usage:
#   scripts/run_all_cores.sh <input-root> <output-root> [--config path/to/config.yaml] [extra flags...]
#
# <input-root>/<borehole>/... must contain raw core .tif photos, one folder per borehole.
# Any [extra flags...] (e.g. --mlflow --debug --cache) are forwarded as-is to every invocation.

set -euo pipefail

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

for borehole_dir in "$input_root"/*/; do
  [[ -d "$borehole_dir" ]] || continue
  borehole=$(basename "$borehole_dir")
  echo "=== cores: $borehole ==="
  uv run boreholes-photo-processor \
    --input "$borehole_dir" \
    --output "$output_root/$borehole" \
    --config "$config" \
    "${extra_args[@]+"${extra_args[@]}"}"
done
