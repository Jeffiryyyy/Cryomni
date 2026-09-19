#!/bin/sh
# Usage: sh scripts/test.sh /path/to/input.mrc [additional inference options]
set -eu

if [ "$#" -eq 0 ]; then
    echo "Usage: sh scripts/test.sh /path/to/input.mrc [additional inference options]" >&2
    exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python "$SCRIPT_DIR/infer.py" "$@"
