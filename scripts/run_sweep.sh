#!/usr/bin/env bash
# The only way to start GPU measurements: sets the environment, shows who else is on the
# GPU, runs one sweep under a timeout and checks that nothing is left running.
#
# Usage: bash scripts/run_sweep.sh [--timeout 45m] --model <id> --grid <name> [--kv-cache-dtype auto|fp8]
#            [--points BATCHxCONTEXT ...]
# Everything except --timeout is passed to `python -m decode_sol.measure`.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
timeout_duration=45m
measure_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)
            timeout_duration=$2
            shift 2
            ;;
        *)
            measure_args+=("$1")
            shift
            ;;
    esac
done

cd "$repo_dir"
source .venv/bin/activate
export HF_HOME=/tmp/$USER/hf
export CPATH=$repo_dir/.venv/python3.12-dev/usr/include/python3.12:$repo_dir/.venv/python3.12-dev/usr/include

echo "GPU processes before the sweep (must be empty):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# timeout signals the whole process group, so vLLM's engine subprocess goes with the main one.
status=0
timeout --kill-after=30s "$timeout_duration" python -m decode_sol.measure "${measure_args[@]}" || status=$?
if [[ $status -eq 124 ]]; then
    echo "Sweep timed out after $timeout_duration. Records appended before that are kept." >&2
elif [[ $status -ne 0 ]]; then
    echo "Sweep failed with exit status $status." >&2
fi

echo "GPU processes after the sweep (must be empty):"
leftover=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "$leftover" ]]; then
    echo "$leftover"
    echo "A process is still on the GPU. Find it with ps and stop it before leaving." >&2
    exit 1
fi
exit $status
