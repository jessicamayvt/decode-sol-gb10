#!/usr/bin/env bash
# The only way to start a profile: sets the environment, shows who else is on the GPU, runs
# one grid point under a timeout and checks that nothing is left running.
#
# Usage: bash scripts/run_profile.sh [--timeout 20m] [--nsys] --model <id> --batch-size <n>
#            --context-len <n> [--kv-cache-dtype auto|fp8]
# Without --nsys, vLLM's torch profiler writes traces under profiles/torch/. With --nsys, the
# profiled decode steps are recorded by Nsight Systems into profiles/nsys/, named like the torch
# traces, and exported to SQLite for `python -m decode_sol.profile_summary`.
# Everything except --timeout and --nsys is passed to `python -m decode_sol.profile_decode`.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
timeout_duration=20m
use_nsys=false
kv_cache_dtype=auto
profile_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) model=$2 ;;&
        --batch-size) batch_size=$2 ;;&
        --context-len) context_len=$2 ;;&
        --kv-cache-dtype) kv_cache_dtype=$2 ;;&
        --timeout)
            timeout_duration=$2
            shift 2
            ;;
        --nsys)
            use_nsys=true
            shift
            ;;
        *)
            profile_args+=("$1")
            shift
            ;;
    esac
done

cd "$repo_dir"
source .venv/bin/activate
export HF_HOME=/tmp/$USER/hf
export CPATH=$repo_dir/.venv/python3.12-dev/usr/include/python3.12:$repo_dir/.venv/python3.12-dev/usr/include

echo "GPU processes before the profile (must be empty):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

command=(python -m decode_sol.profile_decode "${profile_args[@]}")
if [[ $use_nsys == true ]]; then
    mkdir -p profiles/nsys
    # Same name as ProfileConfig.trace_prefix in profile_decode.py.
    report_name=${model##*/}_${kv_cache_dtype}_b${batch_size}_c${context_len}
    # Records only between cudaProfilerStart/Stop. vLLM's engine runs in a forked process,
    # and decode replays CUDA graphs, so trace through the fork and inside each graph.
    command=(/usr/local/cuda/bin/nsys profile --output "profiles/nsys/$report_name" --force-overwrite=true
        --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop
        --cuda-graph-trace=node --trace-fork-before-exec=true
        "${command[@]}" --profiler cuda)
else
    command+=(--profiler torch)
fi

# timeout signals the whole process group, so vLLM's engine subprocess goes with the main one.
status=0
timeout --kill-after=30s "$timeout_duration" "${command[@]}" || status=$?
if [[ $status -eq 124 ]]; then
    echo "Profile timed out after $timeout_duration." >&2
elif [[ $status -ne 0 ]]; then
    echo "Profile failed with exit status $status." >&2
fi

if [[ $use_nsys == true && $status -eq 0 ]]; then
    /usr/local/cuda/bin/nsys export --type sqlite --force-overwrite=true \
        --output "profiles/nsys/$report_name.sqlite" "profiles/nsys/$report_name.nsys-rep"
fi

echo "GPU processes after the profile (must be empty):"
leftover=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [[ -n "$leftover" ]]; then
    echo "$leftover"
    echo "A process is still on the GPU. Find it with ps and stop it before leaving." >&2
    exit 1
fi
exit $status
