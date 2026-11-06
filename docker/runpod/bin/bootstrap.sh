#!/bin/bash
# Bring a fresh pod from the vllm-exl3-plugin image up to working state.
# The editable checkout baked into the image is pulled forward, and any extras declared
# since the image was built are installed (pip skips whatever is already satisfied).
set -euo pipefail
P=/root/vllm-exl3-plugin

source /root/.venv/bin/activate
cd "$P"
if [ -n "$(git status --porcelain --ignore-submodules=none)" ]; then
    echo "!! $P has local changes; not pulling over them:"
    git status --short
    exit 1
fi
git pull --ff-only --recurse-submodules
git submodule update --init --recursive
echo "checkout: $(git log --oneline -1)"
echo "exllamav3: $(git -C deps/exllamav3 log --oneline -1)"

# The image built exllamav3's extension for the commit it was built at (key in ~/.exl3-ext-key,
# see the Dockerfile); a pull that moved the extension's sources leaves that .so stale, running new
# Python over old kernels without a word. Rebuild it then, for this pod's GPUs only (minutes, not the
# image's five architectures), with jobs bounded by memory: nvcc runs at a few GB each
inputs="exllamav3/exllamav3_ext setup.py exllamav3/util/cuda_flags.py"
key=$(cd deps/exllamav3 && for p in $inputs; do echo "$(git rev-parse -q --verify "HEAD:$p" || echo none) $p"; done)
if [ "$key" != "$(cat ~/.exl3-ext-key 2>/dev/null)" ]; then
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr '\n' ' ')
    jobs=$(( $(awk '/MemAvailable/ {print int($2 / 4194304)}' /proc/meminfo) ))
    jobs=$(( jobs < 1 ? 1 : (jobs > $(nproc) ? $(nproc) : jobs) ))
    echo "exllamav3's extension sources changed since the image was built; rebuilding for sm ${archs}(${jobs} jobs)"
    (cd deps/exllamav3 && TORCH_CUDA_ARCH_LIST="$archs" MAX_JOBS=$jobs python setup.py -q build_ext --inplace \
        && rm -rf build && python -c 'import exllamav3.ext')
    echo "$key" > ~/.exl3-ext-key
fi

extras=$(python - <<'PY'
import tomllib
want = [("deps/exllamav3/pyproject.toml", "eval"), ("pyproject.toml", "quantization")]
print(" ".join(d for f, e in want
               for d in tomllib.load(open(f, "rb"))["project"]["optional-dependencies"].get(e, [])))
PY
)
pip install -q $extras
echo "extras satisfied"

# The image patches torch for pytorch#196258 (multi-GPU expandable segments); say so if not
bash "$P/tools/torch-c10cuda-fix.sh" --check || echo "   (matters on multi-GPU pods: run $P/tools/torch-c10cuda-fix.sh)"
