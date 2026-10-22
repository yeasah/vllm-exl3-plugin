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
