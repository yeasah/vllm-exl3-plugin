#!/usr/bin/env bash
# Install vllm-exl3-plugin into the active Python environment: exllamav3 (our fork, with its CUDA
# extension built here), the patched vLLM (our fork, using the released v0.32.0 wheel's binaries),
# and the plugin itself. Run it from the repository root inside a venv.
#
#   ./install.sh           everything, starting with vLLM 0.32.0's dependencies if vLLM is absent
#   ./install.sh --extras  also flash-linear-attention and causal-conv1d (see below)
#   ./install.sh --check   report what is installed and whether it is the expected build
#
# --extras: vLLM carries its own kernels for linear attention and causal conv1d, so native serving
# needs neither package. transformers does not: the Transformers backend on a linear-attention model,
# and the quantization and eval tooling, fall back to slow reference code without them while saying
# they are "not installed". causal-conv1d is built here against this torch, because a prebuilt or
# older build fails to import on an ABI mismatch.
#
# Each step is what README.md "Install" lists by hand; patches.md explains the vLLM part.
set -euo pipefail

VLLM_TAG=v0.32.0
VLLM_WHEEL=https://files.pythonhosted.org/packages/a5/0d/0b15fda9a56ac485ebb607e466f0e3386c8c115bafeced6514298bd325b5/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl
BUILD_REQS=('cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0'
            'setuptools-rust>=1.9.0' wheel jinja2)

cd "$(dirname "$0")"
say() { printf '\n=== %s\n' "$*"; }
die() { printf '!! %s\n' "$*" >&2; exit 1; }

check() {
    # Isolated (-I), and by installed metadata rather than import: run from the repository root, an
    # import of vllm_exl3_plugin finds the source directory whether or not anything is installed
    python -I - <<'PY'
from importlib import metadata
for dist in ("torch", "vllm", "exllamav3", "vllm-exl3-plugin", "flash-linear-attention", "causal-conv1d"):
    try:
        print(f"{dist:24} {metadata.version(dist)}")
    except metadata.PackageNotFoundError:
        print(f"{dist:24} {'not installed (optional, --extras)' if dist in ('flash-linear-attention', 'causal-conv1d') else 'MISSING'}")
if any(not __import__("importlib.util").util.find_spec(m) for m in ("fla", "causal_conv1d")):
    print("  extras without a reinstall: pip install flash-linear-attention && pip install --no-build-isolation causal-conv1d")
try:
    v = metadata.version("vllm")
    print(f"{'vllm build':24}", "ok: the patched fork" if "+g" in v and v.startswith("0.32") else f"NOT the fork ({v})")
except metadata.PackageNotFoundError:
    pass
try:
    from exllamav3.ext import exllamav3_ext  # noqa: F401
    print(f"{'exllamav3 extension':24} built")
except Exception as e:
    print(f"{'exllamav3 extension':24} NOT BUILT ({type(e).__name__})")
PY
}

EXTRAS=0
case "${1:-}" in
    --check) check; exit 0 ;;
    --extras) EXTRAS=1 ;;
    "") ;;
    *) die "unknown option ${1}; see the header of this script" ;;
esac

[ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ] || die "activate a virtual environment first (python -m venv .venv && . .venv/bin/activate)"
python -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "Python 3.10 or newer is required"
command -v nvidia-smi > /dev/null || die "no nvidia-smi: this install is for NVIDIA GPUs (compute capability 8.0+)"

say "submodules"
git submodule update --init deps/exllamav3 deps/vllm

if ! python -c 'import vllm' 2> /dev/null; then
    say "vLLM 0.32.0 and its dependencies (replaced by the fork below)"
    pip install "vllm==${VLLM_TAG#v}"
fi

say "exllamav3, building its CUDA extension"
pip install --no-deps --no-build-isolation -e deps/exllamav3 \
    || die "the exllamav3 build failed; if CUDA_HOME's toolkit does not match torch's, see docs/tensor-parallel.md \"Finishing this\""

say "the patched vLLM, from the released wheel's binaries"
pip install "${BUILD_REQS[@]}"
# Before the install: setuptools-scm derives the version from this tag (patches.md says why it matters)
git -C deps/vllm fetch --no-tags https://github.com/vllm-project/vllm.git tag "$VLLM_TAG"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_WHEEL" \
    pip install --no-deps --no-build-isolation -e deps/vllm

say "the plugin"
pip install --no-deps -e .

if [ "$EXTRAS" = 1 ]; then
    say "extras: flash-linear-attention, and causal-conv1d built against this torch"
    pip install flash-linear-attention
    pip install --no-build-isolation causal-conv1d
fi

say "installed"
check
cat <<'EOF'

Try it:
    vllm serve turboderp/Llama-3.2-1B-Instruct-exl3 --revision 3.0bpw
For more than one GPU on torch <= 2.14.1, also run tools/torch-c10cuda-fix.sh (README "Requirements").
vLLM warns at startup that FlashInfer's precompiled kernels are missing: `vllm download-kernels`
fetches them (about 1.5 GB); without them the few FlashInfer ops used are compiled on first use.
EOF
