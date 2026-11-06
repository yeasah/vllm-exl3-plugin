#!/usr/bin/env bash
# Rebuild torch's libc10_cuda.so with the fix for pytorch/pytorch#196258 cherry-picked
# (pytorch/pytorch@3fda599), and swap it into the active venv's torch.
#
# The bug: with expandable segments (exllamav3 turns them on), empty_cache() and the
# allocator's release-and-retry unmap freed pages on every device but synchronize only the
# caller's current one, so kernels still queued on another GPU write into unmapped (Xid 31)
# or remapped (silent corruption) memory. docs/upstream.md "PyTorch" has the history.
#
# Only libc10_cuda.so contains the caching allocator, so only it is rebuilt: the wheel's own
# commit (torch.version.git_version), the fix's one hunk, the wheel's compile flags and
# headers, CUDA 13.0 headers from the pip nvidia packages. The install refuses unless every
# function symbol the original exports is exported again. Remove this once the minimum
# torch release contains 3fda599.
#
#   tools/torch-c10cuda-fix.sh           build, verify symbols, install (original kept as .orig-<version>)
#   tools/torch-c10cuda-fix.sh --check   report whether the installed library is the patched build
set -euo pipefail

FIX=3fda5993c2572eca77dd53033f057dc0290f287f
WORK=${TORCH_FIX_WORK:-$HOME/git/pytorch-c10fix}
PY=${PYTHON:-python3}

read -r TV GITV SITE < <(cd /tmp && $PY -c "import torch, os, site; print(torch.__version__, torch.version.git_version, os.path.dirname(os.path.dirname(torch.__file__)))")
T=$SITE/torch; L=$T/lib; LIB=$L/libc10_cuda.so; ORIG=$L/libc10_cuda.so.orig-$TV; STAMP=$L/libc10_cuda.so.fix-$FIX

if [ "${1:-}" = "--check" ]; then
    if [ -f "$STAMP" ] && [ "$(sha256sum < "$LIB")" = "$(cat "$STAMP")" ]; then
        echo "torch $TV: libc10_cuda.so is the patched build (fix $FIX)"; exit 0
    fi
    echo "torch $TV: libc10_cuda.so is NOT patched for pytorch#196258 -- run tools/torch-c10cuda-fix.sh" >&2; exit 1
fi

# Source: exactly the wheel's commit, plus the fix's allocator hunk only (not its test)
mkdir -p "$WORK"; cd "$WORK"
[ -d .git ] || { git init -q .; git remote add origin https://github.com/pytorch/pytorch.git; }
git fetch -q --depth 1 origin "$GITV"; git checkout -q -f FETCH_HEAD
curl -sfL "https://github.com/pytorch/pytorch/commit/$FIX.patch" -o fix.patch
git apply --include='c10/cuda/CUDACachingAllocator.cpp' fix.patch
git diff --stat

# Compile with the wheel's flags (torch.__config__.show(): CXX_FLAGS, plus c10/cuda/CMakeLists.txt)
B=$WORK/build-c10cuda; rm -rf "$B"; mkdir -p "$B"; cd "$B"
NV=$SITE/nvidia/cu13
FLAGS="-O2 -fPIC -std=c++20 -fvisibility=hidden -fvisibility-inlines-hidden -DNDEBUG -DC10_NODEPRECATED
       -D_GLIBCXX_USE_CXX11_ABI=1 -DC10_CUDA_BUILD_MAIN_LIB -DPYTORCH_C10_DRIVER_API_SUPPORTED
       -I$T/include -I$NV/include -idirafter /usr/local/cuda/include"
SRCS="CUDAAllocatorConfig CUDACachingAllocator CUDADeviceAssertionHost CUDAException CUDAFunctions
      CUDAMallocAsyncAllocator CUDAMiscFunctions CUDAStream PeerToPeerAccess impl/CUDAGuardImpl
      impl/CUDATest driver_api"
for f in $SRCS; do g++ $FLAGS -c "$WORK/c10/cuda/$f.cpp" -o "$(echo "$f" | tr / _).o" & done; wait
g++ -shared -o libc10_cuda.so ./*.o -Wl,-soname,libc10_cuda.so -L"$L" -lc10 -L"$NV/lib" -l:libcudart.so.13 -ldl \
    -Wl,-rpath,'$ORIGIN/../../nvidia/cudnn/lib:$ORIGIN/../../nvidia/nvshmem/lib:$ORIGIN/../../nvidia/nccl/lib:$ORIGIN/../../nvidia/cusparselt/lib:$ORIGIN/../../nvidia/cu13/lib:$ORIGIN' \
    -Wl,--disable-new-dtags

# Every function the original exports (type T) must be exported again; weak STL template
# instantiations and vtables legitimately differ between compiler versions
REF=$LIB; [ -f "$ORIG" ] && REF=$ORIG
missing=$(comm -23 <(nm -DC --defined-only "$REF" | awk '$2 == "T" {$1=""; $2=""; print}' | sort) \
                   <(nm -DC --defined-only libc10_cuda.so | awk '$2 == "T" {$1=""; $2=""; print}' | sort))
if [ -n "$missing" ]; then echo "!! rebuilt library lacks exported functions:"; echo "$missing"; exit 1; fi
# grep reads everything (no -q): under pipefail, grep -q exiting at its first match SIGPIPEs
# nm, and the pipeline then fails on a match -- a false "compiled out" on Ubuntu 24.04
nm -C libc10_cuda.so | grep 'ExpandableSegment::unmapHandles' >/dev/null || { echo "!! expandable segments compiled out"; exit 1; }

[ -f "$ORIG" ] || cp -p "$LIB" "$ORIG"
# Write alongside and rename over: a process with torch loaded has the old file mapped, and
# rewriting it in place would pull pages out from under it
cp libc10_cuda.so "$LIB.new"; mv -f "$LIB.new" "$LIB"; sha256sum < "$LIB" > "$STAMP"
echo "installed patched libc10_cuda.so into torch $TV (original: $ORIG)"
