# nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
# nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04
FROM "runpod/base:1.4.0-cuda1300-ubuntu2404"

# environment
ENV HOME=/root
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0"
ENV VIRTUAL_ENV=$HOME/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"

WORKDIR $HOME

RUN git clone https://github.com/yeasah/vllm-exl3-plugin.git -b dev --recurse-submodules

# install vllm cuda dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r vllm-exl3-plugin/deps/vllm/requirements/cuda.txt

# vllm build dependencies (for --no-build-isolation)
RUN pip install --no-cache-dir \
    -r vllm-exl3-plugin/deps/vllm/requirements/build/cuda.txt

# build vllm with precompiled wheel (0.29.0 cu130)
RUN VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl \
    pip install --no-cache-dir --no-build-isolation \
    -e vllm-exl3-plugin/deps/vllm

# install llama-cpp-python precompiled
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130

# build exllamav3 + eval
RUN pip install --no-cache-dir --no-build-isolation \
    -e vllm-exl3-plugin/deps/exllamav3[eval]

# build plugin + quantization
RUN pip install --no-cache-dir -e vllm-exl3-plugin[quantization]

# Bring the checkout up to dev's head without rebuilding the layers above. The clone layer is
# cached by its command text, so a rebuild otherwise keeps the code from the first build; this
# ADD's content is dev's current commit, so the cache breaks here exactly when dev has moved.
# Editable installs pick up the pulled code; extras declared since are installed. A pull that
# changes exllamav3's CUDA sources still needs its extension rebuilt (pip install -e ... again)
ADD https://api.github.com/repos/yeasah/vllm-exl3-plugin/git/refs/heads/dev /tmp/dev-ref.json
RUN cd vllm-exl3-plugin \
    && git pull --ff-only --recurse-submodules \
    && git submodule update --init --recursive \
    && pip install --no-cache-dir $(python -c 'import tomllib; print(" ".join( \
        d for f, e in [("deps/exllamav3/pyproject.toml", "eval"), ("pyproject.toml", "quantization")] \
        for d in tomllib.load(open(f, "rb"))["project"]["optional-dependencies"][e]))') \
    && git log --oneline -1

# causal-conv1d, built against this image's torch: a prebuilt or older build fails to import on
# an ABI mismatch, and transformers then falls back to its slow reference path while saying the
# package "is not installed". (flash-linear-attention, pure Triton, comes with the eval extras.)
RUN pip install --no-cache-dir --no-build-isolation causal-conv1d

# torch's libc10_cuda.so rebuilt with the fix for pytorch#196258 (expandable segments unmap
# on the wrong device: Xid 31 or silent corruption on multi-GPU). Last, after every pip
# install: any torch reinstall restores the stock library. Remove once torch carries 3fda599
RUN TORCH_FIX_WORK=/tmp/pytorch-c10fix bash vllm-exl3-plugin/tools/torch-c10cuda-fix.sh \
    && bash vllm-exl3-plugin/tools/torch-c10cuda-fix.sh --check \
    && rm -rf /tmp/pytorch-c10fix

# The fast kernels transformers looks for on hybrid models (Qwen3.5+, gated delta nets): any
# import error here would otherwise only show as a slow reference pass at run time
RUN python -c "import fla, causal_conv1d; print('fla', fla.__version__, 'causal_conv1d', causal_conv1d.__version__)"
