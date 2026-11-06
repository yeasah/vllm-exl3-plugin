ARG CUDAVERS=13.0.3
ARG BASEOS=ubuntu24.04

FROM "nvidia/cuda:${CUDAVERS}-cudnn-devel-${BASEOS}" AS build

ARG BRANCH=dev
ARG VLLM_WHEEL=https://files.pythonhosted.org/packages/a5/0d/0b15fda9a56ac485ebb607e466f0e3386c8c115bafeced6514298bd325b5/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl
ARG VLLM_BASE_TAG=v0.32.0

# install needed things
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git python3 python-is-python3 python3-dev build-essential python3.12-venv curl && \
    rm -rf /var/lib/apt/lists/*

# environment
ENV HOME=/root
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0"
ENV VIRTUAL_ENV=$HOME/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
WORKDIR $HOME

# clone the plugin and dependencies
ADD https://api.github.com/repos/yeasah/vllm-exl3-plugin/git/refs/heads/$BRANCH /tmp/ref.json
RUN echo "vllm base $VLLM_BASE_TAG: $VLLM_WHEEL" \
    && git clone https://github.com/yeasah/vllm-exl3-plugin.git -b $BRANCH --recurse-submodules

# install vllm cuda dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r vllm-exl3-plugin/deps/vllm/requirements/cuda.txt

# build vllm with the released base wheel's binaries (our series is Python only; patches.md).
# setuptools-scm versions it from `git describe` at install time and freezes the answer, and a
# submodule clone has no tags -- nor does our fork carry the base tag -- so fetch it from upstream
# first; the version must read <base>.devN+g<ours>.precompiled (e.g. 0.32.1.dev16+g1f3b03290),
# never 0.1.devN
RUN git -C vllm-exl3-plugin/deps/vllm fetch -q --no-tags \
        https://github.com/vllm-project/vllm.git tag $VLLM_BASE_TAG \
    && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_WHEEL \
    pip install --no-cache-dir ./vllm-exl3-plugin/deps/vllm[audio] \
    && python -c 'import vllm; print(vllm.__version__); assert not vllm.__version__.startswith("0.1.dev")'

# install exllamav3
RUN pip install --no-cache-dir ./vllm-exl3-plugin/deps/exllamav3

# install plugin
RUN pip install --no-cache-dir ./vllm-exl3-plugin

# optional, only needed for transformers
#RUN pip install --no-cache-dir causal-conv1d
#RUN pip install --no-cache-dir flash-linear-attention

# torch's libc10_cuda.so rebuilt with the fix for pytorch#196258 (expandable segments unmap
# on the wrong device: Xid 31 or silent corruption on multi-GPU). Last, after every pip
# install: any torch reinstall restores the stock library. Remove once torch carries 3fda599
RUN TORCH_FIX_WORK=/tmp/pytorch-c10fix bash vllm-exl3-plugin/tools/torch-c10cuda-fix.sh \
    && bash vllm-exl3-plugin/tools/torch-c10cuda-fix.sh --check \
    && rm -rf /tmp/pytorch-c10fix


# base omits the toolkit, 421MiB vs 7.88GiB.
# including nvcc -- needed for flashinfer jit, etc, and flashinfer's cubins alone are more than that
#FROM "nvidia/cuda:${CUDAVERS}-base-${BASEOS}" AS serve
FROM "nvidia/cuda:${CUDAVERS}-cudnn-devel-${BASEOS}" AS serve

# install needed things
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    python3 python-is-python3 python3-dev python3.12-venv curl && \
    rm -rf /var/lib/apt/lists/*

# environment
ENV HOME=/root
ENV VIRTUAL_ENV=$HOME/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR $HOME

# copy the venv from the build step
COPY --from=build /root/.venv /root/.venv

# this actually ends up being 12GiB installed
#RUN flashinfer install-cubin-wheel && flashinfer install-jit-cache-wheel

ENTRYPOINT [ "vllm", "serve" ]

# default example that will work on most hardware
CMD [ "turboderp/Llama-3.2-1B-Instruct-exl3", "--revision", "3.0bpw", "--max-model-len", "auto", "--gpu-memory-utilization", "0.80" ]
