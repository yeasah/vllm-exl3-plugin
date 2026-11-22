ARG CUDAVERS=13.0.3
ARG BASEOS=ubuntu24.04

FROM "nvidia/cuda:${CUDAVERS}-cudnn-devel-${BASEOS}" AS build

ARG BRANCH=dev
ARG VLLM_WHEEL=https://files.pythonhosted.org/packages/0b/2c/55f3b76ceadbfc6d515f16b1c17e2e69136c8e54d47be2a94b9f7326f9bb/vllm-0.34.0-cp38-abi3-manylinux_2_28_x86_64.whl
ARG VLLM_BASE_TAG=v0.34.0

# install needed things
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git python3 python-is-python3 python3-dev build-essential python3.12-venv curl && \
    rm -rf /var/lib/apt/lists/*

# environment
ENV HOME=/root
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0 10.0 12.0"
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

# install exllamav3, its CUDA extension compiled here for TORCH_CUDA_ARCH_LIST. Without --no-build-isolation its
# setup.py cannot see torch and installs no extension, so every container JIT-compiles one into /root/.cache on
# first use -- minutes of silence, and a hang forever (torch's FileBaton) if a container was stopped mid-build and
# left the lock on the cache volume
RUN pip install --no-cache-dir 'setuptools>=77' wheel ninja \
    && pip install --no-cache-dir --no-build-isolation ./vllm-exl3-plugin/deps/exllamav3 \
    && python -c 'import importlib.util as u; assert u.find_spec("exllamav3_ext"), "exllamav3 extension not prebuilt"'

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
# the entrypoint: clears stale build locks on the cache volume, then `vllm serve` (entrypoint.sh)
COPY --from=build /root/vllm-exl3-plugin/docker/serve/entrypoint.sh /usr/local/bin/vllm-exl3-entrypoint

# this actually ends up being 12GiB installed
#RUN flashinfer install-cubin-wheel && flashinfer install-jit-cache-wheel

# WSL2: vLLM never pins host memory there unless told to (platforms/cuda.py is_pin_memory_available), and says so
# in one easily missed line; unpinned offload then pages through WSL2's overcommit like a very slow weight
# offload. Read only under WSL, so harmless elsewhere
ENV VLLM_WSL2_ENABLE_PIN_MEMORY=1

ENTRYPOINT [ "/usr/local/bin/vllm-exl3-entrypoint" ]

# default example that will work on most hardware
CMD [ "turboderp/Llama-3.2-1B-Instruct-exl3", "--revision", "3.0bpw", "--max-model-len", "auto" ]
