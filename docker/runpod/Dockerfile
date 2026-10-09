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

# torch's libc10_cuda.so rebuilt with the fix for pytorch#196258 (expandable segments unmap
# on the wrong device: Xid 31 or silent corruption on multi-GPU). Last, after every pip
# install: any torch reinstall restores the stock library. Remove once torch carries 3fda599
RUN TORCH_FIX_WORK=/tmp/pytorch-c10fix bash vllm-exl3-plugin/tools/torch-c10cuda-fix.sh \
    && bash vllm-exl3-plugin/tools/torch-c10cuda-fix.sh --check \
    && rm -rf /tmp/pytorch-c10fix
