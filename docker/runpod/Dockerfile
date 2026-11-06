# nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
# nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04
# exllamav3's extension is rebuilt exactly when its sources change at the commit dev pins, not on
# every push to the fork (nearly all of them Python). This stage reruns whenever dev moves (the ADD)
# but only writes the git object ids of the extension's inputs, fetched blobless; the main stage
# COPYs that file, and a COPY is cached by content, so the rebuild below it reruns only when an id
# changes. The paths are every input setup.py compiles from: the sources, setup.py itself, and the
# nvcc flags it loads from cuda_flags.py
FROM "runpod/base:1.4.0-cuda1300-ubuntu2404" AS exl3-ext-key
ARG EXL3_EXT_INPUTS="exllamav3/exllamav3_ext setup.py exllamav3/util/cuda_flags.py"
ADD https://api.github.com/repos/yeasah/vllm-exl3-plugin/git/refs/heads/dev /tmp/dev-ref.json
RUN git clone -q --depth 1 --filter=blob:none --no-checkout -b dev \
        https://github.com/yeasah/vllm-exl3-plugin.git /plugin \
    && pin=$(git -C /plugin ls-tree HEAD deps/exllamav3 | awk '{print $3}') \
    && git init -q /exl3 \
    && git -C /exl3 fetch -q --depth 1 --filter=blob:none https://github.com/yeasah/exllamav3.git "$pin" \
    && for p in $EXL3_EXT_INPUTS; do echo "$(git -C /exl3 rev-parse -q --verify "FETCH_HEAD:$p" || echo none) $p"; done > /exl3-ext-key \
    && cat /exl3-ext-key

FROM "runpod/base:1.4.0-cuda1300-ubuntu2404"

# environment
ENV HOME=/root
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0"
ENV VIRTUAL_ENV=$HOME/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"

WORKDIR $HOME

# The vLLM bump is the one place to edit for a new vLLM base. Declared above the clone on purpose:
# everything below reads the clone (vLLM's requirements, the code the wheel's binaries must match),
# and a clone layer cached from before the bump would install the old pin's requirements and code
# under the new binaries. A changed value re-runs the clone and every step after it, which a new
# vLLM (new torch) needs anyway
ARG VLLM_WHEEL=https://files.pythonhosted.org/packages/a5/0d/0b15fda9a56ac485ebb607e466f0e3386c8c115bafeced6514298bd325b5/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl
ARG VLLM_BASE_TAG=v0.32.0
RUN echo "vllm base $VLLM_BASE_TAG: $VLLM_WHEEL" \
    && git clone https://github.com/yeasah/vllm-exl3-plugin.git -b dev --recurse-submodules

# install vllm cuda dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r vllm-exl3-plugin/deps/vllm/requirements/cuda.txt

# vllm build dependencies (for --no-build-isolation)
RUN pip install --no-cache-dir \
    -r vllm-exl3-plugin/deps/vllm/requirements/build/cuda.txt

# build vllm with the released base wheel's binaries (our series is Python only; patches.md).
# setuptools-scm versions it from `git describe` at install time and freezes the answer, and a
# submodule clone has no tags -- nor does our fork carry the base tag -- so fetch it from upstream
# first; the version must read <base>.devN+g<ours>.precompiled (e.g. 0.32.1.dev16+g1f3b03290),
# never 0.1.devN
RUN git -C vllm-exl3-plugin/deps/vllm fetch -q --no-tags \
        https://github.com/vllm-project/vllm.git tag $VLLM_BASE_TAG \
    && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_WHEEL \
    pip install --no-cache-dir --no-deps --no-build-isolation \
    -e vllm-exl3-plugin/deps/vllm \
    && python -c 'import vllm; print(vllm.__version__); assert not vllm.__version__.startswith("0.1.dev")'
# FlashInfer's precompiled kernels (flashinfer-cubin, which <= 0.29 installed as a requirement;
# `vllm download-kernels` since) are left out on purpose despite vLLM's startup warning: 1.5 GB
# of image on every fresh host, against a first-run JIT of the few ops we use (the sampler)

# install llama-cpp-python precompiled
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130

# exllamav3 + eval, without compiling the extension: its dependencies change rarely and are cached
# here; the extension is built in place further down, keyed on its sources (exl3-ext-key)
RUN EXLLAMA_NOCOMPILE=1 pip install --no-cache-dir --no-build-isolation \
    -e vllm-exl3-plugin/deps/exllamav3[eval]

# build plugin + quantization
RUN pip install --no-cache-dir -e vllm-exl3-plugin[quantization]

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

# Install runpodctl for pod management (scaling up GPUs)
ARG RUNPODCTL_VERSION=v2.7.2
ARG RUNPODCTL_SHA256=8216810f9210c3e18c648d828f4d378cdaac0930389b460b1574ff2a3dfd1624
RUN wget -q --https-only --secure-protocol=TLSv1_2 --max-redirect=3 \
    -O /tmp/runpodctl.tar.gz \
    "https://github.com/runpod/runpodctl/releases/download/${RUNPODCTL_VERSION}/runpodctl-linux-amd64.tar.gz" && \
    echo "${RUNPODCTL_SHA256}  /tmp/runpodctl.tar.gz" | sha256sum -c - && \
    tar -xz -C /usr/local/bin -f /tmp/runpodctl.tar.gz runpodctl && \
    rm /tmp/runpodctl.tar.gz

# vast.ai CLI in a venv of its own, only its entry point on PATH: it pins exact versions
# (pillow 12.2) that, installed beside vLLM, downgrade its dependencies below their floors
RUN python -m venv /opt/vastai \
    && /opt/vastai/bin/pip install --no-cache-dir vastai \
    && ln -s /opt/vastai/bin/vastai /usr/local/bin/vastai

# install telegraf
COPY ./influxdata.list /etc/apt/sources.list.d/influxdata.list
COPY ./influxdata-archive.gpg /etc/apt/keyrings
RUN apt update && apt install --yes --no-install-recommends telegraf

# exllamav3's extension, in place (exllamav3/ext.py imports the in-tree .so, and git leaves it
# alone: it is ignored). Its key lives outside the checkout, in ~/.exl3-ext-key: an untracked
# file in the submodule marks the tree dirty, and bootstrap.sh then refuses to pull on the pod.
# Cached until exl3-ext-key's content changes; then the checkout moves to
# dev's head and is rebuilt -- refusing if dev moved the extension again since the key was taken,
# so the .so and its recorded key always describe the same sources. The refresh below checks the
# key against whatever commit it pulls, so a stale extension fails the build rather than shipping
COPY --from=exl3-ext-key /exl3-ext-key /tmp/exl3-ext-key
ARG EXL3_EXT_INPUTS="exllamav3/exllamav3_ext setup.py exllamav3/util/cuda_flags.py"
RUN cd vllm-exl3-plugin \
    && git pull -q --ff-only && git submodule update -q --init --recursive \
    && cd deps/exllamav3 \
    && for p in $EXL3_EXT_INPUTS; do echo "$(git rev-parse -q --verify "HEAD:$p" || echo none) $p"; done > $HOME/.exl3-ext-key \
    && { diff /tmp/exl3-ext-key $HOME/.exl3-ext-key \
         || { echo "dev moved exllamav3's extension sources while building; rebuild"; exit 1; }; } \
    && MAX_JOBS=$(nproc) python setup.py -q build_ext --inplace \
    && rm -rf build \
    && python -c 'import exllamav3.ext'

# Bring the checkout up to dev's head without rebuilding the layers above. The clone layer is
# cached by its command text, so a rebuild otherwise keeps the code from the first build; this
# ADD's content is dev's current commit, so the cache breaks here exactly when dev has moved.
# Editable installs pick up the pulled code; extras declared since are installed. A pull that
# changes exllamav3's extension sources was already rebuilt above (exl3-ext-key); the key check
# here catches the window where dev moved them between the two fetches
ADD https://api.github.com/repos/yeasah/vllm-exl3-plugin/git/refs/heads/dev /tmp/dev-ref.json
RUN cd vllm-exl3-plugin \
    && git pull --ff-only --recurse-submodules \
    && git submodule update --init --recursive \
    && { (cd deps/exllamav3 && for p in $EXL3_EXT_INPUTS; do echo "$(git rev-parse -q --verify "HEAD:$p" || echo none) $p"; done \
          | diff $HOME/.exl3-ext-key -) || { echo "exllamav3's extension is stale for this commit; rebuild"; exit 1; }; } \
    && pip install --no-cache-dir $(python -c 'import tomllib; print(" ".join( \
        d for f, e in [("deps/exllamav3/pyproject.toml", "eval"), ("pyproject.toml", "quantization")] \
        for d in tomllib.load(open(f, "rb"))["project"]["optional-dependencies"][e]))') \
    && git log --oneline -1

# fixup environment
COPY --chmod=755 ./bin/* .local/bin/
COPY ./bashrc .bashrc
RUN echo "StrictModes no" >> /etc/ssh/sshd_config
