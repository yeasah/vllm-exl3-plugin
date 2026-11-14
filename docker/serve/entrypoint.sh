#!/bin/bash
# vllm-exl3-serve entrypoint: clear stale locks on the cache volume, then `vllm serve` with the container's args.
#
# Only existence-based locks can go stale: they block for as long as the file exists, and a process killed while
# holding one leaves it behind. torch's JIT extension builds use one (FileBaton, a bare O_CREAT|O_EXCL `lock`
# file in each build directory), and its waiters poll with no timeout -- a container stopped mid-build hung every
# later one indefinitely, CPU and GPU idle (2026-11-14). flock-based locks (filelock: huggingface_hub, vLLM's
# compile cache, FlashInfer's JIT cache) are released by the kernel when the holder dies, so a leftover file of
# that kind blocks nothing, and deleting one under a live holder would break real exclusion: those are left alone.
#
# At startup nothing in this container holds a lock yet, so every existence-based lock found is stale -- unless
# another live container shares the cache volume; set EXL3_SWEEP_STALE_LOCKS=0 there.
set -u

if [ "${EXL3_SWEEP_STALE_LOCKS:-1}" != "0" ]; then
    ext_dir="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
    for lock in "$ext_dir"/*/lock "$ext_dir"/*/*/lock; do
        [ -f "$lock" ] || continue
        echo "vllm-exl3-serve: removing stale build lock $lock (left by a process stopped mid-build;" \
             "EXL3_SWEEP_STALE_LOCKS=0 keeps it)" >&2
        rm -f -- "$lock"
    done
fi

exec vllm serve "$@"
