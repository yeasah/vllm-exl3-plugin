"""Where an evicted block's KV actually lives.

The manager decides residency and the pool reclaims the memory, but neither
moves a byte -- so without this a restored block is correctly placed and holds
whatever the pool last left in it. This is the part that makes being wrong cost
latency instead of correctness.

One pinned host buffer per layer, shaped like that layer's GPU cache but with
its own slot count. Pinned because an unpinned source makes the driver stage
through a bounce buffer and measures that instead of the link; per layer
because that is how a KV cache is laid out, so moving one block of context is
one copy per attention layer and the scatter is across the layer tensors as
much as within them.

Copies go through plain tensor indexing rather than `swap_blocks`, deliberately
and for now. `swap_blocks` computes byte offsets from a block stride, which
assumes the cache is block-major and contiguous; tensor indexing is correct
whatever `get_kv_cache_stride_order` did to the layout. The batched primitive
is the measured faster path -- `copies x 1.30 us + bytes / 54 GB/s`, against
which per-block indexing is the same cost class -- and switching to it is worth
doing behind a bit-exactness check rather than on the assumption that the
layout is what it looks like.

Copies are synchronous. The ordering the design fixed -- copy out at step N,
free at step N+1 -- is what makes overlapping them *possible*, but an
asynchronous copy that has not landed before the pool hands the block to
another request is precisely the silent corruption this project built a guard
for, so the fast version should arrive with an event to wait on rather than by
deleting the synchronisation.
"""

from __future__ import annotations

import torch


class HostTierFull(Exception):
    """No free host slot. The caller has to widen the tier or evict less."""


class HostTier:
    """A slotted, pinned host-side store for evicted KV blocks."""

    def __init__(self, kv_caches, num_slots: int, pin: bool = True):
        if not kv_caches:
            raise ValueError("no KV caches to shadow")
        self.num_slots = num_slots
        self.buffers = [
            torch.empty((num_slots, *c.shape[1:]), dtype=c.dtype,
                        device="cpu", pin_memory=pin)
            for c in kv_caches
        ]
        self.block_bytes = sum(
            c[0].numel() * c.element_size() for c in kv_caches
        )
        self._free = list(range(num_slots))
        self._slot: dict = {}
        self.stores = 0
        self.loads = 0

    # -- residency of the *host* copy ---------------------------------------

    def __contains__(self, key) -> bool:
        return key in self._slot

    def __len__(self) -> int:
        return len(self._slot)

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def store(self, key, kv_caches, block_id: int) -> int:
        """Copy one GPU block out to the host, and remember where it went.

        Storing a key that is already held overwrites it in place rather than
        taking a second slot: a block can be evicted, restored and evicted
        again, and leaking a slot per cycle would exhaust the tier in a way
        that only shows up under a long run.
        """
        slot = self._slot.get(key)
        if slot is None:
            if not self._free:
                raise HostTierFull(
                    f"all {self.num_slots} host slots are in use")
            slot = self._free.pop()
            self._slot[key] = slot
        for buf, cache in zip(self.buffers, kv_caches):
            buf[slot].copy_(cache[block_id])
        torch.cuda.synchronize()
        self.stores += 1
        return slot

    def load(self, key, kv_caches, block_id: int) -> None:
        """Copy a held block back into `block_id`, which may be anywhere.

        The destination is deliberately unconstrained: a pager that had to
        restore a block to the address it came from would be a much weaker
        mechanism, and block identity carries no positional meaning.
        """
        slot = self._slot.get(key)
        if slot is None:
            raise KeyError(f"{key!r} is not in the host tier")
        for buf, cache in zip(self.buffers, kv_caches):
            cache[block_id].copy_(buf[slot])
        torch.cuda.synchronize()
        self.loads += 1

    def release(self, key) -> None:
        slot = self._slot.pop(key, None)
        if slot is not None:
            self._free.append(slot)

    def release_request(self, req_id) -> None:
        for key in [k for k in self._slot if k[0] == req_id]:
            self.release(key)

    def stats(self) -> dict:
        return {"slots": self.num_slots, "held": len(self._slot),
                "free": len(self._free), "stores": self.stores,
                "loads": self.loads,
                "bytes_per_block": self.block_bytes}
