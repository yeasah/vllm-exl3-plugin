"""The host tier's one contract: a block survives being freed and reused.

These run on real GPU tensors shaped like a KV cache but without an engine, so
the round trip is exercised directly rather than inferred from a model's
output. The shape of every test is the same as the end-to-end one that already
passed in `tools/kv_roundtrip.py`, and for the same reason: a save/restore test
that does not destroy the original in between will pass even when the restore
does nothing, because the destination already held the right bytes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("host tier moves real device memory", allow_module_level=True)

from kv_pager.hosttier import HostTier, HostTierFull  # noqa: E402

LAYERS, BLOCKS, SHAPE = 4, 32, (2, 16, 8)


def make_caches(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.rand((BLOCKS, *SHAPE), generator=g, device="cuda",
                       dtype=torch.bfloat16) for _ in range(LAYERS)]


def scribble(caches, block_id):
    for c in caches:
        c[block_id].normal_()


def test_a_block_survives_being_destroyed_and_restored_elsewhere():
    caches = make_caches()
    tier = HostTier(caches, num_slots=8)
    original = [c[5].clone() for c in caches]

    tier.store(("r", 5), caches, 5)
    scribble(caches, 5)
    assert not torch.equal(caches[0][5], original[0]), (
        "the destroy did not change the block, so nothing was at risk"
    )

    tier.load(("r", 5), caches, 19)          # a different physical block
    for c, want in zip(caches, original):
        assert torch.equal(c[19], want), "the restored block is not bit-exact"
    assert not torch.equal(caches[0][5], original[0]), (
        "the source was quietly repaired; the test proved nothing"
    )


def test_every_layer_is_moved():
    """A tier that shadowed only layer 0 would pass a one-layer check."""
    caches = make_caches()
    tier = HostTier(caches, num_slots=4)
    original = [c[3].clone() for c in caches]
    tier.store(("r", 3), caches, 3)
    for i in range(BLOCKS):
        scribble(caches, i)
    tier.load(("r", 3), caches, 3)
    for layer, (c, want) in enumerate(zip(caches, original)):
        assert torch.equal(c[3], want), f"layer {layer} was not restored"


def test_restoring_twice_from_one_store_is_stable():
    caches = make_caches()
    tier = HostTier(caches, num_slots=4)
    original = [c[7].clone() for c in caches]
    tier.store(("r", 7), caches, 7)
    scribble(caches, 7)
    tier.load(("r", 7), caches, 7)
    scribble(caches, 11)
    tier.load(("r", 7), caches, 11)
    for c, want in zip(caches, original):
        assert torch.equal(c[7], want) and torch.equal(c[11], want)


def test_re_evicting_reuses_the_slot():
    """Otherwise a block that cycles leaks a slot per cycle."""
    caches = make_caches()
    tier = HostTier(caches, num_slots=2)
    for _ in range(10):
        tier.store(("r", 1), caches, 1)
        tier.load(("r", 1), caches, 1)
    assert tier.free_slots == 1, f"leaked slots: {tier.stats()}"


def test_the_tier_refuses_rather_than_overwrites_when_full():
    caches = make_caches()
    tier = HostTier(caches, num_slots=2)
    tier.store(("r", 0), caches, 0)
    tier.store(("r", 1), caches, 1)
    with pytest.raises(HostTierFull):
        tier.store(("r", 2), caches, 2)
    # and the two it holds are still intact
    tier.load(("r", 0), caches, 20)
    tier.load(("r", 1), caches, 21)


def test_release_returns_the_slot_and_forgets_the_block():
    caches = make_caches()
    tier = HostTier(caches, num_slots=2)
    tier.store(("r", 0), caches, 0)
    tier.release(("r", 0))
    assert tier.free_slots == 2 and ("r", 0) not in tier
    with pytest.raises(KeyError):
        tier.load(("r", 0), caches, 1)


def test_release_request_clears_only_that_request():
    caches = make_caches()
    tier = HostTier(caches, num_slots=4)
    tier.store(("a", 0), caches, 0)
    tier.store(("b", 0), caches, 1)
    tier.release_request("a")
    assert ("a", 0) not in tier and ("b", 0) in tier
