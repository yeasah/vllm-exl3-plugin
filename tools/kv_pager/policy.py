"""Residency policies, called from both sides of the engine.

The scheduler decides what to free and the worker decides what the kernel may
read, and those two decisions have to be the same one. The cheapest way to
guarantee that is not two implementations reviewed against each other but a
single pure function that both import -- so the policy takes only what both
sides already know, and takes no engine objects at all.

The clock is `num_computed_tokens`, because it is the one quantity the
scheduler and the worker both have and both agree on. A wall-clock step counter
would drift the moment one side skipped a step.

Every policy returns row indices of *full* blocks, ascending. The partial tail
block is never a policy decision: it holds the key this step is about to write,
so it is always resident and always last, and the callers append it.
"""

from __future__ import annotations


class Recency:
    """Sinks plus the most recent blocks -- StreamingLLM's set.

    Shippable with no calibration, and the honest baseline for anything
    cleverer. Note what it cannot do: its window only ever slides forward, so
    it never asks for a block back and its fetch rate is zero after warmup.
    A pager running this policy exercises eviction and never exercises
    restore, which is why `Stress` exists.
    """

    name = "recency"

    def __init__(self, budget: int, sink: int = 2):
        self.budget = budget
        self.sink = sink

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        if n_full <= self.budget:
            return list(range(n_full))
        sink = min(self.sink, self.budget)
        keep = list(range(sink))
        keep += list(range(n_full - (self.budget - sink), n_full))
        return sorted(set(keep))


class Stress:
    """Deliberately churns the resident set, to exercise the restore path.

    Not a policy anyone would ship: it exists because `Recency` never fetches,
    so a pager tested only under recency would leave its entire restore path --
    the whole difference between paging and eviction -- unexercised. This keeps
    the sinks and the newest blocks, then fills the remaining budget with a
    window that walks backwards through the middle of the context, so blocks
    leave and are asked for again at a rate the caller sets.

    `churn` is how many blocks the walking window advances per decode step,
    which makes the fetch rate a dial: it is the knob that turns "policy error
    costs latency" from an assertion into a measurement.
    """

    name = "stress"

    def __init__(self, budget: int, sink: int = 2, recent: int = 4, churn: int = 1):
        self.budget = budget
        self.sink = sink
        self.recent = recent
        self.churn = churn

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        if n_full <= self.budget:
            return list(range(n_full))
        sink = min(self.sink, self.budget)
        recent = min(self.recent, max(0, self.budget - sink))
        keep = set(range(sink))
        keep |= set(range(n_full - recent, n_full))
        roam = self.budget - len(keep)
        if roam > 0:
            span = max(1, n_full - sink - recent)
            start = (num_computed // max(1, self.churn)) % span
            for i in range(roam):
                keep.add(sink + (start + i) % span)
        return sorted(x for x in keep if 0 <= x < n_full)


class Oracle:
    """Recency, plus a set of blocks the caller says matter. The ceiling.

    Not a policy that could exist in production -- it is told the answer. It
    exists to separate two questions that a bad quality number cannot
    distinguish on its own: whether the *machinery* can deliver good output at
    a low budget, and whether a *policy* can find the right blocks. If the
    oracle retrieves at a budget where recency does not, the mechanism is fine
    and the remaining gap is policy, which is the phase this project is trying
    to reach. If even the oracle fails, nothing about scoring will help.

    `must_keep` is a class attribute because a policy is constructed from a
    frozen spec with nowhere to thread a set through. That is fine for an
    instrument and would not be for anything else.
    """

    name = "oracle"
    must_keep: set = set()

    def __init__(self, budget: int, sink: int = 2):
        self.budget = budget
        self.sink = sink

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        if n_full <= self.budget:
            return list(range(n_full))
        sink = min(self.sink, self.budget)
        keep = set(range(sink))
        keep |= {i for i in self.must_keep if 0 <= i < n_full}
        keep = set(sorted(keep)[: self.budget])
        recent = self.budget - len(keep)
        keep |= set(range(n_full - recent, n_full))
        return sorted(i for i in keep if 0 <= i < n_full)


class OracleLate(Oracle):
    """An oracle that only wants the block back after `after` tokens.

    The point is the round trip, not the policy. Plain `Oracle` keeps the block
    it was told about from the beginning, so it never restores anything and a
    good answer from it proves only that residency works. This one lets the
    block be evicted during prefill -- its contents going out to the host tier
    -- and asks for it back at decode time, so answering correctly means the
    bytes made the journey and still mean what they meant. That is the
    end-to-end version of the bit-exactness the host tier proves in isolation,
    and the only arrangement in which a needle can test it: the answer has to
    be generated after the restore.
    """

    name = "oracle_late"
    after: int = 1 << 60

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        if num_computed < self.after:
            keep = set(range(min(self.sink, self.budget)))
            recent = self.budget - len(keep)
            keep |= set(range(n_full - recent, n_full))
            return sorted(i for i in keep if 0 <= i < n_full)
        return super().resident(n_full, num_computed)


class Full:
    """Everything resident. The control arm, and it must be bit-exact.

    A pager configured with this has to reproduce an unpaged run exactly; any
    deviation is a mechanism bug rather than a policy cost, which is what keeps
    a bug from being read as an accuracy result.
    """

    name = "full"

    def __init__(self, budget: int = 0, sink: int = 0):
        pass

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        return list(range(n_full))


POLICIES = {p.name: p for p in (Recency, Stress, Oracle, OracleLate, Full)}
