"""What the scheduler side decided, in a form the worker side can act on.

The manager and the worker have to agree every step, and the thing that makes
that fragile is that they run in different places: the manager owns allocation,
the worker owns the KV tensors and the view the kernel reads. A worker that
reached into the manager would work in-process and nowhere else, so it does not
-- it reads this, and the manager writes it. Swapping this for a field on
`SchedulerOutput` is then the only change a multiprocess deployment needs, and
nothing above it moves.

What is published per request per step is small and blunt on purpose:

    row        the manager's authoritative logical -> physical mapping, nulls
               included. The worker cannot derive this, because restored blocks
               reach it appended at the end of its own row while the manager
               placed them at their index, so the two disagree by construction.
    resident   the logical indices the kernel may read, tail last
    restored   (index, block id) placed this step -- copy these *in* before the
               forward
    evicting   (index, block id) chosen this step and freed the next -- copy
               these *out* while they are still allocated

Publishing the whole row each step is the obvious thing rather than the cheap
thing: it is O(context) where a delta would be O(change). That is deliberate
for now -- the delta version has to reconstruct the mapping on the worker side
and be checked against this one before it can be trusted, and an unchecked
optimisation here produces exactly the silent wrong-block read this project
built a guard for.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestStep:
    row: list[int] = field(default_factory=list)
    resident: list[int] = field(default_factory=list)
    restored: list[tuple[int, int]] = field(default_factory=list)
    evicting: list[tuple[int, int]] = field(default_factory=list)
    num_computed: int = 0


class PagerState:
    """Per-request decisions, replaced each time the manager runs."""

    def __init__(self):
        self.steps: dict[str, RequestStep] = {}
        self.block_size = 0
        self.publishes = 0

    def publish(self, req_id: str, step: RequestStep) -> None:
        self.steps[req_id] = step
        self.publishes += 1

    def get(self, req_id: str) -> RequestStep | None:
        return self.steps.get(req_id)

    def drop(self, req_id: str) -> None:
        self.steps.pop(req_id, None)


#: One pager per process. A module-level handle because the manager is
#: constructed by vLLM from a spec -- there is nowhere to thread a reference
#: through -- and because a second pager in one engine would be a bug rather
#: than a configuration.
_STATE: PagerState | None = None


def current() -> PagerState:
    global _STATE
    if _STATE is None:
        _STATE = PagerState()
    return _STATE


def reset() -> PagerState:
    global _STATE
    _STATE = PagerState()
    return _STATE
