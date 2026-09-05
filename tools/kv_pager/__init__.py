"""Machinery for score-driven KV residency. See docs/kv-pager.md."""

from .guard import ResidencyGuard, Violation  # noqa: F401
from .hosttier import HostTier, HostTierFull  # noqa: F401
from .policy import POLICIES, Full, Recency, Stress  # noqa: F401
from .state import PagerState, current, reset  # noqa: F401
from .worker import WorkerPager  # noqa: F401
