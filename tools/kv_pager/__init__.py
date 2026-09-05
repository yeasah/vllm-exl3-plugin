"""Machinery for score-driven KV residency. See docs/kv-pager.md."""

from .guard import ResidencyGuard, Violation  # noqa: F401
from .hosttier import HostTier, HostTierFull  # noqa: F401
from .policy import POLICIES, Full, Recency, Stress  # noqa: F401
