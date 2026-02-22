"""Compatibility adapter for recording summary analysis.

Canonical implementation lives in tools.drive_summary_core.
"""

from tools.drive_summary_core import (
    LOW_VISIBILITY_STALE_REASONS,
    analyze_recording_summary,
    safe_float,
)

__all__ = [
    "LOW_VISIBILITY_STALE_REASONS",
    "analyze_recording_summary",
    "safe_float",
]
