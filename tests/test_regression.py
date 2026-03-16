"""
Regression tests to verify fixes don't make things worse.

NOTE: Tests that compared latest vs previous recordings have been removed.
Cross-recording comparison is meaningless when different tracks produce
different statistical properties. Use test_comfort_gate_replay.py (golden
recording approach) for regression testing instead.
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
