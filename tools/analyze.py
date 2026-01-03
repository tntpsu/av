#!/usr/bin/env python3
"""
Simple wrapper for comprehensive analysis tool.

This is the PRIMARY analysis tool - use this for all recording analysis.

Usage:
    python tools/analyze.py [recording_file]
    python tools/analyze.py --latest
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and run comprehensive analyzer
from tools.analyze_recording_comprehensive import main

if __name__ == '__main__':
    main()


