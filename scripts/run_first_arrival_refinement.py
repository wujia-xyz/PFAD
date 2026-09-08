#!/usr/bin/env python3
"""Compatibility entry point for the published PFAD commands."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pfad.core import *
if __name__ == "__main__":
    main()
