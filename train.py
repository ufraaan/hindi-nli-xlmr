#!/usr/bin/env python3
"""Forward to src/train.py so `python train.py` still works from the repo root."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "src" / "train.py"), run_name="__main__")
