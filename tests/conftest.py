"""The modules under test sit in the repository root, one level up. Put that
directory on sys.path here so `pytest` works from any working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
