import sys
from pathlib import Path

# Make the `equity_research` package importable when running `pytest` from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
