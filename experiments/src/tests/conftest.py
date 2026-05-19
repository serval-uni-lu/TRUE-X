# tests/conftest.py
import sys
from pathlib import Path

# Ensure `src/` is on the path so `from datasets.xxx import ...` works
# regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
