"""Root conftest: ensures the repo root is importable as `ml` / `apps` packages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
