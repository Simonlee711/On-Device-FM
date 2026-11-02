import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
