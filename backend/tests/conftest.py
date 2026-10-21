import os
import sys
from pathlib import Path

os.environ.setdefault("SLM_ENABLED", "false")
os.environ.setdefault("WAQI_TOKEN", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
