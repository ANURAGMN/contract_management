"""ADK entry point. `adk web` and `adk eval` import root_agent from here."""

import sys
from pathlib import Path

# Make the sibling `app` package importable when ADK loads this package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.adk_workflow import root_agent  # noqa: E402

__all__ = ["root_agent"]
