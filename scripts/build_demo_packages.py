"""Write offline sample ReviewPackages to data/demo_packages/.

Usage:
    python scripts/build_demo_packages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.demo_packages import list_samples, load_sample, write_all_samples  # noqa: E402


def main() -> int:
    paths = write_all_samples()
    for p in paths:
        pkg = load_sample(
            next(s["id"] for s in list_samples() if s["file"] == p.name)
        )
        print(
            f"wrote {p.relative_to(ROOT)}  "
            f"route={pkg.route.route} findings={len(pkg.findings)} "
            f"redlines={len(pkg.redlines)} sample={pkg.is_sample}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
