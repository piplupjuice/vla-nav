"""Fetch a Kaggle kernel's SMALL artifacts (logs, JSON, JSONL, images).

Never downloads the 15 GB checkpoint path: the file-pattern whitelist is
explicit and passed to the Kaggle CLI, so only evidence files land locally.

Usage:
  python scripts/fetch_kernel.py pikanigga/vlanav-02-thor-pilot \
      kaggle_notebooks/02_thor_pilot/output

Environment: KAGGLE_API_TOKEN (never written to disk by this script).
"""
from __future__ import annotations

import os
import subprocess
import sys

PATTERNS = "|".join([
    r".*\.log$",
    r".*\.json$",
    r".*\.jsonl$",
    r".*\.md$",
    r".*\.txt$",
    r".*\.jpg$",
    r".*\.png$",
    r".*\.mp4$",
])


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    slug, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", "--file-pattern", PATTERNS,
           "-q", "-p", out, slug]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, shell=False)
    if r.returncode != 0:
        print("fetch failed (kernel may still be running)")
        return r.returncode
    total = 0
    for root, _dirs, files in os.walk(out):
        for f in files:
            p = os.path.join(root, f)
            total += os.path.getsize(p)
    print(f"fetched into {out} | total {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
