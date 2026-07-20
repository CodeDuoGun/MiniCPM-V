#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from app.runtime import validate_torch_stack  # noqa: E402


if __name__ == "__main__":
    print(validate_torch_stack())
