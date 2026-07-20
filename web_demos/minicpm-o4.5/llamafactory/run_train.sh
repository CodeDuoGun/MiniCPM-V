#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

required = {
    "numpy": "1.26.4",
    "fsspec": "2025.3.0",
    "datasets": "4.0.0",
    "transformers": "4.51.0",
    "tokenizers": "0.21.4",
    "trl": "0.18.2",
}
required_torch = {
    "torch": ("2.8.0", "cu126"),
    "torchvision": ("0.23.0", "cu126"),
    "torchaudio": ("2.8.0", "cu126"),
}
mismatches = []
for package, expected in required.items():
    installed = version(package)
    if installed != expected:
        mismatches.append(f"{package}=={installed} (required {expected})")

for package, (expected_base, expected_cuda) in required_torch.items():
    installed = Version(version(package))
    local = installed.local or ""
    if installed.base_version != expected_base or expected_cuda not in local:
        mismatches.append(
            f"{package}=={installed} (required {expected_base}+{expected_cuda})"
        )

if mismatches:
    raise SystemExit(
        "MiniCPM-o 4.5 training dependency check failed: "
        + ", ".join(mismatches)
        + ". Run: pip install --upgrade --force-reinstall --no-cache-dir "
          "-r web_demos/minicpm-o4.5/llamafactory/requirements-compat.txt"
    )
print("MiniCPM-o 4.5 training dependency check passed")
PY

llamafactory-cli train "$SCRIPT_DIR/minicpmo45_lora_sft.yaml"
