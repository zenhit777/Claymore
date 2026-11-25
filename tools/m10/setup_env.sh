#!/usr/bin/env bash
# Build the CPU-only reference environment for the M10 generators.
# Requires Python 3.12 (the campaign ran 3.12.3) and network access once,
# to fetch torch 2.15.0+cpu and transformers @ 671aa77576266f41f33f9c28e2d824bfd1b1b63c.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$HERE/venv}"

python3.12 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip==24.0
"$VENV/bin/pip" install -r "$HERE/requirements.cpu.lock.txt"
"$VENV/bin/python" - <<'PY'
import numpy, torch, transformers
print("python ok:", "torch", torch.__version__, "| transformers", transformers.__version__, "| numpy", numpy.__version__)
assert torch.__version__ == "2.15.0+cpu", torch.__version__
assert not torch.cuda.is_available(), "the reference environment must be CPU-only"
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion  # noqa: F401
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig  # noqa: F401
print("diffusion_gemma + gemma4 importable: OK")
PY
echo "environment ready: $VENV"
