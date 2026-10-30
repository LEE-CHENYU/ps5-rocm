#!/bin/bash
# Phase 5a: try Ubuntu 26.04 python3-torch-rocm 2.9.1 before building PyTorch from source. No GPU launch here.
set -euo pipefail
mkdir -p /srv/rocm-build/tmp; export TMPDIR=/srv/rocm-build/tmp LC_ALL=C.UTF-8
echo "== $(date) install"
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-torch-rocm 2>&1 | grep -E "newly installed|^E:|Setting up python3-torch" || true
echo "== arch list / version"
python3 -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.get_arch_list())"
echo "== gfx targets bundled in libtorch_hip"
L=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib
ls $L | head; for f in $(ls /usr/lib/x86_64-linux-gnu/libtorch_hip*.so* $L/libtorch_hip*.so 2>/dev/null | sort -u); do echo "-- $f"; strings -n 8 $f | grep -oE "amdgcn-amd-amdhsa--gfx[0-9a-z-]+" | sort | uniq -c; done
echo "== $(date) done"
