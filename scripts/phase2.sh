#!/bin/bash
# ROCm Phase 2: userspace from the Ubuntu 26.04 archive (ROCm 7.1), then rocminfo (enumeration only, no GPU kernel launch).
set -euo pipefail
echo "== $(date) phase 2 start"
dpkg -l > /mnt/elba/ps5-rocm/dpkg-before-phase2.txt
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rocminfo rocm-smi hipcc librocblas-dev libamdhip64-dev
echo "== installed versions"; dpkg -l | grep -E "rocm|rocblas|hip|hsa|comgr" | awk "{print \$2, \$3}"
echo "== rocminfo"; rocminfo 2>&1 | grep -E "Name:|Marketing|gfx|Compute Unit|Agent|ROCk|HSA Runtime" | head -40 || true
echo "== rocblas kernel libraries for gfx101x"; ls /usr/lib/x86_64-linux-gnu/rocblas/*/library 2>/dev/null | grep -oE "gfx[0-9a-f]+" | sort | uniq -c
echo "== comgr"; ls -l /usr/lib/x86_64-linux-gnu/libamd_comgr.so* 2>/dev/null
echo "== $(date) phase 2 done"
