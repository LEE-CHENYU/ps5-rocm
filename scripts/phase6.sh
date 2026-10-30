#!/bin/bash
# Phase 5c + 6: first PyTorch work on the gfx1013 GPU, then allocation churn and a soak. Stops at the first failure.
#   5c  tests/torch_ops_check.py       common ops + a 50-step training loop, each compared with the CPU in fp32
#   6a  bc250-rocm patches/seq_probe.c  alloc/dispatch/free generations, the BC-250 project's churn reproducer (x3)
#   6b  tests/churn_soak.py 3           3 minutes of PyTorch allocation churn, verified on the CPU
#   6c  tests/churn_soak.py SOAK_MIN    the soak (default 20 minutes), then seq_probe once more
# After every stage the kernel log is checked for the fault lines the BC-250 project watches for.
# HSA_ENABLE_SDMA=0: with the stock SDMA microcode no copy above 16 KB completes on this chip family (bc250-rocm),
# so copies go through shader kernels instead. Run with nothing else on the GPU; with gpu_recovery=0 a fault = reboot.
# Usage: phase6.sh TESTS_DIR BC250_DIR [SOAK_MIN]
set -uo pipefail
T=$1 BC=$2 SOAK=${3:-20}
W=/srv/rocm-build/phase6; mkdir -p $W; cd $W
export HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=/opt/rocblas-gfx1013/lib:/srv/rocm-build/comgr-fixed
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin|ring .* timeout"
faults() { journalctl -k -b 0 --no-pager 2>/dev/null | grep -ciE "$PAT"; }
F0=$(faults)
stage() {  # name timeout command...
  local name=$1 to=$2; shift 2
  echo "== $(date +%T) $name"
  timeout -k 30 "$to" "$@"; local rc=$?
  local df=$(( $(faults) - F0 ))
  echo "== $(date +%T) $name rc=$rc kernel_fault_lines=+$df"
  if [ $rc -ne 0 ] || [ $df -ne 0 ]; then
    journalctl -k -b 0 --no-pager | grep -iE "$PAT" | tail -20
    echo "PHASE6_STOP at $name"; exit 1
  fi
}
echo "== $(date) boot $(cat /proc/sys/kernel/random/boot_id) recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery)" \
     "runlist=$(cat /sys/module/amdgpu/parameters/bc250_flush_by_runlist) kiq=$(cat /sys/module/amdgpu/parameters/bc250_flush_pasid_kiq)" \
     "fault_lines_at_start=$F0"
clang++ -x hip --offload-arch=gfx1013 -O2 "$BC/patches/seq_probe.c" -o seq_probe -lamdhip64 || exit 1
SEQ="4000 8388608 4194304 8388608 6000000 8388608"   # the arguments bc250-rocm's churn_aba_f44.sh uses
stage 5c-torch-ops 900 python3 "$T/torch_ops_check.py"
for i in 1 2 3; do stage "6a-seq_probe-$i" 300 ./seq_probe $SEQ; done
stage 6b-churn-3min 600 python3 "$T/churn_soak.py" 3
stage "6c-soak-${SOAK}min" $(( SOAK * 60 + 600 )) python3 "$T/churn_soak.py" "$SOAK"
stage 6c-seq_probe-after 300 ./seq_probe $SEQ
echo "PHASE6_PASS $(date +%T)"
