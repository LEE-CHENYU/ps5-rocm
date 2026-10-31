#!/bin/bash
# Phase 7: rocSPARSE and rocSOLVER 7.1.0 for gfx1013 (CPU-only build, no GPU use), so torch.linalg and sparse ops stop
# crashing: Ubuntu's librocsolver0 / librocsparse1 ship no gfx101x code. Same toolchain and environment as phase3.sh.
# Offload compression is OFF: Ubuntu's HIP runtime 7.1.0 cannot unbundle compressed code objects (see README).
# Installed to their own prefixes, used with LD_LIBRARY_PATH like rocBLAS; the system libraries stay untouched.
set -euo pipefail
mkdir -p /srv/rocm-build/tmp; export TMPDIR=/srv/rocm-build/tmp TMP=/srv/rocm-build/tmp TEMP=/srv/rocm-build/tmp LC_ALL=C.UTF-8 LANG=C.UTF-8
W=/srv/rocm-build/rb710; B=/usr/lib/llvm-21/bin
echo "== $(date) phase 7 start"
cd $W
git -C rocm-libraries sparse-checkout add projects/rocsolver projects/rocsparse
git -C rocm-libraries log -1 --oneline
export PATH=/srv/rocm-build/shim:$B:/usr/bin:/usr/sbin  # no ROCM_PATH: on Ubuntu it hides the device libs from clang
COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$B/clang -DCMAKE_CXX_COMPILER=$B/clang++
        -DHIP_PLATFORM=amd -DGPU_TARGETS=gfx1013 -DAMDGPU_TARGETS=gfx1013 -DROCM_SYMLINK_LIBS=OFF
        -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF -DBUILD_CLIENTS_SAMPLES=OFF)

echo "== $(date) rocSPARSE"
cmake -S rocm-libraries/projects/rocsparse -B build-rocsparse "${COMMON[@]}" \
  -DCMAKE_INSTALL_PREFIX=/opt/rocsparse-gfx1013 -DCMAKE_PREFIX_PATH=/opt/rocblas-gfx1013 \
  -DBUILD_WITH_OFFLOAD_COMPRESS=OFF -DBUILD_WITH_ROCTX=OFF
cmake --build build-rocsparse -j 4
cmake --install build-rocsparse

echo "== $(date) rocSOLVER"
cmake -S rocm-libraries/projects/rocsolver -B build-rocsolver "${COMMON[@]}" \
  -DCMAKE_INSTALL_PREFIX=/opt/rocsolver-gfx1013 -DCMAKE_PREFIX_PATH="/opt/rocblas-gfx1013;/opt/rocsparse-gfx1013" \
  -DBUILD_OFFLOAD_COMPRESS=OFF -DBUILD_WITH_SPARSE=ON
cmake --build build-rocsolver -j 4
cmake --install build-rocsolver

for f in /opt/rocsparse-gfx1013/lib*/librocsparse.so.* /opt/rocsolver-gfx1013/lib*/librocsolver.so.*; do
  [ -L "$f" ] && continue
  echo "== $f: $(python3 -c "import sys;b=open(sys.argv[1],'rb').read();print('gfx1013 bundles',b.count(b'gfx1013'),'compressed',b.count(b'CCOB'))" "$f")"
done
ls -l /opt/rocsparse-gfx1013/lib*/librocsparse.so* /opt/rocsolver-gfx1013/lib*/librocsolver.so*
echo "== $(date) phase 7 done"
