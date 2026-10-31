#!/bin/bash
# ROCm Phase 3: rocBLAS for gfx1013 on Ubuntu 26.04 (system ROCm 7.1 + LLVM 21). CPU-only build, no GPU use.
# Adapted from bc250-rocm/scripts/build_rocblas711_gfx1013.sh (Fedora paths) to Ubuntu paths and tag rocm-7.1.0.
set -euo pipefail
mkdir -p /srv/rocm-build/tmp; export TMPDIR=/srv/rocm-build/tmp TMP=/srv/rocm-build/tmp TEMP=/srv/rocm-build/tmp LC_ALL=C.UTF-8 LANG=C.UTF-8
W=/srv/rocm-build/rb710; B=/usr/lib/llvm-21/bin; PFX=/opt/rocblas-gfx1013
echo "== $(date) phase 3 start"
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ninja-build rocm-cmake python3-msgpack python3-joblib libmsgpack-cxx-dev
mkdir -p $W; cd $W
if [ ! -d rocm-libraries ]; then
  git clone --depth 1 --branch rocm-7.1.0 --filter=blob:none --sparse https://github.com/ROCm/rocm-libraries.git
  git -C rocm-libraries sparse-checkout set projects/rocblas shared/tensile
fi
git -C rocm-libraries log -1 --oneline
python3 /mnt/elba/ps5-rocm/bc250-rocm/scripts/apply_gfx1013_rocblas711.py $W/rocm-libraries
export PATH=/srv/rocm-build/shim:$B:/usr/bin:/usr/sbin  # no ROCM_PATH: on Ubuntu it hides the device libs from clang
export TENSILE_ROCM_ASSEMBLER_PATH=$B/clang++ TENSILE_ROCM_OFFLOAD_BUNDLER_PATH=$B/clang-offload-bundler
cmake -S rocm-libraries/projects/rocblas -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$B/clang -DCMAKE_CXX_COMPILER=$B/clang++ \
  -DCMAKE_INSTALL_PREFIX=$PFX -DHIP_PLATFORM=amd -DGPU_TARGETS=gfx1013 -DAMDGPU_TARGETS=gfx1013 \
  -DBUILD_WITH_TENSILE=ON -DBUILD_WITH_HIPBLASLT=OFF \
  -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF -DBUILD_FORTRAN_CLIENTS=OFF \
  -DTensile_LIBRARY_FORMAT=msgpack -DTensile_CPU_THREADS=4 -DROCM_SYMLINK_LIBS=OFF \
  -DBUILD_OFFLOAD_COMPRESS=OFF  # Ubuntu's HIP 7.1.0 cannot unbundle compressed code: dot/gemv etc. segfault
cmake --build build -j 4
cmake --install build
python3 -c "import sys;b=open(sys.argv[1],'rb').read();print('== compressed bundles (must be 0):',b.count(b'CCOB'))" $PFX/lib/librocblas.so.5.1
echo "== gfx1013 kernel files: $(ls $PFX/lib*/rocblas/library 2>/dev/null | grep -c gfx1013)"
ls -l $PFX/lib*/librocblas.so* 
echo "== $(date) phase 3 done"
