#!/bin/bash
# Phase 5b: rebuild the Ubuntu 26.04 pytorch-rocm 2.9.1 source package for gfx1013 only (CPU-only build, no GPU use).
# Only change to the Ubuntu packaging: PYTORCH_ROCM_ARCH = gfx1013 and a +gfx1013 version suffix.
# Built without LTO and debug info, 3 jobs: with the packaging defaults (LTO, -g, 4 jobs) the compile of
# RegisterCompositeExplicitAutograd_0.cpp was OOM-killed twice (systemd-oomd at 9.9 GB, then Slurm's 8 GB limit),
# and the LTO link of libtorch would need far more. debian/rules sets DEB_BUILD_MAINT_OPTIONS and
# REL_WITH_DEB_INFO itself, so the environment cannot override them: they are edited in place.
# GPU code is NOT compressed: Ubuntu's HIP runtime 7.1.0 fails "Failed to unbundle code object" on any
# --offload-compress bundle and then segfaults (a two-line vector add reproduces it), so PyTorch's
# --offload-compress (cmake/Dependencies.cmake, and torch.utils.cpp_extension for user extensions) is removed.
# -nc skips debian/rules clean, which deletes build/: a rerun after a flag change only recompiles what changed.
set -euo pipefail
mkdir -p /srv/rocm-build/tmp; export TMPDIR=/srv/rocm-build/tmp LC_ALL=C.UTF-8 LANG=C.UTF-8
cd /srv/rocm-build/torch/pytorch-rocm-2.9.1+dfsg
echo "== $(date) build-dep"
DEBIAN_FRONTEND=noninteractive apt-get build-dep -y pytorch-rocm 2>&1 | grep -E "newly installed|^E:" || true
sed -i "s|^export PYTORCH_ROCM_ARCH = .*|export PYTORCH_ROCM_ARCH = gfx1013|" debian/rules
grep -n "^export PYTORCH_ROCM_ARCH" debian/rules
sed -i -e "s|^export DEB_BUILD_MAINT_OPTIONS = hardening=+all\$|export DEB_BUILD_MAINT_OPTIONS = hardening=+all optimize=-lto|" \
  -e "s|^export REL_WITH_DEB_INFO = ON|export REL_WITH_DEB_INFO = OFF|" -e "s|-gsplit-dwarf ||" debian/rules
grep -nE "^export (DEB_BUILD_MAINT_OPTIONS|REL_WITH_DEB_INFO)|gsplit" debian/rules || true
sed -i "s|^\( *\)list(APPEND HIP_HIPCC_FLAGS --offload-compress)|\1# list(APPEND HIP_HIPCC_FLAGS --offload-compress)  # gfx1013: runtime cannot unbundle|" cmake/Dependencies.cmake
sed -i "/^ *'--offload-compress',\$/d" torch/utils/cpp_extension.py
grep -n "offload-compress" cmake/Dependencies.cmake torch/utils/cpp_extension.py || true
# A build tree configured with other flags (CMake caches them) would be reused by ninja: start clean.
grep -q "CMAKE_BUILD_TYPE:STRING=Release$" build/CMakeCache.txt 2>/dev/null || rm -rf build
head -1 debian/changelog | grep -q gfx1013 || sed -i "1s/(\([^)]*\))/(\1+gfx1013)/" debian/changelog
head -1 debian/changelog
echo "== $(date) dpkg-buildpackage"
DEB_CFLAGS_MAINT_APPEND=-g0 DEB_CXXFLAGS_MAINT_APPEND=-g0 \
  DEB_BUILD_OPTIONS="nocheck nodoc noautodbgsym parallel=3" MAX_JOBS=3 dpkg-buildpackage -b -nc -uc -us -j3
echo "== $(date) done"; ls -l ../*.deb
