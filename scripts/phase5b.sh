#!/bin/bash
# Phase 5b: rebuild the Ubuntu 26.04 pytorch-rocm 2.9.1 source package for gfx1013 only (CPU-only build, no GPU use).
# Only change to the Ubuntu packaging: PYTORCH_ROCM_ARCH = gfx1013 and a +gfx1013 version suffix.
set -euo pipefail
mkdir -p /srv/rocm-build/tmp; export TMPDIR=/srv/rocm-build/tmp LC_ALL=C.UTF-8 LANG=C.UTF-8
cd /srv/rocm-build/torch/pytorch-rocm-2.9.1+dfsg
echo "== $(date) build-dep"
DEBIAN_FRONTEND=noninteractive apt-get build-dep -y pytorch-rocm 2>&1 | grep -E "newly installed|^E:" || true
sed -i "s|^export PYTORCH_ROCM_ARCH = .*|export PYTORCH_ROCM_ARCH = gfx1013|" debian/rules
grep -n "^export PYTORCH_ROCM_ARCH" debian/rules
head -1 debian/changelog | grep -q gfx1013 || sed -i "1s/(\([^)]*\))/(\1+gfx1013)/" debian/changelog
head -1 debian/changelog
echo "== $(date) dpkg-buildpackage"
DEB_BUILD_OPTIONS="nocheck nodoc parallel=4" MAX_JOBS=4 dpkg-buildpackage -b -uc -us -j4
echo "== $(date) done"; ls -l ../*.deb
