#!/bin/bash
# Phase 1: install the patched amdgpu module and boot parameters on PS5 Linux (kernel 7.2.3).
# Run as root. Takes effect at the next boot. Every change keeps a backup; see "Rollback" below.
#
#   ARTIFACTS  folder holding amdgpu-7.2.3-ps5-flush.ko (built from ps5-amd-part.patch + ps5-flush-on-top.patch)
#   BOOTDIR    the loader's boot folder that holds cmdline.txt (on the PS5 Linux USB drive: PS5/Linux/)
#
# The ps5-linux loader's initrd does not contain amdgpu, so the module loads from the root filesystem
# and no initramfs rebuild is needed. Check that on your own initrd first:  lsinitramfs initrd.img | grep amdgpu
set -euo pipefail
ARTIFACTS=${ARTIFACTS:?set ARTIFACTS}; BOOTDIR=${BOOTDIR:?set BOOTDIR}
TS=$(date +%Y%m%dT%H%M%S)
D=/lib/modules/$(uname -r)/kernel/drivers/gpu/drm/amd/amdgpu
[ "$(uname -r)" = 7.2.3 ] || { echo "built for 7.2.3, running $(uname -r)"; exit 1; }
[ "$(modinfo -F vermagic $ARTIFACTS/amdgpu-7.2.3-ps5-flush.ko)" = "$(modinfo -F vermagic $D/amdgpu.ko)" ] || { echo vermagic differs; exit 1; }
cp -p $D/amdgpu.ko $D/amdgpu.ko.stock-$(uname -r)
cp -p $BOOTDIR/cmdline.txt $BOOTDIR/cmdline.txt.before-rocm-$TS
install -m 644 $ARTIFACTS/amdgpu-7.2.3-ps5-flush.ko $D/amdgpu.ko
depmod -a "$(uname -r)"
modinfo -F parm "$(modinfo -n amdgpu)" | grep -q bc250_flush_by_runlist
# gpu_recovery=0: a failed MODE1 reset wedges the host anyway; disabling it avoids the half-reset state.
# bc250_flush_pasid_kiq=0, bc250_flush_by_runlist=3: the TLB-flush fixes (both default to stock behaviour).
# ttm.pages_limit=3145728: let the GPU use up to 12 GiB of system RAM (GTT); the VRAM carve-out is only 512 MB.
sed -i '1s/$/ amdgpu.gpu_recovery=0 amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=3 ttm.pages_limit=3145728/' $BOOTDIR/cmdline.txt
sync; cat $BOOTDIR/cmdline.txt
# Rollback: copy cmdline.txt.before-rocm-* back (the new parameters default to stock), and if the module itself
# misbehaves: cp $D/amdgpu.ko.stock-$(uname -r) $D/amdgpu.ko && depmod -a. The boot folder is also writable
# from the PS5 OS side, so a Linux that does not come up can be fixed without booting it.
