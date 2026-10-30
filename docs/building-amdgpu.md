# Building the patched amdgpu module

Target: the PS5 Linux kernel **7.2.3**. The module must match the running kernel's `vermagic` and
symbol CRCs; this build did (same version tag, all 1,095 CRCs, 0 missing).

## Inputs

- Vanilla Linux 7.2.3 source.
- [`patches/ps5-amd-part.patch`](../patches/ps5-amd-part.patch): the AMD-driver part of the PS5 kernel
  patch (23 files, 63 hunks; the full PS5 patch also touches USB, Bluetooth and platform code). It adds
  the PS5 GPU IDs (for example `0x13DA`) and `amdgpu.force_1080p`. `include/linux/ps5.h` comes from the
  installed PS5 kernel headers.
- [`patches/ps5-flush-on-top.patch`](../patches/ps5-flush-on-top.patch): the BC-250 runlist and SVM
  flush changes, with the runlist flush enabled for `0x13FC` (PS5) as well as `0x13FE` (BC-250).
- bc250-rocm's `scripts/apply_flush_pasid_param.py`, which adds the `gmc_v10_0.c` change (the
  `bc250_flush_pasid_kiq` parameter). It is not part of `ps5-flush-on-top.patch`.

## Steps

```sh
cd linux-7.2.3
patch -p1 < ps5-amd-part.patch
python3 bc250-rocm/scripts/apply_flush_pasid_param.py .
patch -p1 < ps5-flush-on-top.patch
cp /lib/modules/7.2.3/build/.config . && cp /lib/modules/7.2.3/build/Module.symvers .
make olddefconfig && make modules_prepare
make M=drivers/gpu/drm/amd/amdgpu modules
modinfo -F vermagic drivers/gpu/drm/amd/amdgpu/amdgpu.ko     # must equal the running module's
modinfo -F parm drivers/gpu/drm/amd/amdgpu/amdgpu.ko | grep bc250
```

Build a **stock-equivalent** module first (PS5 patch only) and compare its version tag, parameters,
GPU ID table, firmware list and symbol CRCs with the running module. That proves the build path
before any behaviour changes. A compiler-version warning (for example gcc 15 against a kernel built
with 14) did not affect the symbol checks.

## Notes for this kernel

- Modules are uncompressed and the kernel uses zstd, not xz.
- On Ubuntu the initramfs is rebuilt with `update-initramfs`, but the ps5-linux loader's initrd does
  not include amdgpu, so no initramfs change was needed.
- Both new parameters default to stock behaviour. Nothing changes until they are set on the boot
  command line, which makes rollback a one-line edit.
