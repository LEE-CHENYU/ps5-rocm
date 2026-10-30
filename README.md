# ROCm on the PlayStation 5 GPU (gfx1013)

Native ROCm/HIP compute on a PS5 running Linux: no `HSA_OVERRIDE_GFX_VERSION`, real gfx1013 code
objects, rocBLAS built for the chip, and results checked bit-for-bit against the CPU.

The PS5's GPU (Oberon, PCI `1002:13FC`) is driven by Linux as `CHIP_CYAN_SKILLFISH`, the same
enum as the AMD BC-250 mining board, and reports `gfx_target_version 100103`: **gfx1013**, an
RDNA1-generation (GFX10.1) ISA with 36 compute units. ROCm does not support it, and the usual
shortcut of pretending to be an RX 6000-series card (`HSA_OVERRIDE_GFX_VERSION=10.3.0`) runs
RDNA2 code on RDNA1 hardware: on this console that hung the GPU on a one-line `x + 1`, the reset
failed and the whole machine needed a power cycle.

This repository adapts the BC-250 community's recipe ([akandr/bc250-rocm](https://github.com/akandr/bc250-rocm))
to PS5 Linux on **Ubuntu 26.04** and documents what it took.

## Status (2026-10-30)

| Step | State | Evidence |
|---|---|---|
| Patched `amdgpu` module (PS5 patch + BC-250 TLB-flush fixes) | ✅ boots, display and network fine | [docs/building-amdgpu.md](docs/building-amdgpu.md) |
| Boot parameters (`gpu_recovery=0`, flush params, 12 GiB GTT) | ✅ active, `mem_info_gtt_total` = 12288 MiB | [scripts/phase1-install-driver.sh](scripts/phase1-install-driver.sh) |
| ROCm 7.1 runtime (Ubuntu archive) | ✅ `rocminfo` lists **gfx1013**, 36 CU, 12 GiB pool | [logs/phase2-rocminfo.log](logs/phase2-rocminfo.log) |
| rocBLAS 7.1.0 built for gfx1013 | ✅ 56 gfx1013 kernel files, ~14 min on the console | [scripts/phase3.sh](scripts/phase3.sh), [logs/phase3-build-summary.log](logs/phase3-build-summary.log) |
| First GPU launches | ✅ vector add exact ×5; 1024³ SGEMM max error 2.15e-6 ×5; no faults | [tests/](tests/), [tests/phase4.log](tests/phase4.log) |
| Memory-churn / TLB-staleness probes, long soak | ⏳ not run yet | |
| PyTorch 2.9.1 for gfx1013 | ✅ built (3 h 27 min on the console) and installed; `torch.cuda.get_arch_list()` = `['gfx1013']`, device visible, 12 GiB. ⏳ no GPU ops run yet | [scripts/phase5b.sh](scripts/phase5b.sh) |
| SGLang | ❌ not attempted, see [below](#what-about-sglang) | |

**Read the caveat before relying on it.** The BC-250's characteristic failure is not a crash but
*silently stale GPU address translations* after memory is freed and reallocated. The small tests
here do not churn memory, so they cannot rule that out yet. Until the churn probes and a soak pass,
treat results as unverified for anything that matters.

## What it takes

1. **Driver.** The running PS5 kernel's `amdgpu` is vanilla 7.2.3 plus the PS5 patch. Rebuild the
   module with that patch *and* the BC-250 TLB-flush fixes, extending the runlist-flush device check from
   `0x13FE` (BC-250) to `0x13FC` (PS5). Installing a vanilla module would lose the PS5's GPU ID and its
   1080p/HDMI handling. See [docs/building-amdgpu.md](docs/building-amdgpu.md).
2. **Boot parameters.** `amdgpu.gpu_recovery=0 amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=3
   ttm.pages_limit=3145728`. The VRAM carve-out is 512 MB; everything else is system RAM through GTT,
   raised here from 7 to 12 GiB.
3. **ROCm runtime.** Ubuntu 26.04 packages ROCm 7.1 (`rocminfo`, `hipcc`, `libamdhip64`, `librocblas5`).
   It enumerates gfx1013 natively once the driver is right. Its rocBLAS ships no gfx101x kernels.
4. **rocBLAS for gfx1013.** Upstream `rocm-libraries` at tag `rocm-7.1.0`, patched with the BC-250's
   `apply_gfx1013_rocblas711.py` (gfx1013 added to Tensile and rocBLAS), built with Ubuntu's LLVM 21 and
   installed to its own prefix so the system rocBLAS stays untouched.
5. **comgr register count.** ROCm 7.0–7.1.1's comgr says gfx10 has 256 VGPRs instead of 1024, which makes
   HIP's occupancy API report zero blocks for large kernels. The BC-250's `fix_comgr_gfx10_vgprs.py`
   matched all 13 gfx10 rows in Ubuntu's build too. It is applied to a *copy* that is loaded with
   `LD_LIBRARY_PATH`.

## Ubuntu-specific problems (the BC-250 recipe targets Fedora)

| Symptom | Cause | Fix |
|---|---|---|
| `apt` fails inside Slurm jobs: `Symlinking ... failed` | the job's `TMPDIR` was on an exFAT drive | point `TMPDIR` at an ext4 path |
| `Unsupported target gfx1013 by compiler` | exporting `ROCM_PATH=/usr` makes clang look for device libraries in the wrong place | don't set `ROCM_PATH`; Ubuntu's clang finds them itself |
| Tensile: `amdclang either not found` | Tensile hard-codes AMD's compiler names | symlink `amdclang`/`amdclang++` to `/usr/lib/llvm-21/bin/clang(++)` |
| Ubuntu's `python3-torch-rocm` reports an empty arch list on gfx1013 | built for gfx908/90a/942/1030/1151/1200/1201 only | rebuild the source package with `PYTORCH_ROCM_ARCH = gfx1013` |
| PyTorch build OOM-killed at step ~590/2555 (`RegisterCompositeExplicitAutograd_0.cpp`) | the package builds with LTO and debug info; `debian/rules` sets `DEB_BUILD_MAINT_OPTIONS` and `REL_WITH_DEB_INFO` itself, so environment overrides are ignored | edit both in `debian/rules` (`optimize=-lto`, `REL_WITH_DEB_INFO = OFF`); peak then fits in 9 GB with 3 jobs |
| `torch.cuda` init fails: `dlopen: libcaffe2_nvrtc.so: cannot open shared object file` | the unversioned `libcaffe2_nvrtc.so` link ships only in `libtorch-rocm-dev` (also true of Ubuntu's own build) | install `libtorch-rocm-dev` too |

## Installing the PyTorch build

```sh
cd /path/to/debs   # the four *+gfx1013 .deb files phase5b.sh produces
sudo apt-get install ./libtorch-rocm-2.9_*+gfx1013_amd64.deb ./libtorch-rocm-dev_*+gfx1013_amd64.deb \
  ./libtorch-rocm-test_*+gfx1013_amd64.deb ./python3-torch-rocm_*+gfx1013_amd64.deb
sudo apt-mark hold libtorch-rocm-2.9 libtorch-rocm-dev libtorch-rocm-test python3-torch-rocm
LD_LIBRARY_PATH=/opt/rocblas-gfx1013/lib:/path/to/comgr-fixed \
  python3 -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.get_device_properties(0))"
```

The version suffix `+gfx1013` sorts above Ubuntu's, and the hold stops an update from swapping it back. HIP reports
18 multiprocessors: on RDNA it counts dual-CU work-group processors, so that is the expected 36 CUs.

## Running the tests

```sh
cd tests
clang++ -x hip --offload-arch=gfx1013 -O2 vecadd.hip -o vecadd -lamdhip64
clang++ -x hip --offload-arch=gfx1013 -O2 sgemm.cpp -o sgemm \
  -I/opt/rocblas-gfx1013/include -L/opt/rocblas-gfx1013/lib -lrocblas -lamdhip64 -Wl,-rpath,/opt/rocblas-gfx1013/lib
LD_LIBRARY_PATH=/path/to/comgr-fixed ./vecadd && LD_LIBRARY_PATH=/path/to/comgr-fixed ./sgemm
```

Stop every other GPU user first (llama.cpp, stable-diffusion servers, whisper), capture the kernel log
off-box (netconsole), and be ready to power-cycle: with `gpu_recovery=0`, a GPU fault means a reboot.

## What about SGLang?

Not realistic on this GPU today. SGLang's ROCm path targets AMD's Instinct parts (gfx942/gfx950).
Community builds reach RDNA3 (gfx1100) only with patched kernels and features disabled, and nothing
shows its Triton-based kernels running on RDNA1. For LLM serving on this console, llama.cpp (the
BC-250 project runs it on HIP, and its Vulkan backend already works here) is the practical choice.

## Credits and licences

- The approach, the TLB-flush analysis and patches, the rocBLAS gfx1013 patch and the comgr fix come from
  Artur Andrzejczak's [bc250-rocm](https://github.com/akandr/bc250-rocm), which also credits
  [GabriWar/bc250-rocm-working](https://github.com/GabriWar/bc250-rocm-working) for the runlist flush.
- The PS5 kernel patch comes from the PS5 Linux project's kernel tree.
- Scripts and documentation here: **AGPL-3.0-or-later** (several scripts adapt bc250-rocm's, which use that
  licence). Kernel patches in [patches/](patches/): **GPL-2.0**, like the kernel they modify.

Nothing here is affiliated with Sony or AMD. Running Linux on a PS5 requires an exploited console; this
repository contains no exploit code.
