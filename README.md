# ROCm on the PlayStation 5 GPU (gfx1013)

Native ROCm/HIP compute on a PS5 running Linux: no `HSA_OVERRIDE_GFX_VERSION`, real gfx1013 code
objects, rocBLAS built for the chip, and results checked against the CPU (vector add exact, matrix
multiply within 2.15e-6).

The PS5's GPU (Oberon, PCI `1002:13FC`) is driven by Linux as `CHIP_CYAN_SKILLFISH`, the same
enum as the AMD BC-250 mining board, and reports `gfx_target_version 100103`: **gfx1013**, an
RDNA1-generation (GFX10.1) ISA with 36 compute units. ROCm does not support it, and the usual
shortcut of pretending to be an RX 6000-series card (`HSA_OVERRIDE_GFX_VERSION=10.3.0`) runs
RDNA2 code on RDNA1 hardware: on this console that hung the GPU on a one-line `x + 1`, the reset
failed and the whole machine needed a power cycle.

This repository adapts the BC-250 community's recipe ([akandr/bc250-rocm](https://github.com/akandr/bc250-rocm))
to PS5 Linux on **Ubuntu 26.04** and documents what it took.

## Status (2026-10-31)

| Step | State | Evidence |
|---|---|---|
| Patched `amdgpu` module (PS5 patch + BC-250 TLB-flush fixes) | ✅ boots, display and network fine | [docs/building-amdgpu.md](docs/building-amdgpu.md) |
| Boot parameters (`gpu_recovery=0`, flush params, 12 GiB GTT) | ✅ active, `mem_info_gtt_total` = 12288 MiB | [scripts/phase1-install-driver.sh](scripts/phase1-install-driver.sh) |
| ROCm 7.1 runtime (Ubuntu archive) | ✅ `rocminfo` lists **gfx1013**, 36 CU, 12 GiB pool | [logs/phase2-rocminfo.log](logs/phase2-rocminfo.log) |
| rocBLAS 7.1.0 built for gfx1013 | ✅ 56 gfx1013 kernel files, ~14 min on the console | [scripts/phase3.sh](scripts/phase3.sh), [logs/phase3-build-summary.log](logs/phase3-build-summary.log) |
| First GPU launches | ✅ vector add exact ×5; 1024³ SGEMM max error 2.15e-6 ×5; no faults | [tests/](tests/), [tests/phase4.log](tests/phase4.log) |
| Memory-churn / TLB-staleness probes, long soak | ✅ bc250-rocm `seq_probe` ×4 bad_gens=0; PyTorch churn 3 min (249 cycles) + **20-min soak (1,805 cycles)**, every result checked on the CPU, 0 bad, 0 kernel fault lines | [scripts/phase6.sh](scripts/phase6.sh), [tests/phase6.log](tests/phase6.log) |
| PyTorch 2.9.1 for gfx1013 | ✅ built uncompressed, installed; **30/30 op checks** (element-wise, reductions, sort/index, rocBLAS matmul, MIOpen conv, pooling, batch norm) match the CPU; 50-step CNN training matches the CPU to 2.1e-7 relative loss difference | [scripts/phase5b.sh](scripts/phase5b.sh), [tests/torch_ops_check.py](tests/torch_ops_check.py) |
| torchvision 0.24.1, transformers 5.22, diffusers 0.42, accelerate 1.15 | ✅ ResNet-18 train/infer, distilgpt2 logits + greedy generation identical to CPU + fine-tuning, tiny Stable Diffusion pipeline, Accelerator fp16 training | [tests/hf_stack_check.py](tests/hf_stack_check.py), [tests/hf-stack-check.log](tests/hf-stack-check.log) |
| What else works, what does not | 15/22 capability checks; see [below](#what-runs-and-what-does-not) | [tests/torch_capability_probe.py](tests/torch_capability_probe.py), [tests/capability-probe.log](tests/capability-probe.log) |
| SGLang | ❌ not attempted, see [below](#what-about-sglang) | |

**What the soak does and does not show.** The BC-250's characteristic failure is not a crash but *silently stale
GPU address translations* after memory is freed and reallocated. A 20-minute churn soak (allocations from 4 to 512 MB
freed back to the driver each cycle, every result and four long-lived sentinel buffers checked on the CPU) found no
corruption and no kernel fault. That is evidence, not proof: the BC-250 project saw rare faults only after hours of
mixed load, so run your own soak before trusting long jobs.

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
| First kernel segfaults in `libamdhip64` after "Failed to unbundle code object" | Ubuntu's HIP runtime 7.1.0 cannot read `--offload-compress` bundles (a two-line HIP program built with the flag reproduces it); PyTorch always compresses | build without it (`cmake/Dependencies.cmake`, see phase5b.sh) |
| Next segfault: "Couldn't create blit kernels", comgr "no targets are registered" | Ubuntu's `libtorch_cpu` (`USE_LLVM`) carries a private LLVM and exports `LLVMInitialize*`; loaded first, it captures comgr's AMDGPU target registration | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libLLVM.so.21.1` |
| Convolution: MIOpen "Failed to load kernel source: Conv_Winograd_v30_*.s" | Ubuntu's MIOpen (dfsg) ships without its GCN assembly kernels | `MIOPEN_DEBUG_GCN_ASM_KERNELS=0` |
| Copies above 16 KB never complete | stock SDMA microcode for this chip family (bc250-rocm) | `HSA_ENABLE_SDMA=0` |

## Installing the PyTorch build

```sh
cd /path/to/debs   # the four *+gfx1013 .deb files phase5b.sh produces
sudo apt-get install ./libtorch-rocm-2.9_*+gfx1013_amd64.deb ./libtorch-rocm-dev_*+gfx1013_amd64.deb \
  ./libtorch-rocm-test_*+gfx1013_amd64.deb ./python3-torch-rocm_*+gfx1013_amd64.deb
sudo apt-mark hold libtorch-rocm-2.9 libtorch-rocm-dev libtorch-rocm-test python3-torch-rocm
HSA_ENABLE_SDMA=0 MIOPEN_DEBUG_GCN_ASM_KERNELS=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libLLVM.so.21.1 \
  LD_LIBRARY_PATH=/opt/rocblas-gfx1013/lib:/path/to/comgr-fixed python3 -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.get_device_properties(0))"
```

The version suffix `+gfx1013` sorts above Ubuntu's, and the hold stops an update from swapping it back. HIP reports
18 multiprocessors: on RDNA it counts dual-CU work-group processors, so that is the expected 36 CUs.

## What runs and what does not

| Works on the GPU | Measured |
|---|---|
| fp32 / fp16 / bf16 matmul (rocBLAS) | 5.3 / 5.1 / 4.4 TFLOPS at 4096³ |
| convolution, pooling, BatchNorm (with MIOpen off, see below), mixed precision (`autocast` + `GradScaler`) | ResNet-18 training 84 img/s, inference ~200 img/s at batch 32 |
| attention (`scaled_dot_product_attention`, math backend), transformer training, LSTM | 4-layer d256 encoder 24 ms/step |
| FFT, random sampling, dropout, about 11 GiB of tensors in one process | |
| Hugging Face transformers / diffusers / accelerate, torchvision models and transforms | see the status table |

| Does not work | Why | Use instead |
|---|---|---|
| MIOpen BatchNorm training and RNNs | its kernels do not build for gfx1013 (inline asm, CK reductions) | `torch.backends.cudnn.enabled = False`: PyTorch's own kernels, same speed here |
| `torch.linalg` inverse / SVD / Cholesky, sparse matmul | Ubuntu's rocSOLVER and rocSPARSE have no gfx1013 code; this PyTorch has no MAGMA | do it on the CPU (or build rocSOLVER/rocSPARSE for gfx1013 like rocBLAS) |
| int8 matmul | hipBLASLt does not support gfx1013 | fp16 |
| `torch.compile` | TorchDynamo does not support Python 3.14, and Triton has no RDNA1 backend | eager mode |
| flash / memory-efficient attention | not built for gfx1013 | the math backend works, using more memory |
| torchvision's custom ops (`nms`, `roi_align`) on GPU tensors | the PyPI CPU wheel was used | move those tensors to the CPU, or build torchvision from source with HIP |

## Using it

```sh
python3 -m venv --system-site-packages /opt/ml-venv          # sees the system gfx1013 PyTorch
echo "torch==2.9.1+debian" > /opt/ml-venv/constraints.txt    # stops pip replacing it with a CUDA/CPU build
/opt/ml-venv/bin/pip install -c /opt/ml-venv/constraints.txt transformers diffusers accelerate safetensors
/opt/ml-venv/bin/pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torchvision==0.24.1
```

Run with `HSA_ENABLE_SDMA=0 MIOPEN_DEBUG_GCN_ASM_KERNELS=0 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libLLVM.so.21.1
LD_LIBRARY_PATH=/opt/rocblas-gfx1013/lib:/path/to/comgr-fixed` and set `torch.backends.cudnn.enabled = False` in the
script. Keep the Hugging Face cache on a filesystem with symlinks (not exFAT). The default PyPI torchvision wheel is
built for CUDA (`libcudart.so.12: cannot open shared object file`); use the CPU wheel above.

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
