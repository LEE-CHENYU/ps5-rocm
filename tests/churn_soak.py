#!/usr/bin/env python3
"""Phase 6: allocation churn and soak through PyTorch on the gfx1013 GPU.

The BC-250 family's known failure is stale GPU address translations after memory is freed and the same virtual
addresses are reused: the GPU then reads and writes the wrong pages, silently. A GPU-side check would read back its own
wrong data consistently, so every result here is copied to the host and checked on the CPU, and long-lived "sentinel"
tensors are re-checked every cycle to catch stray writes into memory that belongs to someone else.

Each cycle allocates a tensor of random size (4 MB to 512 MB), fills it with a GPU kernel, verifies it on the CPU,
frees it and returns the memory to the driver (torch.cuda.empty_cache), so the next allocation reuses addresses.
Every 25 cycles a 1024^3 matmul is compared with a CPU reference.

Usage: churn_soak.py MINUTES    Prints "CHURN_SOAK_PASS cycles=N" or "CHURN_SOAK_FAIL ..." at the end."""
import random, sys, time
import torch

minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 2
dev = torch.device("cuda")
rng = random.Random(12345)
torch.manual_seed(0)


def pattern(n, salt, device):
    i = torch.arange(n, dtype=torch.int64, device=device)
    return ((i * 2654435761 + salt) & 0x7FFFFFFF).to(torch.int32)


sentinels = []
for k in range(4):  # long-lived tensors, 64 MB each, allocated before the churn starts
    s = pattern(1 << 24, 1000 + k, dev)
    sentinels.append((s, pattern(1 << 24, 1000 + k, "cpu")))
torch.manual_seed(7)
ma, mb = torch.randn(1024, 1024), torch.randn(1024, 1024)
mref = ma @ mb
ma_g, mb_g = ma.to(dev), mb.to(dev)

bad, cycles, t0, last = [], 0, time.time(), 0.0
while time.time() - t0 < minutes * 60:
    cycles += 1
    n = rng.choice([1 << 20, 3 << 20, 1 << 22, 6_000_000, 1 << 24, 1 << 25, 1 << 26, 1 << 27]) + rng.randrange(256)
    salt = rng.randrange(1 << 30)
    x = pattern(n, salt, dev)
    x.mul_(3).add_(1)  # a second kernel writing the same buffer
    got = x.cpu()
    want = pattern(n, salt, "cpu").mul_(3).add_(1)
    wrong = (got != want).sum().item()
    if wrong:
        first = (got != want).nonzero()[0].item()
        bad.append(f"cycle {cycles} n={n} wrong={wrong} first={first}")
        print(f"WRONG cycle {cycles} n={n} wrong={wrong} first={first} zeros={(got == 0).sum().item()}", flush=True)
    del x, got, want
    torch.cuda.empty_cache()
    for k, (s, ref) in enumerate(sentinels):
        w = (s.cpu() != ref).sum().item()
        if w:
            bad.append(f"cycle {cycles} sentinel {k} wrong={w}")
            print(f"WRONG sentinel {k} after cycle {cycles}: {w} words changed", flush=True)
    if cycles % 25 == 0:
        err = ((ma_g @ mb_g).cpu() - mref).abs().max().item()
        if not err < 1e-3:
            bad.append(f"cycle {cycles} matmul err {err:.3g}")
            print(f"WRONG matmul after cycle {cycles}: max abs err {err:.3g}", flush=True)
    if time.time() - last > 60:
        last = time.time()
        print(f"[{(last - t0) / 60:5.1f} min] cycles {cycles} bad {len(bad)} "
              f"peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

print(f"CHURN_SOAK_{'PASS' if not bad else 'FAIL'} cycles={cycles} minutes={minutes:g} bad={len(bad)}", flush=True)
for line in bad[:20]:
    print("  " + line)
sys.exit(1 if bad else 0)
