#!/usr/bin/env python3
"""What kinds of ML work does PyTorch support on gfx1013? Each check runs in isolation, compares with the CPU where
that makes sense, and reports ok / FAIL / unsupported plus timings. Prints one line per check, then a summary."""
import os, subprocess, sys, time
# Each check runs in its own process (a crash in one library must not hide the rest): run with no argument to drive
# all checks, or with a check name to run just that one.
ONLY = sys.argv[1] if len(sys.argv) > 1 else None
if ONLY is None and "--list" not in sys.argv:
    names = subprocess.run([sys.executable, __file__, "--list"], capture_output=True, text=True).stdout.splitlines()
    bad = []
    for n in names:
        p = subprocess.run([sys.executable, __file__, n], capture_output=True, text=True, timeout=600)
        line = [l for l in p.stdout.splitlines() if l[:4] in ("ok  ", "FAIL")]
        print(line[-1] if line else f"CRASH {n:<37} exit {p.returncode} ({'segfault' if p.returncode == -11 else 'error'})", flush=True)
        if not line or line[-1].startswith("FAIL"): bad.append(n)
    print(f"SUMMARY ok {len(names) - len(bad)}/{len(names)}; not working: {', '.join(bad) if bad else 'none'}")
    sys.exit(0)
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "cuda"
results = []


def check(name, fn):
    if ONLY == "--list":
        print(name); return
    if name != ONLY:
        return
    t0 = time.time()
    try:
        info = fn()
        torch.cuda.synchronize()
        status = "ok"
    except Exception as e:  # noqa: BLE001 - every failure is a finding here
        info = (str(e).splitlines() or [type(e).__name__])[0][:150]
        status = "FAIL"
    results.append((status, name))
    print(f"{status:<4} {name:<38} {time.time() - t0:6.2f}s  {info}", flush=True)
    torch.cuda.empty_cache()


def close(g, c, tol):
    err = (g.float().cpu() - c.float()).abs().max().item()
    if not err <= tol:
        raise RuntimeError(f"max abs err {err:.3g} > {tol}")
    return f"max abs err {err:.2g}"


def tflops(dtype, n=4096, reps=10):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    return f"{2 * n**3 * reps / (time.time() - t) / 1e12:.2f} TFLOPS"


torch.manual_seed(0)
x, y = torch.randn(512, 512), torch.randn(512, 512)
if ONLY != "--list": print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).gcnArchName} | "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB", flush=True)

# precision
check("matmul fp32 throughput", lambda: tflops(torch.float32))
check("matmul fp16 vs CPU fp32", lambda: close(x.half().to(dev) @ y.half().to(dev), x @ y, 0.5))
check("matmul fp16 throughput", lambda: tflops(torch.float16))
check("matmul bf16 vs CPU fp32", lambda: close(x.bfloat16().to(dev) @ y.bfloat16().to(dev), x @ y, 2.0))
check("matmul bf16 throughput", lambda: tflops(torch.bfloat16))
ci, cw = torch.randn(4, 16, 64, 64), torch.randn(32, 16, 3, 3)
check("conv2d fp16 vs CPU fp32", lambda: close(F.conv2d(ci.half().to(dev), cw.half().to(dev)), F.conv2d(ci, cw), 0.5))


def amp_step():
    m = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 10)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda")
    first = None
    for _ in range(30):
        xb, yb = torch.randn(64, 256, device=dev), torch.randint(0, 10, (64,), device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(m(xb), yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        first = first or loss.item()
    return f"30 AdamW+GradScaler fp16 steps, loss {first:.3f}->{loss.item():.3f}"


check("autocast fp16 + GradScaler training", amp_step)

# attention
q = torch.randn(2, 8, 256, 64)
ref = F.scaled_dot_product_attention(q, q, q, is_causal=True)
check("SDPA fp32 causal vs CPU", lambda: close(F.scaled_dot_product_attention(q.to(dev), q.to(dev), q.to(dev), is_causal=True), ref, 1e-4))
check("SDPA fp16 causal vs CPU", lambda: close(F.scaled_dot_product_attention(*(q.half().to(dev),) * 3, is_causal=True), ref, 1e-2))


def sdpa_backends():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    out = []
    for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(b):
                F.scaled_dot_product_attention(*(q.half().to(dev),) * 3)
            out.append(b.name + ":yes")
        except Exception:
            out.append(b.name + ":no")
    return " ".join(out)


check("SDPA backends available (fp16)", sdpa_backends)


def transformer_train():
    layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True), 4).to(dev)
    opt = torch.optim.AdamW(layer.parameters(), 1e-4)
    xb = torch.randn(16, 128, 256, device=dev)
    for _ in range(5):
        loss = layer(xb).pow(2).mean(); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        loss = layer(xb).pow(2).mean(); loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    return f"4-layer d256 encoder, batch 16x128: {(time.time() - t) / 10 * 1000:.0f} ms/step"


check("transformer encoder train (fp32)", transformer_train)


def lstm():
    m = nn.LSTM(128, 256, 2, batch_first=True).to(dev)
    o, _ = m(torch.randn(8, 64, 128, device=dev)); o.sum().backward()
    return "fwd+bwd"


check("LSTM fwd+bwd (MIOpen RNN)", lstm)


def lstm_native():
    torch.backends.cudnn.enabled = False  # route RNNs to PyTorch's own kernels instead of MIOpen
    return lstm() + " with torch.backends.cudnn.enabled=False"


check("LSTM fwd+bwd (native, MIOpen off)", lstm_native)

# other libraries PyTorch dispatches to
a = torch.randn(256, 256)
spd = a @ a.T + 256 * torch.eye(256)
check("linalg.inv (rocSOLVER/MAGMA)", lambda: close(torch.linalg.inv(spd.to(dev)), torch.linalg.inv(spd), 1e-4))
check("linalg.svd", lambda: close(torch.linalg.svdvals(a.to(dev)), torch.linalg.svdvals(a), 1e-3))
check("linalg.cholesky", lambda: close(torch.linalg.cholesky(spd.to(dev)), torch.linalg.cholesky(spd), 1e-3))
check("fft.rfft2 (rocFFT)", lambda: close(torch.fft.rfft2(a.to(dev)).abs(), torch.fft.rfft2(a).abs(), 1e-2))
check("sparse COO matmul (hipSPARSE)", lambda: close(torch.sparse.mm(a.to_sparse().to(dev), a.to(dev)), a @ a, 1e-2))
check("randn / dropout / multinomial", lambda: f"{torch.multinomial(F.dropout(torch.rand(1000, device=dev).softmax(0), 0.1), 10).shape}")
check("int8 matmul (torch._int_mm)", lambda: str(torch._int_mm(torch.randint(-8, 8, (64, 64), device=dev, dtype=torch.int8),
                                                               torch.randint(-8, 8, (64, 64), device=dev, dtype=torch.int8)).dtype))


def compiled():
    f = torch.compile(lambda t: torch.sin(t) * 2 + 1)
    return close(f(x.to(dev)), torch.sin(x) * 2 + 1, 1e-5)


check("torch.compile (Inductor, needs Triton)", compiled)


def big_alloc():
    gib = []
    try:
        for _ in range(11):
            gib.append(torch.empty(2**28, device=dev))  # 1 GiB each
    finally:
        n = len(gib); del gib
    return f"{n} x 1 GiB tensors allocated"


check("allocate 11 GiB in 1 GiB tensors", big_alloc)

