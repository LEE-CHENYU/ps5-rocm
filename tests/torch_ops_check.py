#!/usr/bin/env python3
"""Phase 5c: run common PyTorch operations and a 50-step training loop on the gfx1013 GPU and compare every
result with the same computation on the CPU, in fp32. Prints one line per check and "TORCH_OPS_PASS" or
"TORCH_OPS_FAIL n" at the end. Checks run from simplest to most complex, so a hang names the first risky step."""
import sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = torch.device("cuda")
torch.manual_seed(0)
failures = []


def report(name, gpu, cpu, rtol, atol):
    gpu = gpu.detach().float().cpu()
    cpu = cpu.detach().float()
    ok = gpu.shape == cpu.shape and torch.allclose(gpu, cpu, rtol=rtol, atol=atol)
    err = (gpu - cpu).abs().max().item() if gpu.shape == cpu.shape else float("nan")
    print(f"{'ok  ' if ok else 'FAIL'} {name:<28} max abs err {err:.3g}", flush=True)
    if not ok:
        failures.append(name)


def check(name, fn, *inputs, rtol=1e-5, atol=1e-5):
    out_cpu = fn(*inputs)
    out_gpu = fn(*[x.to(dev) if torch.is_tensor(x) else x for x in inputs])
    torch.cuda.synchronize()
    if isinstance(out_cpu, tuple):
        for i, (g, c) in enumerate(zip(out_gpu, out_cpu)):
            report(f"{name}[{i}]", g, c, rtol, atol)
    else:
        report(name, out_gpu, out_cpu, rtol, atol)


print(f"torch {torch.__version__} hip {torch.version.hip} arch {torch.cuda.get_arch_list()} "
      f"device {torch.cuda.get_device_properties(0).gcnArchName}", flush=True)

a = torch.randn(1 << 20)
b = torch.randn(1 << 20)
pos = torch.rand(1 << 20) + 0.1
m = torch.randn(512, 768)
idx = torch.randint(0, 512, (4096,))

# elementwise and reductions
check("add", torch.add, a, b, rtol=0, atol=0)
check("mul", torch.mul, a, b, rtol=0, atol=0)
check("exp", torch.exp, a)
check("log", torch.log, pos)
check("tanh", torch.tanh, a)
check("sigmoid", torch.sigmoid, a)
check("relu", F.relu, a, rtol=0, atol=0)
check("gelu", F.gelu, a)
check("sum", lambda x: x.sum(), a, atol=1e-2)
check("mean+std", lambda x: (x.mean(), x.std()), a, atol=1e-4)
check("max/argmax", lambda x: (x.max(dim=1).values, x.max(dim=1).indices), m, rtol=0, atol=0)
check("cumsum", lambda x: x.cumsum(1), m, atol=1e-3)
check("softmax", lambda x: F.softmax(x, 1), m)
check("log_softmax", lambda x: F.log_softmax(x, 1), m, atol=1e-4)
check("layer_norm", lambda x: F.layer_norm(x, (768,)), m, atol=1e-4)
# indexing and sorting
check("index_select", lambda x, i: x.index_select(0, i), m, idx, rtol=0, atol=0)
check("gather", lambda x: x.gather(1, torch.arange(768, device=x.device).flip(0).expand(512, 768)), m, rtol=0, atol=0)
check("scatter_add", lambda i: torch.zeros(512, device=i.device).scatter_add(0, i, torch.ones(4096, device=i.device)),
      idx, rtol=0, atol=0)
check("sort", lambda x: x.sort(1).values, m, rtol=0, atol=0)
check("topk", lambda x: x.topk(8, 1).values, m, rtol=0, atol=0)
check("embedding", lambda i, w: F.embedding(i, w), idx, m, rtol=0, atol=0)
# matrix multiply (rocBLAS)
for n in (64, 257, 1024):
    x, y = torch.randn(n, n), torch.randn(n, n)
    check(f"matmul {n}", torch.matmul, x, y, rtol=1e-4, atol=1e-3)
check("bmm 16x128x64x96", torch.bmm, torch.randn(16, 128, 64), torch.randn(16, 64, 96), rtol=1e-4, atol=1e-4)
check("linear", F.linear, torch.randn(256, 768), torch.randn(512, 768), torch.randn(512), rtol=1e-4, atol=1e-3)
# convolution and pooling (MIOpen)
img = torch.randn(8, 3, 64, 64)
check("conv2d 3x3", lambda x, w: F.conv2d(x, w, padding=1), img, torch.randn(16, 3, 3, 3), rtol=1e-4, atol=1e-4)
check("max_pool2d", lambda x: F.max_pool2d(x, 2), img, rtol=0, atol=0)
check("batch_norm", lambda x: F.batch_norm(x, None, None, training=True), img, atol=1e-4)
check("cross_entropy", lambda x, t: F.cross_entropy(x, t), m[:, :10], idx[:512] % 10, atol=1e-5)


# 50-step training loop: same initial weights and batches on both devices. Momentum SGD amplifies fp32 rounding
# differences over the steps, so the tolerance is 1e-2 relative; a stale-translation fault gives garbage, not 1e-3.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


torch.manual_seed(1)
data = torch.randn(50, 32, 1, 28, 28)
labels = torch.randint(0, 10, (50, 32))
torch.manual_seed(2)
cpu_net = Net()
gpu_net = Net().to(dev)
gpu_net.load_state_dict(cpu_net.state_dict())
opt_c = torch.optim.SGD(cpu_net.parameters(), lr=0.05, momentum=0.9)
opt_g = torch.optim.SGD(gpu_net.parameters(), lr=0.05, momentum=0.9)
loss_c, loss_g = [], []
t0 = time.time()
for step in range(50):
    for net, opt, losses, d in ((cpu_net, opt_c, loss_c, "cpu"), (gpu_net, opt_g, loss_g, dev)):
        opt.zero_grad()
        loss = F.cross_entropy(net(data[step].to(d)), labels[step].to(d))
        loss.backward()
        opt.step()
        losses.append(loss.item())
torch.cuda.synchronize()
rel = max(abs(g - c) / max(abs(c), 1e-6) for g, c in zip(loss_g, loss_c))
print(f"train: loss cpu {loss_c[0]:.4f}->{loss_c[-1]:.4f} gpu {loss_g[0]:.4f}->{loss_g[-1]:.4f} "
      f"max rel loss diff {rel:.3g} ({time.time() - t0:.1f} s)", flush=True)
if rel > 1e-2 or not loss_g[-1] < loss_g[0]:
    failures.append("train loss")
    print("FAIL train loss", flush=True)
for (n, pc), pg in zip(cpu_net.named_parameters(), gpu_net.parameters()):
    report(f"train param {n}", pg, pc, rtol=1e-2, atol=1e-3)

print("TORCH_OPS_PASS" if not failures else f"TORCH_OPS_FAIL {len(failures)}: {', '.join(failures)}", flush=True)
sys.exit(1 if failures else 0)
