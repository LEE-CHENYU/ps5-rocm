#!/usr/bin/env python3
"""torchvision, transformers, diffusers and accelerate on the gfx1013 GPU, each compared with the CPU where the result
is deterministic. Needs the venv from the README (system PyTorch + PyPI libraries) and network access to the Hugging
Face Hub for two small models. Each check runs in its own process; prints ok / FAIL / CRASH and a summary."""
import os, subprocess, sys, time

CHECKS = ["torchvision", "transformers", "diffusers", "accelerate"]
if len(sys.argv) == 1:
    bad = []
    for c in CHECKS:
        p = subprocess.run([sys.executable, __file__, c], capture_output=True, text=True, timeout=1800)
        lines = [l for l in p.stdout.splitlines() if l[:4] in ("ok  ", "FAIL")]
        print("\n".join(lines) if lines else f"CRASH {c} exit {p.returncode}: {p.stderr.strip().splitlines()[-1:]}", flush=True)
        if not lines or any(l.startswith("FAIL") for l in lines): bad.append(c)
    print(f"SUMMARY ok {len(CHECKS) - len(bad)}/{len(CHECKS)}; not working: {', '.join(bad) or 'none'}")
    sys.exit(0)

import torch
# MIOpen's BatchNorm-training and RNN kernels do not build for gfx1013; PyTorch's own kernels do, at the same speed.
torch.backends.cudnn.enabled = False
dev = "cuda"
which = sys.argv[1]


def report(name, fn):
    t0 = time.time()
    try:
        info = fn(); torch.cuda.synchronize(); s = "ok  "
    except Exception as e:  # noqa: BLE001
        info, s = (str(e).splitlines() or [type(e).__name__])[0][:160], "FAIL"
    print(f"{s} {which + ': ' + name:<44} {time.time() - t0:6.1f}s  {info}", flush=True)


def maxerr(a, b):
    return (a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item()


if which == "torchvision":
    import torchvision
    from torchvision.transforms import v2

    def resnet():
        torch.manual_seed(0)
        m = torchvision.models.resnet18(num_classes=10).eval()
        x = torch.randn(8, 3, 224, 224)
        ref = m(x)
        g = m.to(dev)(x.to(dev))
        err = maxerr(g, ref)
        assert err < 1e-3, f"max abs err {err:.3g}"
        m.train(); loss = m(x.to(dev)).sum(); loss.backward()
        return f"resnet18 fwd matches CPU (max abs err {err:.2g}), fwd+bwd ok"

    def transforms():
        img = torch.randint(0, 256, (4, 3, 300, 400), dtype=torch.uint8)
        t = v2.Compose([v2.Resize(224, antialias=True), v2.CenterCrop(224), v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        err = maxerr(t(img.to(dev)), t(img))
        # GPU and CPU antialiased resize round differently by up to one 8-bit level: 1/255/0.229 = 0.017 after Normalize
        assert err < 0.02, f"max abs err {err:.3g}"
        return f"resize/crop/normalize on GPU tensors match CPU (max abs err {err:.2g})"

    report("ResNet-18", resnet)
    report("transforms v2", transforms)

elif which == "transformers":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", attn_implementation="sdpa").eval()
    ids = tok("The PlayStation 5 has a graphics processor that", return_tensors="pt")

    def logits():
        with torch.no_grad():
            ref = model(**ids).logits
            g = model.to(dev)(**{k: v.to(dev) for k, v in ids.items()}).logits
        err = maxerr(g, ref)
        assert err < 1e-2, f"max abs err {err:.3g}"
        return f"distilgpt2 logits match CPU (max abs err {err:.2g})"

    def generate():
        out = model.to(dev).generate(**{k: v.to(dev) for k, v in ids.items()}, max_new_tokens=20, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        cpu = model.to("cpu").generate(**ids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
        same = torch.equal(out.cpu(), cpu)
        assert same, "greedy GPU and CPU continuations differ"
        return "greedy 20 tokens identical to CPU: " + repr(tok.decode(out[0][ids["input_ids"].shape[1]:]))[:60]

    def train_step():
        m = model.to(dev).train(); opt = torch.optim.AdamW(m.parameters(), 5e-5)
        batch = {k: v.to(dev) for k, v in ids.items()}
        l0 = m(**batch, labels=batch["input_ids"]).loss
        for _ in range(5):
            loss = m(**batch, labels=batch["input_ids"]).loss; loss.backward(); opt.step(); opt.zero_grad()
        assert loss.item() < l0.item()
        return f"5 AdamW steps, loss {l0.item():.3f} -> {loss.item():.3f}"

    report("logits vs CPU", logits)
    report("greedy generate", generate)
    report("fine-tune steps", train_step)

elif which == "diffusers":
    from diffusers import StableDiffusionPipeline

    def tiny_sd():
        pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch",
                                                       safety_checker=None, requires_safety_checker=False)
        kw = dict(prompt="a console on a desk", num_inference_steps=10, output_type="pt", height=64, width=64)
        ref = pipe(**kw, generator=torch.Generator().manual_seed(0)).images
        g = pipe.to(dev)(**kw, generator=torch.Generator().manual_seed(0)).images
        err = maxerr(g, ref)
        assert err < 5e-2, f"max abs err {err:.3g}"
        return f"tiny SD 10 steps, image {tuple(g.shape)} matches CPU (max abs err {err:.2g})"

    report("tiny Stable Diffusion pipeline", tiny_sd)

elif which == "accelerate":
    from accelerate import Accelerator

    def loop():
        acc = Accelerator(mixed_precision="fp16")
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
        opt = torch.optim.AdamW(m.parameters(), 1e-2)
        xs = torch.randn(512, 64); ys = xs[:, :1] * 3 + 1
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=64)
        m, opt, dl = acc.prepare(m, opt, dl)
        first = None
        for _ in range(20):
            for xb, yb in dl:
                loss = torch.nn.functional.mse_loss(m(xb), yb); acc.backward(loss); opt.step(); opt.zero_grad()
                first = first or loss.item()
        assert str(acc.device).startswith("cuda") and loss.item() < first / 10
        return f"device {acc.device}, fp16, loss {first:.2f} -> {loss.item():.3f}"

    report("Accelerator fp16 training", loop)
