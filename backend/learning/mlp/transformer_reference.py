"""CPU/GPU reference encoder for comparing tokenized windows with the course MLP."""

import sys

import numpy as np

from src.ai.tokenizer import NumericReturnTokenizer


def build_runs_on(supported, major, minor):
    """Whether a torch build with ``supported`` arches has kernels for major.minor.

    CUDA binaries (``sm_XY``) run on cards of the same major version and an
    equal or newer minor, so ``sm_50`` covers a 5.2 card; PTX (``compute_XY``)
    is JIT-compiled for any card at least that new.
    """
    for arch in supported:
        kind, _, version = arch.partition("_")
        if not version.isdigit():
            continue
        built = int(version[:-1]), int(version[-1])
        if kind == "sm" and built[0] == major and built[1] <= minor:
            return True
        if kind == "compute" and built <= (major, minor):
            return True
    return False


def cuda_problem(torch, device):
    """Return why ``device`` cannot run this model, or None when it can.

    Old cards (e.g. Tesla M60/M40, compute capability 5.2) are dropped by
    recent PyTorch wheels; such a build reports CUDA as available but has no
    kernels for the card. The check compares the card's capability with
    ``torch.cuda.get_arch_list()`` and then runs one tiny kernel.
    """
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if supported and not build_runs_on(supported, major, minor):
        return (
            f"{torch.cuda.get_device_name(device)} is {arch}, but this torch "
            f"{torch.__version__} build only ships {', '.join(supported)}. "
            "Install a torch build that still includes this architecture."
        )
    try:
        (torch.ones(8, device=device) * 2).sum().item()
    except RuntimeError as exc:
        return f"{torch.cuda.get_device_name(device)} failed a test kernel: {exc}"
    return None


def resolve_device(name="cpu"):
    """Return a torch.device.

    ``cpu`` and ``cuda[:N]`` are strict: an unusable GPU raises. ``auto`` picks
    the usable GPU with the most free memory and falls back to CPU (with the
    reason on stderr) when there is none, so the same command runs anywhere.
    """
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if name == "auto":
        if not torch.cuda.is_available():
            print(
                "device auto: no CUDA device visible, using CPU",
                file=sys.stderr,
                flush=True,
            )
            return torch.device("cpu")
        free, reasons = {}, []
        for index in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{index}")
            problem = cuda_problem(torch, device)
            if problem:
                reasons.append(f"cuda:{index}: {problem}")
            else:
                free[device] = torch.cuda.mem_get_info(device)[0]
        if not free:
            print(
                "device auto: no usable GPU, using CPU\n  " + "\n  ".join(reasons),
                file=sys.stderr,
                flush=True,
            )
            return torch.device("cpu")
        device = max(free, key=free.get)
        print(
            f"device auto: {device} ({torch.cuda.get_device_name(device)}, "
            f"{free[device] / 2**30:.1f} GiB free)",
            file=sys.stderr,
            flush=True,
        )
        return device
    if not name.startswith("cuda"):
        raise ValueError(f"Unknown device {name!r}; use cpu, auto or cuda[:N]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(name)
    problem = cuda_problem(torch, device)
    if problem:
        raise RuntimeError(problem)
    return device


def fit_transformer(xt, yt, xs, ys, seed, variant, epochs, checkpoint, device="cpu"):
    """Train on (xt, yt); select the checkpoint on (xs, ys), the inner-validation set."""
    import torch
    from torch import nn

    dev = resolve_device(device)
    torch.manual_seed(seed)
    if dev.type == "cpu":
        torch.set_num_threads(1)
    tokenizer = NumericReturnTokenizer().fit(xt)
    edges = np.asarray(tokenizer.bin_edges)

    def tokens(x):
        return torch.tensor(
            np.searchsorted(edges, x, side="right"), dtype=torch.long, device=dev
        )

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(tokenizer.vocab_size, 16)
            self.position = nn.Parameter(torch.randn(64, 16) * 0.02)
            layer = nn.TransformerEncoderLayer(
                16,
                2,
                32,
                batch_first=True,
                dropout=0 if variant == "no-dropout" else 0.1,
            )
            self.block = nn.TransformerEncoder(layer, 1)
            self.head = nn.Linear(16, 2)

        def forward(self, ids):
            x = self.embedding(ids)
            if variant != "no-position":
                x = x + self.position
            return self.head(self.block(x).mean(dim=1))

    model = Encoder().to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0 if variant == "no-decay" else 0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    x, valid = tokens(xt), tokens(xs)
    y = torch.tensor(np.asarray(yt), device=dev)
    labels = torch.tensor(np.asarray(ys), device=dev)
    loss = nn.CrossEntropyLoss()
    best = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(x), device=dev).split(64):
            optimizer.zero_grad()
            objective = loss(model(x[indices]), y[indices])
            objective.backward()
            optimizer.step()
        if variant != "fixed-lr":
            scheduler.step()
        model.eval()
        with torch.no_grad():
            training = sum(
                float(loss(model(a), b)) * len(a)
                for a, b in zip(x.split(128), y.split(128), strict=True)
            ) / len(x)
            validation = float(loss(model(valid), labels))
        history.append(dict(epoch=epoch + 1, train_loss=training, val_loss=validation))
        if validation < best:
            best = validation
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                dict(weights=state, edges=edges.tolist(), variant=variant), checkpoint
            )
    selected = min(range(len(history)), key=lambda i: history[i]["val_loss"])
    for i, entry in enumerate(history):
        entry["selected"] = i == selected
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["weights"])
    model.eval()

    def predict(windows):
        with torch.no_grad():
            return model(tokens(windows)).softmax(-1)[:, 1].cpu().numpy()

    return (
        predict,
        history,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


if __name__ == "__main__":
    # Probe: python learning/mlp/transformer_reference.py [auto|cpu|cuda[:N]]
    import torch

    print("torch", torch.__version__, "arch list", torch.cuda.get_arch_list())
    print("selected", resolve_device(sys.argv[1] if len(sys.argv) > 1 else "auto"))
