"""CPU/GPU reference encoder for comparing tokenized windows with the course MLP."""

import numpy as np

from src.ai.tokenizer import NumericReturnTokenizer


def resolve_device(name="cpu"):
    """Return a torch.device, failing fast when CUDA cannot actually run.

    Old cards (e.g. Tesla M60/M40, compute capability 5.2) are dropped by
    recent PyTorch wheels; such a build reports CUDA as available but has no
    kernels for the card. The check compares the card's ``sm_XY`` with
    ``torch.cuda.get_arch_list()``.
    """
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if not name.startswith("cuda"):
        raise ValueError(f"Unknown device {name!r}; use cpu or cuda[:N]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(name)
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if supported and arch not in supported and f"compute_{major}{minor}" not in supported:
        raise RuntimeError(
            f"{torch.cuda.get_device_name(device)} is {arch}, but this torch "
            f"{torch.__version__} build only ships {', '.join(supported)}. "
            "Install a torch build that still includes this architecture."
        )
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
