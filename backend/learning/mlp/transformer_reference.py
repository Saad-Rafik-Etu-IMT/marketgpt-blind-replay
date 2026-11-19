"""CPU reference encoder for comparing tokenized windows with the course MLP."""

import numpy as np

from src.ai.tokenizer import NumericReturnTokenizer


def fit_transformer(xt, yt, xv, yv, seed, variant, epochs, checkpoint):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    tokenizer = NumericReturnTokenizer().fit(xt)
    edges = np.asarray(tokenizer.bin_edges)

    def tokens(x):
        return torch.tensor(np.searchsorted(edges, x, side="right"), dtype=torch.long)

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

    model = Encoder()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0 if variant == "no-decay" else 0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    x, y, valid, labels = tokens(xt), torch.tensor(yt), tokens(xv), torch.tensor(yv)
    loss = nn.CrossEntropyLoss()
    best = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(x)).split(64):
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
            torch.save(
                dict(weights=model.state_dict(), edges=edges.tolist(), variant=variant),
                checkpoint,
            )
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["weights"])

    def predict(windows):
        with torch.no_grad():
            return model(tokens(windows)).softmax(-1)[:, 1].numpy()

    return (
        predict,
        history,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
