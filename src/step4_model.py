import random, time
import torch, torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv
from step3_graphs import DXY, POS_SCALE

OVERFIT_TEST = False      # True = quick check on 100 samples. Set False for the real training.

NODE_TYPES = ["ship", "aircraft"]
EDGE_TYPES = [("ship", "near", "ship"), ("aircraft", "near", "aircraft"),
              ("aircraft", "near", "ship"), ("ship", "near", "aircraft")]

class AeroMarineGNN(nn.Module):
    def __init__(self, in_dim=7, hid=32, heads=2, n_layers=2):
        super().__init__()
        self.inp = nn.ModuleDict({n: nn.Linear(in_dim, hid) for n in NODE_TYPES})
        self.convs = nn.ModuleList([
            HeteroConv({et: GATv2Conv((hid, hid), hid // heads, heads=heads,
                                      edge_dim=1, add_self_loops=False)
                        for et in EDGE_TYPES}, aggr="sum")
            for _ in range(n_layers)])
        self.skip = nn.ModuleList([nn.ModuleDict({n: nn.Linear(hid, hid) for n in NODE_TYPES})
                                   for _ in range(n_layers)])
        self.gru = nn.ModuleDict({n: nn.GRU(hid, hid, batch_first=True) for n in NODE_TYPES})
        self.head = nn.ModuleDict({n: nn.Linear(hid, 4) for n in NODE_TYPES})
        for h in self.head.values():
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
    def spatial(self, g):
        x = {n: torch.relu(self.inp[n](g[n].x)) for n in NODE_TYPES}
        for conv, skip in zip(self.convs, self.skip):
            out = conv(x, g.edge_index_dict, edge_attr_dict=g.edge_attr_dict)
            x = {n: torch.relu(out.get(n, 0) + skip[n](x[n])) for n in NODE_TYPES}
        return x

    def forward(self, graphs):
        seq = [self.spatial(g) for g in graphs]            # one result per minute
        out = {}
        for n in NODE_TYPES:
            if seq[0][n].shape[0] == 0:
                out[n] = torch.zeros(0, 4)
                continue
            s = torch.stack([h[n] for h in seq], dim=1)    # [vehicles, 5 minutes, hid]
            _, last = self.gru[n](s)
            x = graphs[-1][n].x
            guess = torch.cat([x[:, 5:7], x[:, 3:5]], dim=1)   # "keep doing what you did last minute"
            out[n] = guess + self.head[n](last[-1])            # model learns a correction
        return out

def loss_fn(pred, y):
    return sum(((pred[n] - y[n]) ** 2).mean() for n in NODE_TYPES if len(y[n]))

@torch.no_grad()
def eval_loss(model, data):
    model.eval()
    return sum(loss_fn(model(g), y).item() for g, y, _, _ in data) / len(data)

@torch.no_grad()
def constant_velocity_loss(data):
    """Loss of the simplest guess: 'next move = last move'. The model should beat this."""
    tot = 0
    for g, y, _, _ in data:
        for n in NODE_TYPES:
            if len(y[n]) == 0:
                continue
            dxy = (g[-1][n].x[:, :2] - g[-2][n].x[:, :2]) * POS_SCALE / DXY[n]
            pred = torch.cat([dxy, g[-1][n].x[:, 3:5]], dim=1)
            tot += ((pred - y[n]) ** 2).mean().item()
    return tot / len(data)

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    samples = torch.load("data/processed/samples.pt", weights_only=False)
    train, val = samples["train"], samples["val"]
    if OVERFIT_TEST:
        train, val = train[:100], train[:100]
    print("Training samples:", len(train), "| Validation samples:", len(val))
    print("Constant-velocity loss on validation (to beat):", round(constant_velocity_loss(val), 4))

    model = AeroMarineGNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3 if OVERFIT_TEST else 1e-3)
    epochs = 40 if OVERFIT_TEST else 30
    save_path = "models/model_overfit.pt" if OVERFIT_TEST else "models/model.pt"
    best = 1e9
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        random.shuffle(train)
        total = 0
        for i, (g, y, _, _) in enumerate(train):
            loss = loss_fn(model(g), y)
            total += loss.item()
            (loss / 8).backward()
            if (i + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        v = eval_loss(model, val)
        print(f"epoch {epoch:2d} | train loss {total/len(train):.4f} | val loss {v:.4f} | {time.time()-t0:.0f}s")
        if v < best:
            best = v
            torch.save(model.state_dict(), save_path)
    print("Best val loss:", round(best, 4), "-> saved to", save_path)