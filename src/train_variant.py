import sys, random, time
import torch
from step3_graphs import load_frames, make_sample
from step4_model import AeroMarineGNN, loss_fn, eval_loss

variant = sys.argv[1]                      # "sea" or "air"
assert variant in ("sea", "air")
use_ships, use_air = variant == "sea", variant == "air"

random.seed(0); torch.manual_seed(0)
df, frames = load_frames()
split_of = df.groupby("time")["split"].first()
samples = {"train": [], "val": [], "test": []}
for t in sorted(frames):
    s = make_sample(frames, t, use_ships=use_ships, use_air=use_air)
    if s:
        samples[split_of[t]].append(s)
print("Samples:", {k: len(v) for k, v in samples.items()})
torch.save(samples, f"data/processed/samples_{variant}.pt")

train, val = samples["train"], samples["val"]
model = AeroMarineGNN(in_dim=7)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best = 1e9
for epoch in range(30):
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
    print(f"epoch {epoch:2d} | train {total/len(train):.4f} | val {v:.4f} | {time.time()-t0:.0f}s")
    if v < best:
        best = v
        torch.save(model.state_dict(), f"models/model_{variant}.pt")
print("Best val loss:", round(best, 4), "-> models/model_" + variant + ".pt")