import pandas as pd, torch
from step3_graphs import load_frames, make_sample
from step5_eval import load_model, score_samples
from inject import evaluate

log = pd.read_csv("data/processed/anomaly_log.csv", parse_dates=["ws", "we"])
events_all = log.to_dict("records")
_, frames = load_frames("data/processed/unified_test_corrupted.csv")
times = sorted(frames)

def build(use_ships, use_air):
    out = []
    for t in times:
        s = make_sample(frames, t, use_ships=use_ships, use_air=use_air)
        if s:
            out.append(s)
    return out

# name, model file, which vehicles, clean samples for the threshold, events to score
configs = [
    ("Constant velocity", None,                 dict(use_ships=True,  use_air=True),  "data/processed/samples.pt",     None),
    ("Sea-only GNN",      "models/model_sea.pt", dict(use_ships=True,  use_air=False), "data/processed/samples_sea.pt", "ship"),
    ("Air-only GNN",      "models/model_air.pt", dict(use_ships=False, use_air=True),  "data/processed/samples_air.pt", "aircraft"),
    ("Unified GNN",       "models/model.pt",     dict(use_ships=True,  use_air=True),  "data/processed/samples.pt",     None),
]

metrics, detected = {}, {}
for name, path, flags, val_path, only in configs:
    model = load_model(path) if path else None
    val = torch.load(val_path, weights_only=False)["val"]
    thr = score_samples(val, model).groupby("type")["err"].quantile(0.99)
    ev = [e for e in events_all if only is None or e["vtype"] == only]
    rows, det = evaluate(score_samples(build(**flags), model), thr, ev)
    metrics[name], detected[name] = rows, det

print("\n=== Metrics per vehicle type (auc = threshold-free score, 0.5 = guessing, 1 = perfect) ===")
print(pd.concat(metrics).to_string())
print("\n=== Share of injected events detected ===")
print(pd.concat(detected, axis=1).to_string())