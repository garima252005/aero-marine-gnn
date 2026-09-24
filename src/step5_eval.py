import pandas as pd, torch
from step4_model import AeroMarineGNN, NODE_TYPES

def load_model(path="models/model.pt"):
    m = AeroMarineGNN(in_dim=7)
    m.load_state_dict(torch.load(path))
    m.eval()
    return m

@torch.no_grad()
def score_samples(samples, model=None):
    """Prediction error per vehicle per minute. model=None means the constant-velocity guess."""
    rows = []
    for g, y, ids, t in samples:
        if model is None:
            x = {n: g[-1][n].x for n in NODE_TYPES}
            pred = {n: torch.cat([x[n][:, 5:7], x[n][:, 3:5]], dim=1) for n in NODE_TYPES}
        else:
            pred = model(g)
        for n in NODE_TYPES:
            if len(y[n]) == 0:
                continue
            err = ((pred[n] - y[n]) ** 2).mean(dim=1).numpy()
            rows += [(t, i, n, float(e)) for i, e in zip(ids[n], err)]
    return pd.DataFrame(rows, columns=["time", "id", "type", "err"])

if __name__ == "__main__":
    samples = torch.load("data/processed/samples.pt", weights_only=False)
    model = load_model()
    for name, m in (("GNN", model), ("Constant velocity", None)):
        val = score_samples(samples["val"], m)
        test = score_samples(samples["test"], m)
        thr = val.groupby("type")["err"].quantile(0.99)
        test["alert"] = test["err"] > test["type"].map(thr)
        print(f"\n=== {name} ===")
        print("Threshold (99th percentile of validation error):")
        print(thr.round(4).to_string())
        print("Rows in test set:")
        print(test.groupby("type").size().to_string())
        print("Alert rate on CLEAN test data (about 1-3% is expected):")
        print(test.groupby("type")["alert"].mean().round(4).to_string())
        if m is not None:
            thr.to_json("data/processed/thresholds_gnn.json")