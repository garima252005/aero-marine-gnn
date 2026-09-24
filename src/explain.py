import numpy as np, pandas as pd, torch
from step3_graphs import load_frames, make_sample, RADIUS, DXY
from step4_model import NODE_TYPES, EDGE_TYPES
from step5_eval import load_model, score_samples

def heading_deg(v):                          # v = (sin, cos); sin = east, cos = north
    return np.degrees(np.arctan2(v[0], v[1]))

def attention_table(model, graphs, ids):
    """For every vehicle: who is connected to it, with attention weight and distance (first GNN layer)."""
    g = graphs[-1]
    x = {n: torch.relu(model.inp[n](g[n].x)) for n in NODE_TYPES}
    table = {n: {} for n in NODE_TYPES}      # table[type][node index] -> [(attention, neighbour id, km)]
    convs = dict(zip(EDGE_TYPES, model.convs[0].convs.values()))
    for et in EDGE_TYPES:
        a, _, b = et
        ei = g[et].edge_index
        if ei.shape[1] == 0:
            continue
        conv = convs[et]
        _, (ei2, alpha) = conv((x[a], x[b]), ei, edge_attr=g[et].edge_attr,
                               return_attention_weights=True)
        alpha = alpha.mean(dim=1)
        km = g[et].edge_attr[:, 0] * RADIUS[(a, b)]
        for e in range(ei2.shape[1]):
            src, dst = ei2[0, e].item(), ei2[1, e].item()
            table[b].setdefault(dst, []).append((alpha[e].item(), ids[a][src], km[e].item()))
    return table

def explain(n, j, pred, actual, att, ids):
    p, yv = pred.numpy(), actual.numpy()
    exp_km = np.linalg.norm(p[:2]) * DXY[n]
    act_km = np.linalg.norm(yv[:2]) * DXY[n]
    dev_km = np.linalg.norm(yv[:2] - p[:2]) * DXY[n]
    turn = abs((heading_deg(yv[2:]) - heading_deg(p[2:]) + 180) % 360 - 180)
    if dev_km / DXY[n] >= 3:
        what = f"position jumped about {dev_km:.1f} km away from where it was expected (possible spoofing)"
    elif turn >= 40:
        what = f"heading is about {turn:.0f} degrees away from the expected heading (sudden course change)"
    else:
        what = f"moved {act_km:.1f} km in one minute where about {exp_km:.1f} km was expected (speed change)"
    text = f"{n.capitalize()} {ids[n][j]}: {what}."
    top = sorted(att[n].get(j, []), reverse=True)[:3]
    if top:
        text += " Nearby vehicles with the highest attention: " + \
                ", ".join(f"{nid} ({d:.1f} km)" for _, nid, d in top) + "."
    else:
        text += " No other vehicles nearby."
    return text

if __name__ == "__main__":
    model = load_model()
    val = torch.load("data/processed/samples.pt", weights_only=False)["val"]
    thr = score_samples(val, model).groupby("type")["err"].quantile(0.99)

    _, frames = load_frames("data/processed/unified_test_corrupted.csv")
    log = pd.read_csv("data/processed/anomaly_log.csv", parse_dates=["ws", "we"])
    events = log.to_dict("records")
    injected = lambda vid, t: any(e["id"] == vid and e["ws"] <= t <= e["we"] for e in events)

    rows = []
    with torch.no_grad():
        for t in sorted(frames):
            s = make_sample(frames, t)
            if not s:
                continue
            graphs, y, ids, _ = s
            pred, att = model(graphs), None
            for n in NODE_TYPES:
                if len(y[n]) == 0:
                    continue
                err = ((y[n] - pred[n]) ** 2).mean(dim=1)
                for j in torch.nonzero(err > thr[n]).flatten().tolist():
                    if att is None:
                        att = attention_table(model, graphs, ids)
                    rows.append(dict(time=t, id=ids[n][j], type=n, err=float(err[j]),
                                     score=float(err[j]) / float(thr[n]),
                                     is_injected=injected(ids[n][j], t),
                                     text=explain(n, j, pred[n][j], y[n][j], att, ids)))

    out = pd.DataFrame(rows)
    out.to_csv("data/processed/alerts.csv", index=False)
    print(f"{len(out)} alerts in total, {int(out['is_injected'].sum())} inside injected anomaly windows\n")
    print("=== Highest-scoring alerts ===")
    for _, r in out.sort_values("score", ascending=False).head(6).iterrows():
        print(f"\n[{r['time']}] score {r['score']:.1f} | injected anomaly: {r['is_injected']}\n  {r['text']}")
    print("\n=== Three alerts that were NOT injected anomalies (false alarms) ===")
    for _, r in out[~out["is_injected"]].sort_values("score").tail(3).iterrows():
        print(f"\n[{r['time']}] score {r['score']:.1f}\n  {r['text']}")