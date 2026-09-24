import numpy as np, pandas as pd, torch
from scipy.spatial import cKDTree
from torch_geometric.data import HeteroData

FEATS   = ["x_s", "y_s", "speed_n", "heading_sin", "heading_cos", "dx_s", "dy_s"]
RADIUS  = {("ship", "ship"): 10, ("aircraft", "aircraft"): 50,
           ("aircraft", "ship"): 20, ("ship", "aircraft"): 20}      # km
DXY     = {"ship": 0.3, "aircraft": 5.0}   # rough km-per-minute scale for targets
POS_SCALE = 100                            # your region is about 190 km wide
T = 5                                      # minutes of history per sample
EDGE_TYPES = [("ship", "near", "ship"), ("aircraft", "near", "aircraft"),
              ("aircraft", "near", "ship"), ("ship", "near", "aircraft")]

def load_frames(path="data/processed/unified.csv"):
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values(["id", "time"]).reset_index(drop=True)
    df["x_s"], df["y_s"] = df.x_km / POS_SCALE, df.y_km / POS_SCALE
    # movement during the previous minute, on the same scale as the targets
    gap = df.groupby("id")["time"].diff()
    scale = df["type"].map(DXY)
    for col, new in (("x_km", "dx_s"), ("y_km", "dy_s")):
        d = df.groupby("id")[col].diff()
        d[gap != pd.Timedelta(minutes=1)] = 0.0
        df[new] = d / scale
    return df, {t: g.set_index("id") for t, g in df.groupby("time")}

def radius_edges(src, dst, r, same):
    if len(src) == 0 or len(dst) == 0:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 1)
    nbrs = cKDTree(dst).query_ball_point(src, r)
    s = np.repeat(np.arange(len(src)), [len(n) for n in nbrs])
    d = np.array([j for n in nbrs for j in n], dtype=int)
    if same:
        keep = s != d
        s, d = s[keep], d[keep]
    dist = np.linalg.norm(src[s] - dst[d], axis=1)
    return (torch.tensor(np.stack([s, d]), dtype=torch.long),
            torch.tensor(dist, dtype=torch.float).unsqueeze(1))

def graph_at(fr, ids):
    g, pos = HeteroData(), {}
    for n in ("ship", "aircraft"):
        sub = fr.loc[ids[n]]
        g[n].x = torch.tensor(sub[FEATS].values, dtype=torch.float)
        pos[n] = sub[["x_km", "y_km"]].values
    for (a, b), r in RADIUS.items():
        ei, dist = radius_edges(pos[a], pos[b], r, same=(a == b))
        g[a, "near", b].edge_index = ei
        g[a, "near", b].edge_attr = dist / r
    return g

def make_sample(frames, t, use_ships=True, use_air=True):
    steps = [t + pd.Timedelta(minutes=k) for k in range(-(T - 1), 2)]   # 5 inputs + 1 target
    if any(s not in frames for s in steps):
        return None
    common = set(frames[steps[0]].index)
    for s in steps[1:]:
        common &= set(frames[s].index)
    ids = {"ship":     sorted(i for i in common if i[0] == "s") if use_ships else [],
           "aircraft": sorted(i for i in common if i[0] == "a") if use_air else []}
    if not ids["ship"] and not ids["aircraft"]:
        return None
    graphs = [graph_at(frames[s], ids) for s in steps[:-1]]
    cur, nxt, y = frames[t], frames[steps[-1]], {}
    for n in ("ship", "aircraft"):
        dxy = (nxt.loc[ids[n], ["x_km", "y_km"]].values
               - cur.loc[ids[n], ["x_km", "y_km"]].values) / DXY[n]
        hd = nxt.loc[ids[n], ["heading_sin", "heading_cos"]].values
        y[n] = torch.tensor(np.hstack([dxy, hd]), dtype=torch.float)
    return graphs, y, ids, t

if __name__ == "__main__":
    df, frames = load_frames()
    split_of = df.groupby("time")["split"].first()
    samples = {"train": [], "val": [], "test": []}
    for t in sorted(frames):
        s = make_sample(frames, t)
        if s:
            samples[split_of[t]].append(s)
    print("Samples:", {k: len(v) for k, v in samples.items()})

    print("\nAverage edges per graph (train):")
    for et in EDGE_TYPES:
        m = np.mean([s[0][-1][et].edge_index.shape[1] for s in samples["train"]])
        print(" ", et, round(m, 1))

    print("\nTarget std (dx, dy, hs, hc) - should be near 1 for dx, dy:")
    for n in ("ship", "aircraft"):
        allv = torch.cat([s[1][n] for s in samples["train"] if len(s[1][n])])
        print(" ", n, allv.std(0).round(decimals=2).tolist())

    torch.save(samples, "data/processed/samples.pt")
    print("\nSaved data/processed/samples.pt")