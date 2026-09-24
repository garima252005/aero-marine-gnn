import numpy as np, pandas as pd, torch
from scipy.spatial import cKDTree
from step3_graphs import load_frames, make_sample
from step5_eval import load_model, score_samples
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
MIN = pd.Timedelta(minutes=1)

def rotate(dx, dy, a):
    return np.cos(a) * dx - np.sin(a) * dy, np.sin(a) * dx + np.cos(a) * dy

def anchor(test, vid, t0):
    r = test[(test.id == vid) & (test.time == t0 - MIN)].iloc[0]
    return r.x_km, r.y_km

# ---------- the anomaly functions (each starts at minute t0) ----------
def course_change(test, vid, t0):
    ax, ay = anchor(test, vid, t0)
    m = (test.id == vid) & (test.time >= t0)
    ang = np.deg2rad(rng.uniform(60, 120)) * rng.choice([-1, 1])
    rx, ry = rotate(test.loc[m, "x_km"] - ax, test.loc[m, "y_km"] - ay, ang)
    test.loc[m, "x_km"], test.loc[m, "y_km"] = ax + rx, ay + ry
    hs, hc = rotate(test.loc[m, "heading_sin"], test.loc[m, "heading_cos"], ang)
    test.loc[m, "heading_sin"], test.loc[m, "heading_cos"] = hs, hc

def speed_change(test, vid, t0):
    ax, ay = anchor(test, vid, t0)
    m = (test.id == vid) & (test.time >= t0)
    k = rng.uniform(2, 3) if rng.random() < 0.5 else rng.uniform(0, 0.1)
    test.loc[m, "x_km"] = ax + k * (test.loc[m, "x_km"] - ax)
    test.loc[m, "y_km"] = ay + k * (test.loc[m, "y_km"] - ay)
    test.loc[m, "speed"] = test.loc[m, "speed"] * k

def spoof_jump(test, vid, t0):
    m = (test.id == vid) & (test.time >= t0)
    d, ang = rng.uniform(5, 20), rng.uniform(0, 2 * np.pi)
    test.loc[m, "x_km"] = test.loc[m, "x_km"] + d * np.cos(ang)
    test.loc[m, "y_km"] = test.loc[m, "y_km"] + d * np.sin(ang)

def eligible(test, typ, min_len):
    out = {}
    for vid, g in test[test["type"] == typ].groupby("id"):
        t = g["time"]
        if len(g) >= min_len and (t.diff().dropna() == MIN).all():
            out[vid] = t.tolist()
    return out

def add_meetings(test, n_pairs, used, events):
    """Two nearby ships steer towards each other and then travel together."""
    by_time = {t: g.set_index("id") for t, g in test.groupby("time")}
    times = sorted(by_time)
    ships = set(test[test["type"] == "ship"].id) - used
    found = 0
    for _ in range(500):
        if found >= n_pairs:
            break
        i = int(rng.integers(5, len(times) - 14))
        steps = [times[i + k] for k in range(-5, 13)]
        if steps[-1] - steps[0] != 17 * MIN:
            continue
        t0 = times[i]
        ids = sorted(set.intersection(*[set(by_time[s].index) for s in steps]) & ships)
        if len(ids) < 2:
            continue
        P = by_time[t0 - MIN].loc[ids, ["x_km", "y_km"]].values
        pairs = list(cKDTree(P).query_pairs(6.0))       # ships less than 6 km apart
        if not pairs:
            continue
        a, b = pairs[int(rng.integers(len(pairs)))]
        A, B = ids[a], ids[b]
        pa = test[test.id == A].set_index("time")[["x_km", "y_km"]]
        pb = test[test.id == B].set_index("time")[["x_km", "y_km"]]
        ct = pa.index.intersection(pb.index)
        ct = ct[ct >= t0]
        PA, PB = pa.loc[ct].values, pb.loc[ct].values
        M = (PA + PB) / 2
        k = np.array([(t - t0) / MIN for t in ct])
        w = np.minimum(1, (k + 1) / 8)[:, None]         # 8 minutes to come together
        for vid, P in ((A, PA), (B, PB)):
            m = (test.id == vid) & test.time.isin(ct)
            test.loc[m, ["x_km", "y_km"]] = (1 - w) * P + w * M
            events.append(dict(id=vid, kind="meeting", vtype="ship",
                               ws=t0, we=t0 + 12 * MIN))
            used.add(vid)
        found += 1
    print("Ship-ship meetings injected:", found, "pairs")

def add_air_sea(test, n_events, used, events):
    """An aircraft slows down, flies to a nearby ship and follows it for a while."""
    by_time = {t: g.set_index("id") for t, g in test.groupby("time")}
    ships_all = set(test[test["type"] == "ship"].id) - used
    tracks = {v: ts for v, ts in eligible(test, "aircraft", 16).items() if v not in used}
    ids = list(tracks)
    rng.shuffle(ids)
    found = 0
    for vid in ids:
        if found >= n_events:
            break
        ts = tracks[vid]
        for _ in range(40):
            t0 = ts[int(rng.integers(5, len(ts) - 10))]
            steps = [t0 + k * MIN for k in range(10)]
            if any(s not in by_time for s in steps):
                continue
            ships = sorted(set.intersection(*[set(by_time[s].index) for s in [t0 - MIN] + steps]) & ships_all)
            if not ships:
                continue
            a0 = by_time[t0 - MIN].loc[vid, ["x_km", "y_km"]].values.astype(float)
            P = by_time[t0 - MIN].loc[ships, ["x_km", "y_km"]].values
            d = np.linalg.norm(P - a0, axis=1)
            if d.min() > 15.0:                           # need a ship within 15 km
                continue
            ship = ships[int(d.argmin())]
            S = np.array([by_time[s].loc[ship, ["x_km", "y_km"]].values for s in steps], dtype=float)
            path = np.array([(a0 + min(1, (k + 1) / 5) * (S[4] - a0)) if k <= 4 else S[k]
                             for k in range(10)])
            m = (test.id == vid) & test.time.isin(steps)
            test.loc[m, ["x_km", "y_km"]] = path
            events.append(dict(id=vid, kind="air-sea", vtype="aircraft",
                                ws=t0 - MIN, we=t0))
            used.add(vid)
            found += 1
            break
    print("Air-sea loiter events injected:", found)
# ---------- evaluation ----------
def evaluate(scores, thr, events):
    s = scores.copy()
    s["label"], s["skip"] = 0, False
    for e in events:
        m = s.id == e["id"]
        s.loc[m & (s.time >= e["ws"]) & (s.time <= e["we"]), "label"] = 1
        s.loc[m & (s.time > e["we"]), "skip"] = True    # ignore the "new normal" afterwards
    s = s[~s.skip].copy()
    s["alert"] = s["err"] > s["type"].map(thr)
    rows = []
    for typ, g in s.groupby("type"):
        tp = int((g.alert & (g.label == 1)).sum()); fp = int((g.alert & (g.label == 0)).sum())
        fn = int((~g.alert & (g.label == 1)).sum()); tn = int((~g.alert & (g.label == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        auc = roc_auc_score(g.label, g["err"]) if g.label.nunique() == 2 else np.nan
        rows.append(dict(type=typ, precision=prec, recall=rec, f1=f1,
                         fpr=fp / (fp + tn), auc=auc))
    det = []
    for e in events:
        sub = s[(s.id == e["id"]) & (s.time >= e["ws"]) & (s.time <= e["we"])]
        det.append(dict(kind=e["kind"], vtype=e["vtype"], detected=bool(sub.alert.any())))
    det = pd.DataFrame(det).groupby(["kind", "vtype"])["detected"].mean().round(2)
    return pd.DataFrame(rows).set_index("type").round(3), det

if __name__ == "__main__":
    df = pd.read_csv("data/processed/unified.csv", parse_dates=["time"])
    test = df[df.split == "test"].copy().sort_values(["id", "time"]).reset_index(drop=True)

    used, events = set(), []
    plan = [("course", course_change), ("speed", speed_change), ("spoof", spoof_jump)]
    sizes = [("ship", 20, 20), ("aircraft", 8, 12)]          # (type, events per kind, min minutes)
    for kind, fn in plan:
        for typ, n, min_len in sizes:
            el = {v: ts for v, ts in eligible(test, typ, min_len).items() if v not in used}
            ids = list(el)
            rng.shuffle(ids)
            for vid in ids[:n]:
                ts = el[vid]
                t0 = ts[int(rng.integers(5, len(ts) - 2))]
                fn(test, vid, t0)
                events.append(dict(id=vid, kind=kind, vtype=typ, ws=t0 - MIN, we=t0))
                used.add(vid)
    add_meetings(test, 8, used, events)
    add_air_sea(test, 6, used, events)

    log = pd.DataFrame(events)
    log.to_csv("data/processed/anomaly_log.csv", index=False)
    print("\nEvents injected (ground truth saved to data/processed/anomaly_log.csv):")
    print(log.groupby(["kind", "vtype"]).size().to_string())

    # rebuild the columns that depend on the changed values
    stats = pd.read_json("data/processed/speed_stats.json")
    test["speed_n"] = (test["speed"] - test["type"].map(stats["mean"])) / test["type"].map(stats["std"])
    test.to_csv("data/processed/unified_test_corrupted.csv", index=False)
    _, frames = load_frames("data/processed/unified_test_corrupted.csv")
    samples = []
    for t in sorted(frames):
        s = make_sample(frames, t)
        if s:
            samples.append(s)

    model = load_model()
    val = torch.load("data/processed/samples.pt", weights_only=False)["val"]
    results = {}
    for name, m in (("GNN", model), ("Constant velocity", None)):
        thr = score_samples(val, m).groupby("type")["err"].quantile(0.99)
        rows, det = evaluate(score_samples(samples, m), thr, events)
        results[name] = det
        print(f"\n=== {name}: metrics per vehicle type ===")
        print(rows.to_string())
    print("\n=== Share of injected events detected (any alert inside the event window) ===")
    print(pd.concat(results, axis=1).to_string())
