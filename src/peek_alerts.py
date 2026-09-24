import pandas as pd

a = pd.read_csv("data/processed/alerts.csv")
a["kind"] = a["text"].str.extract(r"\((possible spoofing|sudden course change|speed change)\)")[0]
print("Alerts by vehicle type and explanation kind:")
print(pd.concat([a.groupby(["type", "kind"]).size().rename("all"),
                 a[a.is_injected].groupby(["type", "kind"]).size().rename("injected")],
                axis=1).fillna(0).astype(int).to_string())
print("\nOne example per group of injected alerts:")
for (typ, kind), g in a[a.is_injected].groupby(["type", "kind"]):
    print(f"\n{typ} / {kind}:\n  {g.sort_values('score').iloc[len(g) // 2]['text']}")

vid = "s247314300"
o = pd.read_csv("data/processed/unified.csv", parse_dates=["time"])
c = pd.read_csv("data/processed/unified_test_corrupted.csv", parse_dates=["time"])
cols = ["time", "lat", "lon", "x_km", "y_km"]
w = lambda d: d[(d.id == vid) & (d.time >= "2026-09-24 07:09") & (d.time <= "2026-09-24 07:16")][cols]
print(f"\n{vid} in the ORIGINAL data:")
print(w(o).to_string(index=False))
print(f"\n{vid} after injection:")
print(w(c).to_string(index=False))
