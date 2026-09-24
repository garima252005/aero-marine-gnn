import json
import pandas as pd
from pyproj import Transformer

tf = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
inv = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

# recover the offset Step 2 subtracted, so x/y in km can be turned back into lat/lon
orig = pd.read_csv("data/processed/unified.csv", nrows=1)
x0, y0 = tf.transform(orig.lon[0], orig.lat[0])
off_x, off_y = x0 / 1000 - orig.x_km[0], y0 / 1000 - orig.y_km[0]

df = pd.read_csv("data/processed/unified_test_corrupted.csv", parse_dates=["time"])
lon, lat = inv.transform(((df.x_km + off_x) * 1000).values, ((df.y_km + off_y) * 1000).values)
df["lat"], df["lon"] = lat, lon

alerts = pd.read_csv("data/processed/alerts.csv", parse_dates=["time"])
alerts["time"] = alerts["time"] + pd.Timedelta(minutes=1)   # alert appears when the odd position arrives
alert_map = {(r.time, r.id): r for r in alerts.itertuples()}

frames = []
for t, g in df.groupby("time"):
    vehicles = []
    for r in g.itertuples():
        a = alert_map.get((t, r.id))
        vehicles.append(dict(id=r.id, type=r.type, lat=round(r.lat, 5), lon=round(r.lon, 5),
                             alert=a is not None,
                             score=round(float(a.score), 1) if a is not None else 0,
                             injected=bool(a.is_injected) if a is not None else False,
                             explanation=a.text if a is not None else ""))
    frames.append(dict(t=t.strftime("%Y-%m-%d %H:%M"), vehicles=vehicles))

with open("backend/replay.json", "w") as f:
    json.dump(frames, f)
print("Saved backend/replay.json:", len(frames), "minutes,",
      sum(v["alert"] for fr in frames for v in fr["vehicles"]), "alerts shown")