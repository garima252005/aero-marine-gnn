import numpy as np
import pandas as pd
from pyproj import Transformer

# Your region (same box as the ADS-B recorder)
LAT_MIN, LAT_MAX = 54.5, 56.0
LON_MIN, LON_MAX = 13.0, 16.0

# ---------- 1. Load ----------
ships_raw = pd.read_csv("data/processed/ais_ships.csv", parse_dates=["time"])
adsb_raw = pd.read_csv("data/raw/adsb_log.csv")
adsb_raw["time"] = pd.to_datetime(adsb_raw["recorded_at"], utc=True,
                                  format="ISO8601").dt.tz_localize(None)

# ---------- 2. Move ship dates (2021) onto the aircraft recording ----------
shift = adsb_raw["time"].min().floor("1min") - ships_raw["time"].min()
ships_raw["time"] = ships_raw["time"] + shift
print("Ship timestamps shifted by:", shift)

# ---------- 3. Same columns for both ----------
ships = pd.DataFrame({
    "id": "s" + ships_raw["mmsi"].astype(str), "type": "ship",
    "time": ships_raw["time"], "lat": ships_raw["lat"], "lon": ships_raw["lon"],
    "speed": ships_raw["speed"] * 1.852,        # knots -> km/h (I assume the file is in knots)
    "heading": ships_raw["course"]})

air = adsb_raw[adsb_raw["baro_altitude"] > 300]  # drops aircraft on the ground
planes = pd.DataFrame({
    "id": "a" + air["icao24"].astype(str), "type": "aircraft",
    "time": air["time"], "lat": air["lat"], "lon": air["lon"],
    "speed": air["velocity"] * 3.6,             # m/s -> km/h
    "heading": air["heading"]})

df = pd.concat([ships, planes], ignore_index=True).dropna()

# ---------- 4. Remove bad rows ----------
df = df[(df.lat != 0) & (df.lon != 0)]
df = df[df.lat.between(LAT_MIN, LAT_MAX) & df.lon.between(LON_MIN, LON_MAX)]
df = df[~((df.type == "ship") & (df.speed > 60 * 1.852))]
df = df[~((df.type == "aircraft") & (df.speed > 1300))]
df = df[df.heading.between(0, 360)]
df = df[~((df.type == "ship") & (df.groupby("id")["speed"].transform("max") <= 5))]  # moored ships

# keep only the time when BOTH ships and aircraft exist
span = df.groupby("type")["time"].agg(["min", "max"])
t_start, t_end = span["min"].max(), span["max"].min()
df = df[(df.time >= t_start) & (df.time <= t_end)]
print("Overlap window:", t_start, "to", t_end,
      "=", round((t_end - t_start).total_seconds() / 3600, 1), "hours")

# ---------- 5. lat/lon -> km ----------
tf = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
x, y = tf.transform(df["lon"].values, df["lat"].values)
df["x_km"] = x / 1000 - (x / 1000).min()
df["y_km"] = y / 1000 - (y / 1000).min()

# ---------- 6. heading -> sin/cos ----------
rad = np.deg2rad(df["heading"])
df["heading_sin"], df["heading_cos"] = np.sin(rad), np.cos(rad)

# ---------- 7. One row per vehicle per minute ----------
df["time"] = df["time"].dt.floor("1min")
num = ["lat", "lon", "x_km", "y_km", "speed", "heading_sin", "heading_cos"]
df = df.groupby(["id", "type", "time"], as_index=False)[num].mean()

parts = []
for vid, g in df.groupby("id"):
    typ = g["type"].iloc[0]
    g = g.set_index("time")[num].asfreq("1min")
    g = g.interpolate(limit=3, limit_area="inside").dropna()
    g["id"], g["type"] = vid, typ
    parts.append(g.reset_index())
df = pd.concat(parts, ignore_index=True)

# aircraft cross the area quickly, so they only need to last 8 minutes
MIN_LEN = {"ship": 30, "aircraft": 8}
df = df.groupby("id").filter(lambda g: len(g) >= MIN_LEN[g["type"].iloc[0]])

n = np.maximum(np.hypot(df.heading_sin, df.heading_cos), 1e-6)
df["heading_sin"] /= n
df["heading_cos"] /= n

# ---------- 8. Split by time, normalise speed with TRAIN stats ----------
times = np.sort(df["time"].unique())
t_train, t_val = times[int(len(times) * 0.6)], times[int(len(times) * 0.8)]
df["split"] = np.where(df.time < t_train, "train", np.where(df.time < t_val, "val", "test"))
stats = df[df.split == "train"].groupby("type")["speed"].agg(["mean", "std"])
df["speed_n"] = (df["speed"] - df["type"].map(stats["mean"])) / df["type"].map(stats["std"])
stats.to_json("data/processed/speed_stats.json")

df.to_csv("data/processed/unified.csv", index=False)
print("\nVehicles per split:")
print(df.groupby(["split", "type"])["id"].nunique())
print("\nVehicles per minute:")
print(df.groupby(["time", "type"])["id"].count().unstack().describe().round(1))