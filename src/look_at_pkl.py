import pickle
import pandas as pd

f = "data/raw/ais_dtu/labelled/data_AIS_Custom_13122021_13122021_CarFisHigMilPasPleSaiTan_600_99999999_0.pkl"

tracks = []
with open(f, "rb") as fh:
    while True:
        try:
            tracks.append(pickle.load(fh))
        except EOFError:
            break

print("=== SHIPS (AIS) ===")
print("Ships in file:", len(tracks))
rows = pd.concat([
    pd.DataFrame({"mmsi": t["mmsi"], "lat": t["lat"], "lon": t["lon"],
                  "speed": t["speed"], "course": t["course"], "ts": t["timestamp"]})
    for t in tracks], ignore_index=True)
rows["time"] = pd.to_datetime(rows["ts"], unit="s")
print("Total rows:", len(rows))
print("Lat range:", rows.lat.min(), "to", rows.lat.max())
print("Lon range:", rows.lon.min(), "to", rows.lon.max())
print("Time range:", rows.time.min(), "to", rows.time.max())
print("Typical seconds between reports:", rows.groupby("mmsi")["ts"].diff().median())
print("Max speed:", rows.speed.max())

print("\n=== AIRCRAFT (ADS-B) ===")
a = pd.read_csv("data/raw/adsb_log.csv")
print("Aircraft seen:", a.icao24.nunique())
print("Lat range:", a.lat.min(), "to", a.lat.max())
print("Lon range:", a.lon.min(), "to", a.lon.max())
print("Time range:", a.recorded_at.min(), "to", a.recorded_at.max())