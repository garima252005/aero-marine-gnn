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

ships = pd.concat([
    pd.DataFrame({"mmsi": t["mmsi"], "lat": t["lat"], "lon": t["lon"],
                  "speed": t["speed"], "course": t["course"], "ts": t["timestamp"]})
    for t in tracks], ignore_index=True)
ships["time"] = pd.to_datetime(ships["ts"], unit="s")
ships = ships.drop(columns="ts").sort_values(["mmsi", "time"])

ships.to_csv("data/processed/ais_ships.csv", index=False)
print("Saved", len(ships), "rows,", ships.mmsi.nunique(), "ships")