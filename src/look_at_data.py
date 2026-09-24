import glob
import pandas as pd

print("=== ADS-B file ===")
adsb = pd.read_csv("data/raw/adsb_log.csv")
print("Columns:", adsb.columns.tolist())
print("Rows:", len(adsb))
print(adsb.head(3))

print("\n=== AIS files ===")
files = glob.glob("data/raw/ais_dtu/**/*", recursive=True)
print("Files found:", files)
ais_files = [f for f in files if f.endswith((".csv", ".zip"))]
if ais_files:
    ais = pd.read_csv(ais_files[0], nrows=3)
    print("Columns:", ais.columns.tolist())
    print(ais)