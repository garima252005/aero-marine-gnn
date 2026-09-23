import requests, pandas as pd, time, json
from datetime import datetime, timezone

LAMIN, LAMAX = 55.4, 55.9
LOMIN, LOMAX = 12.3, 13.0
OUT_FILE = "data/raw/adsb_log.csv"
POLL_SECONDS = 20

with open("credentials.json") as f:
    creds = json.load(f)

def get_token():
    r = requests.post(
        "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token",
        data={"grant_type": "client_credentials",
              "client_id": creds["clientId"],
              "client_secret": creds["clientSecret"]},
        timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]

def fetch_states(token):
    url = "https://opensky-network.org/api/states/all"
    params = {"lamin": LAMIN, "lamax": LAMAX, "lomin": LOMIN, "lomax": LOMAX}
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def main():
    print("Starting ADS-B recorder. Press Ctrl+C to stop.")
    token = get_token()
    token_time = time.time()
    while True:
        try:
            if time.time() - token_time > 1500:
                token = get_token()
                token_time = time.time()
            data = fetch_states(token)
            ts = datetime.now(timezone.utc).isoformat()
            states = data.get("states") or []
            rows = [{
                "recorded_at": ts, "icao24": s[0], "callsign": (s[1] or "").strip(),
                "time_position": s[3], "lon": s[5], "lat": s[6],
                "baro_altitude": s[7], "velocity": s[9], "heading": s[10],
                "vertical_rate": s[11],
            } for s in states]
            if rows:
                df = pd.DataFrame(rows)
                import os
                header = not (os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0)
                df.to_csv(OUT_FILE, mode="a", header=header, index=False)
                print(f"{ts} — logged {len(rows)} aircraft")
            else:
                print(f"{ts} — no aircraft in range")
        except Exception as e:
            print("Error:", e)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    main()