import { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const WS_URL = "ws://localhost:8000/ws";

export default function App() {
  const [frame, setFrame] = useState(null);
  const [frameNo, setFrameNo] = useState(0);
  const [incidents, setIncidents] = useState([]);
  const [status, setStatus] = useState("connecting...");
  const [run, setRun] = useState(0);
  const counter = useRef(0);

  useEffect(() => {
    counter.current = 0;
    setIncidents([]);
    setFrame(null);
    setFrameNo(0);
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => setStatus("streaming");
    ws.onclose = () => setStatus("replay finished");
    ws.onerror = () => setStatus("cannot reach the backend (is uvicorn running?)");
    ws.onmessage = (e) => {
      const f = JSON.parse(e.data);
      const n = ++counter.current;
      setFrame(f);
      setFrameNo(n);
      const alerts = f.vehicles.filter((v) => v.alert);
      if (alerts.length === 0) return;
      setIncidents((prev) => {
        const next = [...prev];
        for (const v of alerts) {
          const i = next.findIndex((x) => x.id === v.id && n - x.lastFrame <= 3);
          if (i >= 0) {
            const old = next[i];
            next[i] = { ...old, lastFrame: n,
              ...(v.score > old.score ? { score: v.score, explanation: v.explanation } : {}) };
          } else {
            next.push({ id: v.id, type: v.type, firstTime: f.t, lastFrame: n, score: v.score,
              explanation: v.explanation, injected: v.injected });
          }
        }
        return next;
      });
    };
    return () => {
      ws.onopen = ws.onclose = ws.onerror = ws.onmessage = null;
      ws.close();
    };
  }, [run]);

  const redIds = new Set(incidents.filter((x) => frameNo - x.lastFrame <= 4).map((x) => x.id));
  const vehicles = frame ? frame.vehicles : [];
  const ships = vehicles.filter((v) => v.type === "ship").length;
  const aircraft = vehicles.length - ships;

  return (
    <div style={{ display: "flex", height: "100%" }}>
      <MapContainer center={[55.25, 14.5]} zoom={8} style={{ flex: 1 }}>
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {vehicles.map((v) => {
          const red = redIds.has(v.id);
          const color = red ? "#e11d48" : v.type === "ship" ? "#2563eb" : "#16a34a";
          return (
            <CircleMarker key={v.id} center={[v.lat, v.lon]}
              radius={red ? 9 : v.type === "ship" ? 4 : 6}
              pathOptions={{ color, fillColor: color, fillOpacity: 0.85, weight: 1 }}>
              <Tooltip>{v.type} {v.id}</Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>

      <div style={{ width: 380, overflowY: "auto", padding: 14, borderLeft: "1px solid #ddd" }}>
        <h2 style={{ margin: "0 0 4px" }}>Aero-Marine-GNN</h2>
        <div style={{ fontSize: 13, color: "#555" }}>Status: {status}</div>
        <div style={{ margin: "10px 0", fontSize: 15 }}>
          <b>{frame ? frame.t : "--"}</b> (replayed, 1 second = 1 minute)
        </div>
        <div style={{ fontSize: 14, marginBottom: 8 }}>
          <span style={{ color: "#2563eb" }}>&#9679; ships {ships}</span>{" "}
          <span style={{ color: "#16a34a" }}>&#9679; aircraft {aircraft}</span>{" "}
          <span style={{ color: "#e11d48" }}>&#9679; alerts {incidents.length}</span>
        </div>
        <button onClick={() => setRun((r) => r + 1)} style={{ marginBottom: 12 }}>
          Restart replay
        </button>

        <h3 style={{ margin: "8px 0" }}>Alerts</h3>
        {incidents.length === 0 && <div style={{ color: "#777" }}>No alerts yet.</div>}
        {[...incidents].reverse().map((x) => (
          <div key={x.id + x.firstTime}
            style={{ borderLeft: "4px solid #e11d48", background: "#fff5f6",
                     padding: "8px 10px", marginBottom: 10, fontSize: 13 }}>
            <div><b>{x.type} {x.id}</b> &middot; first seen {x.firstTime}</div>
            <div style={{ margin: "4px 0" }}>{x.explanation}</div>
            <div style={{ color: "#777" }}>
              score {x.score} &middot; {x.injected ? "injected test anomaly" : "not an injected anomaly"}
            </div>
          </div>
        ))}

        <div style={{ marginTop: 16, fontSize: 11, color: "#777" }}>
          Replay of recorded ADS-B (Sept 2026) and AIS (Dec 2021, time-aligned) traffic with
          synthetic anomalies. Attention neighbours are hints, not proof.
        </div>
      </div>
    </div>
  );
}