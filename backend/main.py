import asyncio, json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()
frames = json.loads((Path(__file__).parent / "replay.json").read_text())

@app.get("/")
def home():
    return {"status": "ok", "minutes_in_replay": len(frames)}

@app.websocket("/ws")
async def replay(ws: WebSocket):
    await ws.accept()
    try:
        for f in frames:
            await ws.send_json(f)
            await asyncio.sleep(1)      # 1 second = 1 minute of traffic
    except WebSocketDisconnect:
        pass