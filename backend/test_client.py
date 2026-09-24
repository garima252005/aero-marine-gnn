import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        for _ in range(3):
            f = json.loads(await ws.recv())
            n_alerts = sum(v["alert"] for v in f["vehicles"])
            print(f["t"], "-", len(f["vehicles"]), "vehicles,", n_alerts, "alerts")

asyncio.run(main())