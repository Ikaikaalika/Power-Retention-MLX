from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
import json
import asyncio

app = FastAPI(title="Power Retention Dashboard")

# Setup templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mock training status
training_status = {
    "is_training": False,
    "step": 0,
    "loss": 0.0,
    "logs": []
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # In a real app, this would read from a shared state or database
            # For now, we just send the current status
            await websocket.send_json(training_status)
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.post("/api/start_training")
async def start_training():
    global training_status
    training_status["is_training"] = True
    # In real app, this would spawn a subprocess
    return {"status": "started"}

@app.post("/api/stop_training")
async def stop_training():
    global training_status
    training_status["is_training"] = False
    return {"status": "stopped"}

def run_dashboard(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_dashboard()
