from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse
from diffusers import DiffusionPipeline
from PIL import Image
import asyncio, io, os, torch, base64
from pathlib import Path

app = FastAPI()

# === Config ===
ROOT_DIR = Path("/home/jovyan/shared/tlu37/genex-server/I2P")
ROOT_DIR.mkdir(parents=True, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained(
    "TaiMingLu/GenEx-World-Initializer",
    custom_pipeline="genex_world_initializer_pipeline",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda")

queue = asyncio.Queue()
active_ips = set()
user_generations = {}  # ip -> list of (index, path)

# === Helper: Load existing generations on startup ===
def load_existing_generations():
    for ip_dir in ROOT_DIR.iterdir():
        if not ip_dir.is_dir():
            continue
        ip = ip_dir.name
        gen_list = []
        for img_file in sorted(ip_dir.glob("*.png")):
            try:
                index = int(img_file.stem)
                gen_list.append((index, img_file))
            except ValueError:
                continue
        if gen_list:
            user_generations[ip] = sorted(gen_list)

@app.post("/generate/")
async def generate(request: Request, image: UploadFile = File(...)):
    ip = request.client.host

    if ip in active_ips:
        return JSONResponse({"status": "Already queued or generating."}, status_code=429)

    future = asyncio.Future()
    await queue.put((ip, image, future))
    active_ips.add(ip)

    position = sum(1 for item in queue._queue if item[0] != ip)

    try:
        result = await asyncio.wait_for(future, timeout=300)
        return StreamingResponse(result, media_type="image/png")
    except asyncio.TimeoutError:
        return JSONResponse({"status": "Timeout, try again later"}, status_code=504)
    finally:
        active_ips.remove(ip)

@app.get("/status/")
async def queue_status(request: Request):
    ip = request.client.host
    pos = sum(1 for item in queue._queue if item[0] != ip)
    return {"queue_position": pos, "in_queue": ip in active_ips}

@app.get("/history/")
async def get_history(request: Request):
    ip = request.client.host
    items = user_generations.get(ip, [])
    return {
        "total_generations": len(items),
        "generation_ids": [i for i, _ in items]
    }

@app.get("/recent/")
async def recent_generations(request: Request, count: int = 1):
    ip = request.client.host
    items = user_generations.get(ip, [])

    if not items:
        return JSONResponse({"error": "No generations found."}, status_code=404)

    count = max(1, min(count, len(items)))
    recent = items[-count:]

    encoded = []
    for idx, path in recent:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            encoded.append({"index": idx, "image_base64": b64})

    return {"recent": encoded}

async def worker_loop():
    while True:
        ip, image_file, future = await queue.get()
        try:
            image = Image.open(image_file.file).convert("RGB")
            _, output = pipe(image=image)
            img = output.images[0]

            # Save to disk
            ip_dir = ROOT_DIR / ip
            ip_dir.mkdir(parents=True, exist_ok=True)
            index = len(user_generations.get(ip, []))
            path = ip_dir / f"{index}.png"
            img.save(path)

            # Update memory index (path only)
            user_generations.setdefault(ip, []).append((index, path))

            # Return result
            with open(path, "rb") as f:
                buf = io.BytesIO(f.read())
            future.set_result(buf)
        except Exception as e:
            future.set_exception(e)
        finally:
            queue.task_done()

@app.on_event("startup")
async def startup_event():
    load_existing_generations()
    asyncio.create_task(worker_loop())
