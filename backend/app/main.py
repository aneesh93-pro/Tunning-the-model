"""FastAPI GPU Worker entry point for remote CUDA training and inference."""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MiniMind AI GPU Worker API", version="1.0.0")

_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",")
    if origin.strip()
]
if os.getenv("VERCEL_URL"):
    _origins.append(f"https://{os.getenv('VERCEL_URL')}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "minimind-gpu-worker", "status": "ok"}


@app.get("/health")
def health():
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "cpu"
    except ImportError:
        cuda = False
        device = "torch-not-installed"
    return {"status": "ok", "cuda": cuda, "device": device}
