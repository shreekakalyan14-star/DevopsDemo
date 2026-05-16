"""
main.py  –  MindAdapt AI  |  Adaptive Learning Backend
────────────────────────────────────────────────────────

Start dev server:
    uvicorn main:app --reload --port 8000

Production:
    uvicorn main:app --workers 4 --port 8000

Env vars required:
    OPENAI_API_KEY         – required
    AWS_ACCESS_KEY_ID      – optional (enables real Polly TTS)
    AWS_SECRET_ACCESS_KEY  – optional
    AWS_S3_BUCKET          – optional
    AWS_REGION             – optional (default: us-east-1)
    AUDIO_CDN_BASE         – optional (default: mock CDN URL)
"""
from dotenv import load_dotenv
load_dotenv()
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes import submit_answer_router, generate_video_router, health_router

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = os.getenv("LOG_LEVEL", "INFO"),
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("mindadapt.main")


# ─── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MindAdapt AI starting up…")
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️  OPENAI_API_KEY not set – AI endpoints will fail")
    yield
    logger.info("🛑 MindAdapt AI shutting down")


# ─── App factory ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "MindAdapt AI",
    description = (
        "Production-ready adaptive learning backend. "
        "Evaluates answers, adapts difficulty, generates personalised "
        "explanations and TTS audio using OpenAI + AWS Polly."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Request timing middleware ───────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
    if ms > 3000:
        logger.warning("⚠️  Slow response %.0f ms  %s %s", ms, request.method, request.url.path)
    return response


# ─── Global error handler ────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(submit_answer_router)
app.include_router(generate_video_router)


# ─── Root ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "app":     "MindAdapt AI",
        "version": "1.0.0",
        "docs":    "/docs",
    }
