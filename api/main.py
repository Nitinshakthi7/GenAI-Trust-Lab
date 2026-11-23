# main.py - the web API (FastAPI) that the website talks to.
#
# Start it from the project folder with:
#   .venv\Scripts\python -m uvicorn api.main:app --port 8000
# Then open http://localhost:8000/docs for an interactive page where you can try every endpoint.
#
# Real endpoints here: GET /health, GET /generators, POST /generate, POST /compare (without the Words panel),
# POST /detect (the DEV detector: trained on one weak generator only, see detect.py)
# and POST /evidence (EXIF only, see evidence.py).
# The other two endpoints are stubs (fixed example answers) and live in stubs.py.
# Every error is answered as {"error": "<plain text>"}:
#   400 bad request, 404 unknown generator or address, 503 model not ready, 500 anything else.

import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import compare, detect, evidence, generators, stubs
from api.common import MAX_UPLOAD_BYTES, ApiError, read_upload  # MAX_UPLOAD_BYTES (5 MB) is used by the upload endpoints in stubs.py

# Which website addresses may call this API. The default is the Vite website on this computer.
# Set ALLOWED_ORIGINS (comma-separated) to change it. It holds addresses only, never secrets.
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]

app = FastAPI(title="GenAI Trust Lab API", description="Image generators, plus stubs for the detector and audit tools.")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], expose_headers=["X-Stub"])
app.include_router(stubs.router)


# ----------------------------------------------------------------- error answers

def error_answer(status, message):
    return JSONResponse({"error": message}, status_code=status)


@app.exception_handler(ApiError)
async def on_api_error(request, error):
    return error_answer(error.status, error.message)


@app.exception_handler(RequestValidationError)
async def on_bad_request(request, error):
    first = error.errors()[0]                                   # show only the first problem, in plain text
    where = ".".join(str(part) for part in first["loc"][1:])
    return error_answer(400, f"bad request: {where + ': ' if where else ''}{first['msg']}")


@app.exception_handler(StarletteHTTPException)
async def on_http_error(request, error):
    return error_answer(error.status_code, "not found" if error.status_code == 404 else str(error.detail))


@app.exception_handler(Exception)
async def on_any_error(request, error):
    return error_answer(500, "something went wrong on the server")


# --------------------------------------------------------------- real endpoints

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/generators")
def get_generators():
    return generators.list_generators()


@app.post("/generate")
def post_generate(request: generators.GenerateRequest):
    return generators.generate(request)


@app.post("/detect")
async def post_detect(image: UploadFile = File(...)):
    data = await read_upload(image)                  # PNG or JPG, at most 5 MB, else 400
    return await run_in_threadpool(detect.detect, data)   # the model work runs off the main loop


@app.post("/evidence")
async def post_evidence(image: UploadFile = File(...)):
    data = await read_upload(image)                  # PNG or JPG, at most 5 MB, else 400
    return await run_in_threadpool(evidence.evidence, data)


@app.post("/compare")
async def post_compare(image_a: UploadFile = File(...), image_b: UploadFile = File(...),
                       use_words: bool = Form(False), focus: str = Form("")):
    data_a = await read_upload(image_a)              # PNG or JPG, at most 5 MB, else 400
    data_b = await read_upload(image_b)
    return await run_in_threadpool(compare.compare, data_a, data_b, use_words, focus)   # the maths runs off the main loop
