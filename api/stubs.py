# stubs.py - the two endpoints that are NOT built yet (/compare, /detect and /evidence are real now, see compare.py, detect.py and evidence.py).
# Each one answers with the fixed example from api/contract.json, so the website can already
# be wired to every endpoint. Every stub answer has:
#   - a "note" that starts with "stub:"
#   - the response header X-Stub: true
# The upload endpoints already check the upload (PNG or JPG, at most 5 MB, see common.py),
# so the website sees the real error behaviour now.

import copy
import json
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from api.common import read_upload

# The contract file sits next to this file, so the path works on any computer.
CONTRACT = json.loads((Path(__file__).parent / "contract.json").read_text(encoding="utf-8"))["endpoints"]

router = APIRouter()


def stub_answer(endpoint):
    """The example response of this endpoint from the contract, marked as a stub."""
    answer = copy.deepcopy(CONTRACT[endpoint]["response"])
    if not answer["note"].startswith("stub:"):
        answer["note"] = "stub: " + answer["note"]
    return JSONResponse(answer, headers={"X-Stub": "true"})


@router.get("/audit/results")
def audit_results():
    return stub_answer("GET /audit/results")


@router.post("/audit/check")
async def audit_check(image: UploadFile = File(...)):
    await read_upload(image)
    return stub_answer("POST /audit/check")
