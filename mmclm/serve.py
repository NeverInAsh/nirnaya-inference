"""HTTP endpoint for the audio engine. POST multipart/form-data with `request` (JSON: model, state, questions) and
`audio` (WAV), or application/json for a text-only request. Returns {"model", "answers": {id: {"choice", "probabilities"}}}.

    NIRNAYA_API_KEY=<key> python -m mmclm.serve --model <model dir> --port 8000

Binds to 127.0.0.1 by default. Requests run one at a time.
"""

from __future__ import annotations

import argparse
import collections
import hmac
import json
import os
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


def state_text(state):
    """{"context": "..."} -> the context text; anything else is passed on unchanged."""
    return state["context"] if isinstance(state, dict) and set(state) == {"context"} else state


def build_app(engine, api_key=None, max_cached=10000, max_body=32 * 1024 * 1024):
    app, lock, done = FastAPI(docs_url=None, redoc_url=None, openapi_url=None), threading.Lock(), collections.OrderedDict()

    @app.post("/v1/decisions")
    async def decide(req: Request):
        if api_key and not hmac.compare_digest(req.headers.get("authorization", ""), f"Bearer {api_key}"):
            raise HTTPException(401, "unauthorized")
        if int(req.headers.get("content-length") or 0) > max_body:
            raise HTTPException(413, "request too large")
        audios = []
        try:
            if req.headers.get("content-type", "").startswith("multipart/"):
                form = await req.form()
                body = json.loads(form["request"])
                if "audio" in form:
                    audios = [await form["audio"].read()]
            else:
                body = await req.json()
            state, questions = body.get("state"), body["questions"]
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise HTTPException(400, f"bad request: {type(e).__name__}") from e
        key = req.headers.get("idempotency-key")
        with lock:
            if key and key in done:
                return JSONResponse(done[key])
            try:
                out, _ = engine(state_text(state), questions, audios=audios)
            except Exception as e:  # noqa: BLE001  never echo inputs back
                raise HTTPException(422, f"could not score this request: {type(e).__name__}") from e
            out = {"model": body.get("model", out["model"]), "answers": out["answers"]}
            if key:
                done[key] = out
                while len(done) > max_cached:
                    done.popitem(last=False)
        return JSONResponse(out)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--twice-max", type=int, default=3000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    import uvicorn

    from mmclm.di_engine import NirnayaAudioEngine

    engine = NirnayaAudioEngine(model=a.model, head=f"{a.model}/head.pt", temperature_fit=f"{a.model}/t_entropy.json",
                                twice_max=a.twice_max, max_model_len=a.max_model_len, gpu_mem=a.gpu_mem)
    uvicorn.run(build_app(engine, os.environ.get("NIRNAYA_API_KEY")), host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()
