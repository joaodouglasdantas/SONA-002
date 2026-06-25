"""API FastAPI do SONA — Photo → The Sims 3.

Endpoints:
  GET  /health        → status
  POST /transform     → recebe foto (multipart), retorna versão The Sims 3
  GET  /              → UI de teste

O Pipeline é criado uma vez (lifespan) e reutilizado entre requisições.
"""
from __future__ import annotations

import base64

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from . import __version__
from .config import get_settings
from .pipeline import Pipeline
from .schemas import HealthResponse, RejectionResponse, TransformResponse
from .utils.image_io import ImageDecodeError

app = FastAPI(title="SONA — Photo to The Sims 3", version=__version__)
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(version=__version__, provider=s.provider)


@app.post("/transform")
async def transform(
    file: UploadFile = File(...),
    extra_style: str = Form(""),
    seed: int | None = Form(None),
):
    s = get_settings()

    if file.content_type not in s.allowed_mime:
        raise HTTPException(415, f"tipo não suportado: {file.content_type}")

    data = await file.read()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"arquivo acima de {s.max_upload_mb} MB")

    try:
        result = get_pipeline().run(data, extra_style=extra_style, seed=seed)
    except ImageDecodeError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not result.accepted:
        # 422: imagem recebida mas rejeitada pela validação (porteiro)
        return JSONResponse(
            status_code=422,
            content=RejectionResponse(
                reason=result.reason.value if result.reason else "rejected",
                message=result.message, num_faces=result.num_faces,
                realness_score=result.realness_score,
            ).model_dump(),
        )

    return TransformResponse(
        image_base64=base64.b64encode(result.image_png).decode(),
        similarity=round(result.similarity, 4),
        fidelity_ok=result.fidelity_ok, attempts=result.attempts,
        provider=result.provider, num_faces=result.num_faces,
        realness_score=result.realness_score,
    )


@app.get("/")
def index():
    html = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return JSONResponse({"message": "SONA API", "docs": "/docs"})
