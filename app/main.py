from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app import deps
from app.api.routes import LATEST, VERSIONS, make_router
from app.db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Stock Move Explainer",
    description="Explains major daily stock moves with company, industry and (v2) macro/political news.\n\n"
    "Versioned by URL path. **v2** is current and adds the macro tier; **v1** is frozen and still served.",
    version="2.0.0",
    lifespan=lifespan,
)
for version in VERSIONS:
    app.include_router(make_router(version))


@app.get("/health", tags=["meta"])
def health(news=Depends(deps.get_news_provider), llm=Depends(deps.get_llm)):
    return {
        "status": "ok",
        "api_versions": [v.name for v in VERSIONS],
        "latest": LATEST.name,
        "news_provider": getattr(news, "name", None),
        "llm_model": getattr(llm, "model", None),
    }
