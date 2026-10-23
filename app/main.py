"""FastAPI application entry point for Check-in Buddy."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app import models  # noqa: F401 - registers SQLAlchemy models
from app.routes import router

APP_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Check-in Buddy",
    description="Wellness companion for older adults",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(router, prefix="/api")


@app.get("/")
def senior_view():
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/family")
def family_dashboard_page():
    page = TEMPLATE_DIR / "family.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Family dashboard page is unavailable.")
    return FileResponse(page)


@app.get("/health")
def health():
    from app.llm import provider_status

    return {"status": "ok", **provider_status()}


@app.post("/seed")
def seed_data(normal: bool = False, variant: str = "demo", force: bool = False):
    from app.seed import seed_database

    allowed_variants = {"normal", "demo", "missed_meds", "low_mood", "declining"}
    if variant not in allowed_variants:
        raise HTTPException(status_code=422, detail="Unsupported seed variant.")
    try:
        return seed_database(normal=normal, variant=variant, force=force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to seed demo data.") from exc
