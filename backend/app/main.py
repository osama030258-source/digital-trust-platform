from fastapi import FastAPI
from backend.app.core.model_loader import load_all_models
from backend.app.api.routes import router

app = FastAPI(title="Digital Trust Platform API", version="0.1.0")

app.include_router(router, prefix="/api", tags=["verification"])


@app.on_event("startup")
def startup_event():
    load_all_models()


@app.get("/")
def root():
    return {"message": "Digital Trust Platform API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}