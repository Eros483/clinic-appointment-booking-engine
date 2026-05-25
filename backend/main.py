from fastapi import FastAPI

from backend.utils.logger import logger

app = FastAPI(title="Clinic Voice Agent")


@app.get("/health")
async def health():
    return {"status": "ok"}
