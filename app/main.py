from fastapi import FastAPI
from app.api.routers import api_router

app = FastAPI(
    title="끄적 프로젝트 AI 서버",
    description="API Documentation",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1")
