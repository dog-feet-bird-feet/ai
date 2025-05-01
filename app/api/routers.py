from fastapi import APIRouter
from app.api import analyze

api_router = APIRouter()
api_router.include_router(analyze.router, tags=["analyze"])
