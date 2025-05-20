import logging

from fastapi import APIRouter, Request
from app.models.analyzeRequest import AnalyzeRequest
from app.models.analyzeResponse import AnalyzeResponse
import app.api.s3_manager as s3_manager
import ai.app as ai_model

router = APIRouter()

@router.get("/test")
def analyze():
    ai_model.analyze()

@router.post("/analyze")
def generate(request: Request,req_body: AnalyzeRequest):
    model = request.app.state.model

    # 1. 이미지 다운로드
    s3_manager.download_reference_images(req_body.comparisonImageUrls)
    s3_manager.download_test_image(req_body.verificationImageUrl)

    # 2. 감정 시작
    appraisal_response = ai_model.analyze(model)
    appraisal_response.verificationImageUrl = req_body.verificationImageUrl

    # 3. 이미지 삭제
    s3_manager.delete_reference_images()
    s3_manager.delete_test_image()

    return appraisal_response