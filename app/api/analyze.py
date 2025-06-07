import logging

from fastapi import APIRouter, Request
from app.models.analyzeRequest import AnalyzeRequest
from app.models.analyzeResponse import AnalyzeResponse
from app.models.personalityRequest import PersonalityRequest
import app.api.s3_manager as s3_manager
import ai.app as ai_model
import ai.graphology as personality_ai_model

import time

router = APIRouter()

@router.get("/test")
def analyze():
    ai_model.analyze()

@router.post("/analyze")
def generate(request: Request, req_body: AnalyzeRequest):
    model = request.app.state.model

    # 1. 이미지 다운로드
    reference_hash = s3_manager.download_reference_images(req_body.comparisonImageUrls)
    test_hash = s3_manager.download_test_image(req_body.verificationImageUrl)

    start = time.time();
    # 2. 감정 시작
    appraisal_response = ai_model.analyze(model, reference_hash, test_hash)
    appraisal_response.verificationImageUrl = req_body.verificationImageUrl

    print('latency: ', time.time() - start)

    # 3. 이미지 삭제
    s3_manager.delete_reference_images(reference_hash)
    s3_manager.delete_test_image(test_hash)

    return appraisal_response

@router.post("/personality/analyze")
def generatePersonality(request: PersonalityRequest):
    print(str)
    # 1. 이미지 다운로드
    image_hash = s3_manager.download_personality_image(request.imageUrl)

    # 2. 감정 시작
    response = personality_ai_model.main(image_hash)

    # 3. 이미지 삭제
    s3_manager.delete_personality_image(image_hash)

    return response

if __name__ == "__main__":
    personality_ai_model.main()