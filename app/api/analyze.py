from fastapi import APIRouter
from app.models.analyzeRequest import AnalyzeRequest
from app.models.analyzeResponse import AnalyzeResponse
import app.api.s3_manager as s3_manager
import ai.infer_from_folder as ai_model

router = APIRouter()

@router.get("/test")
def analyze():
    return {"id": 3}

@router.post("/analyze")
def generate(request: AnalyzeRequest):
    
    # 1. 이미지 다운로드
    s3_manager.download_reference_images(request.reference_urls)
    s3_manager.download_test_image(request.test_url)

    # 2. 감정 시작
    ai_model.analyze()
    appraisal_response = AnalyzeResponse(request.id, 1.0, 23.1, 222.4, "https://s3.aws.com/verifi-url?q=AgfDs3dF5Fgas")

    # 3. 이미지 삭제
    s3_manager.delete_reference_images()
    s3_manager.delete_test_image()

    return appraisal_response