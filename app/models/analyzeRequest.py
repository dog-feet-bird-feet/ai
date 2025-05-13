from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    verificationImageUrl: str
    comparisonImageUrls: list[str]

