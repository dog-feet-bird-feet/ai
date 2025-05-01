from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    id: int
    reference_urls: list[str]
    test_url: str
