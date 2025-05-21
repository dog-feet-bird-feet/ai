from pydantic import BaseModel

class PersonalityRequest(BaseModel):
    imageUrl: str
