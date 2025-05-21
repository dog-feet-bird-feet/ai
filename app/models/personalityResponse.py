from pydantic import BaseModel
from typing import Optional

class Trait(BaseModel):
    score: str
    detail: str

class Traits(BaseModel):
    size: Trait
    pressure: Trait
    inclination: Trait
    shape: Trait

class PersonalityResponse(BaseModel):
    traits: Traits
    type: str
    description: str