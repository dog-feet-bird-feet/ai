from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class InputData(BaseModel):
    id: int

class OutputData:

    def __init__(self, id, similarity, pressure, inclination, verificationImgUrl):
        self.id = id
        self.similarity = similarity
        self.pressure = pressure
        self.inclination = inclination
        self.verificationImgUrl = verificationImgUrl


    def to_dict(self):
        return {
            "id": self.id,
            "similarity": self.similarity,
            "pressure": self.pressure,
            "inclination": self.inclination,
            "verificationImgUrl": self.verificationImgUrl
        }
    
@app.post("/generate")
def generate(inputData: InputData):
    # Create OutputData using the received id from InputData
    appraisal_response = OutputData(inputData.id, 1.0, 23.1, 222.4, "https://s3.aws.com/verifi-url?q=AgfDs3dF5Fgas")

    return appraisal_response
