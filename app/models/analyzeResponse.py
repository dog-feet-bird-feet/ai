
class AnalyzeResponse:

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