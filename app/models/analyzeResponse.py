
class AnalyzeResponse:

    def __init__(self, similarity, pressure, inclination, verificationImageUrl):
        self.similarity = similarity
        self.pressure = pressure
        self.inclination = inclination
        self.verificationImageUrl = verificationImageUrl


    def to_dict(self):
        return {
            "similarity": self.similarity,
            "pressure": self.pressure,
            "inclination": self.inclination,
            "verificationImageUrl": self.verificationImageUrl
        }