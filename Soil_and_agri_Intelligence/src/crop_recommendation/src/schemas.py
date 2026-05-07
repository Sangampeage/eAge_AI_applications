from dataclasses import dataclass

@dataclass
class SoilInput:
    soil: str
    N: float
    P: float
    K: float
    ph: float
    temperature: float
    moisture: float
    ec: float

@dataclass
class CropPrediction:
    crop: str
    confidence: float