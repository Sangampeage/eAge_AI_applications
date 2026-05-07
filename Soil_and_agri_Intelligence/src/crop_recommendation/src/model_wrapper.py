import numpy as np
from schemas import CropPrediction

class CropModel:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, features):
        probs = self.model.predict_proba([features])[0]

        top3_idx = np.argsort(probs)[-3:][::-1]
        crops = self.label_encoder.inverse_transform(top3_idx)

        return [
            CropPrediction(crop=crop, confidence=float(probs[i]))
            for crop, i in zip(crops, top3_idx)
        ]