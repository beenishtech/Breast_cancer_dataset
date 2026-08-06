from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Breast Cancer Classification API")

# Model load karein
model = joblib.load('model/cancer_model.pkl')

class CancerInput(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: CancerInput):
    if len(data.features) != 30:
        return {"error": f"Exactly 30 features expected, got {len(data.features)}"}
    
    input_array = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(input_array)[0])
    probabilities = model.predict_proba(input_array)[0].tolist()
    
    label = "Malignant (Cancerous)" if prediction == 1 else "Benign (Safe)"
    
    return {
        "prediction_code": prediction,
        "result": label,
        "confidence": {
            "benign_probability": round(probabilities[0], 4),
            "malignant_probability": round(probabilities[1], 4)
        }
    }