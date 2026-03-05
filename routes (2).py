
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

# Assuming model.py is in the same directory or accessible via PYTHONPATH
from model import ScamDetectorModel

router = APIRouter()

# Initialize the model globally to avoid reloading on each request
scam_detector = ScamDetectorModel()

class TextInput(BaseModel):
    text: str

@router.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}

@router.post("/predict", response_model=Dict[str, Any])
async def predict_scam_endpoint(input: TextInput):
    prediction_result = scam_detector.predict_scam(input.text)
    return prediction_result
