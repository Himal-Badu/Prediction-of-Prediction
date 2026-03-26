"""
FastAPI application for PoP prediction service.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
import os

from pop.core.base_model import BaseModel
from pop.core.pop_layer import PoPLayer
from pop.core.feedback import FeedbackMechanism

app = FastAPI(
    title="Prediction-of-Prediction API",
    description="Meta-learning engine for error prediction and model improvement",
    version="0.1.0"
)

# Global state (in production, use proper state management)
_models = {}
_feedback = {}


class PredictionRequest(BaseModel):
    features: List[List[float]]
    predictions: List[float]
    model_id: Optional[str] = "default"


class PredictionResponse(BaseModel):
    corrected_predictions: List[float]
    predicted_errors: List[float]
    confidence: List[float]
    low_confidence_indices: List[int]


class TrainingRequest(BaseModel):
    features: List[List[float]]
    predictions: List[float]
    true_values: List[float]
    base_model_type: Optional[str] = "ridge"
    pop_model_type: Optional[str] = "ridge"
    model_id: Optional[str] = "default"


class FeedbackRequest(BaseModel):
    features: List[List[float]]
    true_values: List[float]
    model_id: Optional[str] = "default"


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "Prediction-of-Prediction (PoP)",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": len(_models)}


@app.post("/train")
def train_models(request: TrainingRequest):
    """
    Train the base model and PoP layer.
    """
    try:
        X = np.array(request.features)
        base_preds = np.array(request.predictions)
        y = np.array(request.true_values)
        
        # Create and train base model
        base_model = BaseModel(model_type=request.base_model_type)
        base_model.fit(X, y)
        
        # Create and train PoP layer
        pop_layer = PoPLayer(error_model_type=request.pop_model_type)
        pop_layer.fit(X, base_preds, y)
        
        # Create feedback mechanism
        feedback = FeedbackMechanism()
        
        # Store models
        _models[request.model_id] = {
            "base_model": base_model,
            "pop_layer": pop_layer,
            "feedback": feedback
        }
        
        return {
            "status": "trained",
            "model_id": request.model_id,
            "base_model_type": request.base_model_type,
            "pop_model_type": request.pop_model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Get corrected predictions from PoP layer.
    """
    if request.model_id not in _models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        X = np.array(request.features)
        base_preds = np.array(request.predictions)
        
        models = _models[request.model_id]
        pop_layer = models["pop_layer"]
        
        result = pop_layer.predict(X, base_preds)
        
        return PredictionResponse(
            corrected_predictions=result["corrected_predictions"].tolist(),
            