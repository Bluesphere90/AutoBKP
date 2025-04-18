from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import numpy as np
import logging

from app.ml.inference.predict import ModelPredictor
from app.core.logging import Logger

# Khởi tạo router
router = APIRouter()
logger = Logger(__name__)


# Định nghĩa request và response models
class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: Dict[str, float]


class TrainingRequest(BaseModel):
    data_path: str
    model_params: Optional[Dict[str, Any]] = None


class TrainingResponse(BaseModel):
    model_id: str
    metrics: Dict[str, float]
    status: str


# Tạo instance của predictor
predictor = ModelPredictor()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """API endpoint để dự đoán class từ feature vector."""
    try:
        # Chuyển đổi features thành numpy array
        features = np.array(request.features).reshape(1, -1)

        # Thực hiện dự đoán
        class_id, class_name, probabilities = predictor.predict(features)

        return PredictionResponse(
            class_id=class_id,
            class_name=class_name,
            probabilities={str(k): float(v) for k, v in probabilities.items()}
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """API endpoint để train hoặc retrain models."""
    try:
        from app.ml.models.train import ModelTrainer

        trainer = ModelTrainer()
        model_id, metrics = trainer.train(
            data_path=request.data_path,
            model_params=request.model_params
        )

        return TrainingResponse(
            model_id=model_id,
            metrics={k: float(v) for k, v in metrics.items()},
            status="success"
        )
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


@router.get("/models", response_model=List[str])
async def list_models():
    """API endpoint để liệt kê các mô hình có sẵn."""
    try:
        from app.ml.models.model import get_available_models

        models = get_available_models()
        return models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@router.get("/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "ok"}