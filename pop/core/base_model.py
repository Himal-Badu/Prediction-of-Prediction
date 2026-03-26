"""
Base Model Layer - The foundation ML model that PoP monitors and improves.
"""
import numpy as np
from typing import Optional, Dict, Any
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class BaseModel:
    """
    Base model that makes predictions. This is the model being monitored
    and improved by the PoP system.
    """
    
    def __init__(
        self,
        model_type: str = "ridge",
        **model_kwargs
    ):
        """
        Initialize the base model.
        
        Args:
            model_type: Type of model ('ridge', 'linear', 'rf', 'gbm')
            **model_kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.model = self._create_model()
        self.is_fitted = False
        
    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == "ridge":
            return Ridge(**self.model_kwargs)
        elif self.model_type == "linear":
            return LinearRegression(**self.model_kwargs)
        elif self.model_type == "rf":
            return RandomForestRegressor(**self.model_kwargs)
        elif self.model_type == "gbm":
            return GradientBoostingRegressor(**self.model_kwargs)
        else:
            return Ridge()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the base model to training data."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the base model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions with confidence estimates.
        
        Returns:
            Dictionary with predictions and confidence metrics
        """
        predictions = self.predict(X)
        
        # Simple confidence based on prediction variance
        # In production, could use conformal prediction or bootstrap
        pred_std = np.std(predictions)
        confidence = 1.0 / (1.0 + pred_std)
        
        return {
            "predictions": predictions,
            "confidence": confidence,
            "model_type": self.model_type
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_)
        return None
    
    def update_weights(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Update model weights based on new data or feedback.
        This allows the base model to adapt based on PoP feedback.
        """
        self.fit(X, y)
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "model_type": self.model_type,
            "is_fitted": self.is_fitted,
            "params": self.model.get_params() if hasattr(self.model, "get_params") else {}
        }