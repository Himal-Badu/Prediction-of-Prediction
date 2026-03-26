"""
Feedback Mechanism - Updates base model weights based on error analysis.
"""
import numpy as np
from typing import Dict, Any, Optional, Callable


class FeedbackMechanism:
    """
    Feedback mechanism that analyzes errors and updates base model weights.
    Implements adaptive weight adjustment based on prediction performance.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        adaptation_threshold: float = 0.05,
        min_samples: int = 10,
        weight_decay: float = 0.95
    ):
        """
        Initialize feedback mechanism.
        
        Args:
            learning_rate: Rate at which to adjust weights
            adaptation_threshold: Error threshold to trigger adaptation
            min_samples: Minimum samples before adapting
            weight_decay: Decay factor for older weights
        """
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.min_samples = min_samples
        self.weight_decay = weight_decay
        
        self.error_buffer = []
        self.weight_adjustments = []
        self.performance_history = []
        
    def analyze_error(
        self,
        base_predictions: np.ndarray,
        corrected_predictions: np.ndarray,
        true_values: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze errors between base and corrected predictions.
        
        Returns:
            Dictionary with error analysis
        """
        base_errors = np.abs(true_values - base_predictions)
        corrected_errors = np.abs(true_values - corrected_predictions)
        
        improvement = base_errors - corrected_errors
        improvement_ratio = np.mean(improvement > 0)
        
        return {
            "base_error_mean": float(np.mean(base_errors)),
            "corrected_error_mean": float(np.mean(corrected_errors)),
            "improvement_mean": float(np.mean(improvement)),
            "improvement_ratio": float(improvement_ratio),
            "error_reduction_percent": float(
                (np.mean(base_errors) - np.mean(corrected_errors)) / 
                (np.mean(base_errors) + 1e-10) * 100
            )
        }
    
    def compute_weight_adjustment(
        self,
        errors: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute weight adjustments based on error patterns.
        
        Args:
            errors: Prediction errors
            features: Optional feature matrix for feature-specific adjustments
            
        Returns:
            Dictionary with weight adjustment info
        """
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Determine adjustment direction
        if abs(mean_error) < self.adaptation_threshold:
            adjustment_magnitude = 0.0
            status = "stable"
        elif mean_error > 0:
            adjustment_magnitude = -self.learning_rate * (mean_error / (std_error + 1e-10))
            status = "underestimating"
        else:
            adjustment_magnitude = self.learning_rate * (abs(mean_error) / (std_error + 1e-10))
            status = "overestimating"
        
        # Feature-specific adjustments if features provided
        feature_adjustments = {}
        if features is not None and len(features.shape) > 1:
            n_features = features.shape[1]
            for i in range(min(n_features, 10)):  # Limit to first 10 features
                feature_corr = np.corrcoef(features[:, i], errors)[0, 1]
                if not np.isnan(feature_corr):
                    feature_adjustments[f"feature_{i}"] = float(feature_corr)
        
        return {
            "adjustment_magnitude": float(adjustment_magnitude),
            "status": status,
            "mean_error": float(mean_error),
            "std_error": float(std_error),
            "feature_adjustments": feature_adjustments,
            "should_adapt": abs(mean_error) >= self.adaptation_threshold and len(errors) >= self.min_samples
        }
    
    def update_base_model(
        self,
        base_model: Any,
        X: np.ndarray,
        y: np.ndarray,
        weight_adjustment: float = 1.0
    ) -> Any:
        """
        Update the base model with adjusted weights.
        
        Args:
            base_model: The base model to update
            X: Training features
            y: Training targets
            weight_adjustment: Additional weight adjustment factor
            
        Returns:
            Updated base model
        """
        # Store current performance
        current_pred = base_model.predict(X)
        current_mse = np.mean((y - current_pred) ** 2)
        self.performance_history.append(current_mse)
        
        # Retrain with current data
        base_model.fit(X, y)
        
        # Store weight adjustment info
        self.weight_adjustments.append({
            "adjustment": weight_adjustment,
            "before_mse": current_mse,
            "timestamp": len(self.performance_history)
        })
        
        return base_model
    
    def feedback_loop(
        self,
        base_model: Any,
        pop_layer: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Complete feedback loop: analyze errors, compute adjustments, update models.
        
        Args:
            base_model: The base model
            pop_layer: The PoP layer
            X: Input features
            y: True values
            
        Returns:
            Dictionary with feedback results
        """
        # Get base predictions
        base_predictions = base_model.predict(X)
        
        # Get PoP corrected predictions
        pop_result = pop_layer.predict(X, base_predictions)
        corrected_predictions = pop_result["corrected_predictions"]
        
        # Analyze errors
        error_analysis = self.analyze_error(
            base_predictions, corrected_predictions, y
        )
        
        # Compute weight adjustment
        errors = y - base_predictions
        adjustment_info = self.compute_weight_adjustment(errors, X)
        
        # Update if needed
        updated_model = base_model
        if adjustment_info["should_adapt"]:
            updated_model = self.update_base_model(
                base_model, X, y, adjustment_info["adjustment_magnitude"]
            )
        
        # Store in buffer
        self.error_buffer.extend(errors.tolist())
        
        return {
            "error_analysis": error_analysis,
            "adjustment_info": adjustment_info,
            "model_updated": adjustment_info["should_adapt"],
            "performance_history_length": len(self.performance_history)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of feedback performance."""
        if not self.performance_history:
            return {"status": "No history available"}
        
        return {
            "total_updates": len(self.performance_history),
            "latest_mse": float(self.performance_history[-1]),
            "best_mse": float(min(self.performance_history)),
            "improvement_trend": (
                "improving" if len(self.performance_history) > 1 and 
                self.performance_history[-1] < self.performance_history[0]
                else "stable"
            )
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get mechanism parameters."""
        return {
            "learning_rate": self.learning_rate,
            "adaptation_threshold": self.adaptation_threshold,
            "min_samples": self.min_samples,
            "weight_decay": self.weight_decay,
            "buffer_size": len(self.error_buffer),
            "total_adjustments": len(self.weight_adjustments)
        }