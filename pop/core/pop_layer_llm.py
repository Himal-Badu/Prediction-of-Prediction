"""
PoP Layer for LLM - Meta-learning layer for language models.
Predicts when the LLM is likely wrong based on its output distributions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMErrorPredictor(nn.Module):
    """
    Neural network that predicts LLM errors from logits/probabilities.
    Takes LLM output (logits/probabilities) as input and predicts:
    - Error magnitude (how wrong the prediction might be)
    - Error direction (over/under confident)
    - Confidence score (how sure the LLM is)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize the PoP layer for LLM.
        
        Args:
            vocab_size: Size of LLM vocabulary
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Input: top-k logits + probability distribution stats
        # We'll dynamically compute features from full distribution
        input_dim = 16  # Features extracted from distribution
        
        # Normalize features to same scale
        self.feature_norm = nn.LayerNorm(input_dim)
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.hidden = nn.Sequential(*layers)
        
        # Output heads
        self.error_head = nn.Linear(hidden_dim, 1)  # Error magnitude
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Confidence score
        self.direction_head = nn.Linear(hidden_dim, 1)  # Error direction
        
    def extract_features(self, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from logits/probability distribution.
        
        Features include:
        - Entropy of distribution
        - Top-k probability mass
        - Probability concentration
        - Logit range
        - Prediction confidence
        """
        # Ensure 2D: (batch, vocab)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        features = []
        
        for i in range(logits.shape[0]):
            logit_slice = logits[i]
            prob_slice = probs[i]
            
            feat = []
            
            # Entropy (uncertainty measure)
            entropy = -torch.sum(prob_slice * torch.log(prob_slice + 1e-10), dim=-1)
            feat.append(entropy.unsqueeze(-1))
            
            # Top-1 probability
            top1_prob, _ = torch.max(prob_slice, dim=-1)
            feat.append(top1_prob.unsqueeze(-1))
            
            # Top-3 probability mass
            top3_probs, _ = torch.topk(prob_slice, 3, dim=-1)
            top3_mass = torch.sum(top3_probs, dim=-1)
            feat.append(top3_mass.unsqueeze(-1))
            
            # Top-10 probability mass
            top10_probs, _ = torch.topk(prob_slice, 10, dim=-1)
            top10_mass = torch.sum(top10_probs, dim=-1)
            feat.append(top10_mass.unsqueeze(-1))
            
            # Logit range (max - min)
            logit_range = torch.max(logit_slice) - torch.min(logit_slice)
            feat.append(logit_range.unsqueeze(-1))
            
            # Logit mean
            logit_mean = torch.mean(logit_slice)
            feat.append(logit_mean.unsqueeze(-1))
            
            # Logit std
            logit_std = torch.std(logit_slice)
            feat.append(logit_std.unsqueeze(-1))
            
            # Number of tokens with prob > 0.01
            n_active = torch.sum(prob_slice > 0.01, dim=-1).float()
            feat.append(n_active.unsqueeze(-1))
            
            # Number of tokens with prob > 0.1
            n_confident = torch.sum(prob_slice > 0.1, dim=-1).float()
            feat.append(n_confident.unsqueeze(-1))
            
            # Probability percentiles
            sorted_probs, _ = torch.sort(prob_slice)
            p25 = sorted_probs[int(0.25 * len(sorted_probs))]
            p50 = sorted_probs[int(0.5 * len(sorted_probs))]
            p75 = sorted_probs[int(0.75 * len(sorted_probs))]
            feat.append(p25.unsqueeze(-1))
            feat.append(p50.unsqueeze(-1))
            feat.append(p75.unsqueeze(-1))
            
            # Normalized probability variance
            prob_var = torch.var(prob_slice)
            feat.append(prob_var.unsqueeze(-1))
            
            # Gini coefficient (inequality of distribution)
            sorted_desc, _ = torch.sort(prob_slice, dim=-1, descending=True)
            n = prob_slice.shape[-1]
            cumsum = torch.cumsum(sorted_desc, dim=-1)
            gini = (2 * torch.sum((torch.arange(1, n + 1, device=logits.device).float() * sorted_desc)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)
            feat.append(gini.unsqueeze(-1))
            
            # Max/min ratio (log-scaled to prevent explosion)
            max_min_ratio = torch.log(top1_prob + 1e-10) - torch.log(torch.min(prob_slice) + 1e-10)
            feat.append(max_min_ratio.unsqueeze(-1))
            
            # Log-sum-exp (partition function)
            log_sum_exp = torch.logsumexp(logit_slice, dim=-1)
            feat.append(log_sum_exp.unsqueeze(-1))
            
            features.append(torch.cat(feat, dim=-1))
        
        return torch.stack(features)
    
    def forward(self, logits: torch.Tensor, probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            logits: Raw logits from LLM
            probs: Probability distribution from LLM
            
        Returns:
            Dictionary with predictions
        """
        features = self.extract_features(logits, probs)
        features = self.feature_norm(features)
        hidden = self.hidden(features)
        
        error_magnitude = torch.sigmoid(self.error_head(hidden)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1)
        error_direction = torch.tanh(self.direction_head(hidden)).squeeze(-1)
        
        return {
            "error_magnitude": error_magnitude,
            "confidence": confidence,
            "error_direction": error_direction,
            "features": features
        }


@dataclass
class TrainingExample:
    """Training example for PoP layer."""
    logits: torch.Tensor       # Raw LLM logits (vocab_size,)
    probs: torch.Tensor        # LLM probability distribution (vocab_size,)
    error_magnitude: float  # 0-1: how wrong the LLM was
    confidence: float  # 0-1: how confident the LLM actually was
    error_direction: float  # -1 to 1: over/under estimate


class PoPLayerLLM:
    """
    PoP Layer for LLM - meta-learning system that predicts LLM errors.
    """
    
    def __init__(
        self,
        vocab_size: int,
        device: Optional[str] = None,
        hidden_dim: int = 256,
        learning_rate: float = 0.001
    ):
        """
        Initialize the PoP layer for LLM.
        
        Args:
            vocab_size: Size of the LLM vocabulary
            device: Device to use ('cpu', 'cuda')
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for training
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        
        # Create neural network
        self.model = LLMErrorPredictor(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.is_trained = False
        self.training_history = []
        
    def _compute_error_label(
        self,
        predicted_token: int,
        correct_token: int,
        predicted_prob: float,
        correct_prob: float
    ) -> Tuple[float, float, float]:
        """
        Compute error labels from prediction results.
        
        Returns:
            Tuple of (error_magnitude, confidence, error_direction)
        """
        # Error magnitude: 1 if wrong, 0 if correct
        error_magnitude = 1.0 if predicted_token != correct_token else 0.0
        
        # Confidence: the probability the model assigned to its prediction
        confidence = predicted_prob
        
        # Error direction: positive = overconfident, negative = underconfident
        if predicted_token != correct_token:
            # If wrong, was overconfident (predicted high prob but wrong)
            error_direction = predicted_prob - correct_prob
        else:
            # If correct, direction is less relevant
            error_direction = 0.0
        
        return error_magnitude, confidence, error_direction
    
    def train_step(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        error_magnitude: float,
        confidence: float,
        error_direction: float
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            logits: LLM logits
            probs: LLM probabilities
            error_magnitude: Target error magnitude
            confidence: Target confidence
            error_direction: Target error direction
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(logits, probs)
        
        # Compute loss
        loss = (
            self.criterion(outputs["error_magnitude"], torch.tensor([error_magnitude], device=self.device)) +
            self.criterion(outputs["confidence"], torch.tensor([confidence], device=self.device)) +
            self.criterion(outputs["error_direction"], torch.tensor([error_direction], device=self.device))
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.is_trained = True
        
        return {"loss": loss.item()}
    
    def train_on_examples(
        self,
        examples: List[TrainingExample],
        epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Train on a list of training examples.
        
        Args:
            examples: List of training examples
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if not examples:
            return {"status": "No examples provided"}
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for ex in examples:
                self.optimizer.zero_grad()
                
                outputs = self.model(ex.logits.unsqueeze(0), ex.probs.unsqueeze(0))
                loss = (
                    self.criterion(outputs["error_magnitude"], torch.tensor([ex.error_magnitude], device=self.device)) +
                    self.criterion(outputs["confidence"], torch.tensor([ex.confidence], device=self.device)) +
                    self.criterion(outputs["error_direction"], torch.tensor([ex.error_direction], device=self.device))
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(examples)
            self.training_history.append({"epoch": epoch + 1, "loss": avg_loss})
            
        self.is_trained = True
        return {"status": "Training complete", "history": self.training_history}
    
    def predict(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Predict error likelihood for LLM output.
        
        Args:
            logits: LLM logits
            probs: LLM probabilities
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.is_trained:
            logger.warning("PoP layer not trained yet, using untrained predictions")
        
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(logits, probs)
            
        error_magnitude = outputs["error_magnitude"].item()
        confidence = outputs["confidence"].item()
        error_direction = outputs["error_direction"].item()
        
        # Determine if we should correct
        # Rule: IF (PoP confident) AND (PoP adjustment NOT worse than LLM): Apply correction
        should_correct = (
            confidence > 0.7 and  # PoP confident
            error_magnitude > 0.3  # LLM might be wrong
        )
        
        return {
            "error_magnitude": error_magnitude,
            "confidence": confidence,
            "error_direction": error_direction,
            "should_correct": should_correct,
            "llm_likely_wrong": error_magnitude > 0.5,
            "llm_overconfident": error_direction > 0.3,
            "llm_underconfident": error_direction < -0.3
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "device": self.device,
            "vocab_size": self.vocab_size,
            "is_trained": self.is_trained,
            "training_history_length": len(self.training_history)
        }


def create_pop_llm(vocab_size: int, device: Optional[str] = None) -> PoPLayerLLM:
    """Factory function to create a PoP layer for LLM."""
    return PoPLayerLLM(vocab_size=vocab_size, device=device)