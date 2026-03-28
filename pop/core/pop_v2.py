"""
PoP Layer v2 — Improved meta-learning layer for language models.

Key improvements over v1:
- BCEWithLogitsLoss for binary heads (error_magnitude, confidence)
- Batch normalization for training stability
- Residual blocks for deeper capacity without gradient degradation
- Fully vectorized feature extraction (no Python loops)
- Expanded feature set (24 features, more discriminative)
- Proper batched training interface with DataLoader support
- Learning rate scheduling and gradient clipping
- Model serialization (save/load checkpoints)
- Backward-compatible API with existing integration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import logging
import math
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Extraction (Fully Vectorized)
# ---------------------------------------------------------------------------

def extract_features_vectorized(
    logits: torch.Tensor,
    probs: torch.Tensor,
    num_features: int = 24,
) -> torch.Tensor:
    """
    Extract distributional features from LLM output.
    
    Fully vectorized — no Python loops over batch dimension.
    Produces 24 features per sample.
    
    Args:
        logits: Raw logits, shape (B, V) or (V,)
        probs: Probability distribution, shape (B, V) or (V,)
        num_features: Expected number of features (for validation)
    
    Returns:
        Feature tensor of shape (B, num_features)
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    
    B, V = logits.shape
    eps = 1e-10
    
    # Clamp probabilities for numerical stability
    probs = probs.clamp(min=eps)
    
    # Sort probabilities once for percentile computation
    sorted_probs, _ = torch.sort(probs, dim=-1)
    
    # ── Core uncertainty features ──────────────────────────────────────
    
    # 1. Shannon entropy (bits of uncertainty)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)  # (B,)
    
    # 2. Normalized entropy (0-1 scale, independent of vocab size)
    norm_entropy = entropy / math.log(V)
    
    # 3. Perplexity (= exp(entropy), standard NLP metric)
    perplexity = torch.exp(entropy)
    
    # ── Confidence features ────────────────────────────────────────────
    
    # Get top-k values
    topk_vals, _ = torch.topk(probs, k=min(100, V), dim=-1)
    
    # 4. Top-1 probability (model's confidence in its best guess)
    top1 = topk_vals[:, 0]
    
    # 5. Margin (top-1 minus top-2 — how "decided" the model is)
    margin = top1 - topk_vals[:, 1]
    
    # 6. Top-3 probability mass
    top3_mass = topk_vals[:, :min(3, V)].sum(dim=-1)
    
    # 7. Top-10 probability mass
    top10_mass = topk_vals[:, :min(10, V)].sum(dim=-1)
    
    # 8. Top-50 probability mass
    top50_mass = topk_vals[:, :min(50, V)].sum(dim=-1)
    
    # 9. Head-to-tail ratio (sharpness of distribution head)
    head_tail_ratio = top3_mass / (top10_mass + eps)
    
    # ── Logit statistics ──────────────────────────────────────────────
    
    # 10. Logit range (max - min)
    logit_range = logits.max(dim=-1).values - logits.min(dim=-1).values
    
    # 11. Logit std (spread of raw scores)
    logit_std = logits.std(dim=-1)
    
    # 12. Normalized logit spread (std / |mean|, scale-invariant)
    logit_mean = logits.mean(dim=-1)
    logit_norm = logit_std / (torch.abs(logit_mean) + eps)
    
    # 13. Logit skewness (asymmetry of logit distribution)
    logit_centered = logits - logit_mean.unsqueeze(-1)
    logit_var = (logit_centered ** 2).mean(dim=-1)
    logit_skew = ((logit_centered ** 3).mean(dim=-1)) / (logit_var ** 1.5 + eps)
    
    # ── Distribution shape features ───────────────────────────────────
    
    # Percentile computation
    def percentile(sorted_vals, p):
        idx = int(p * V)
        return sorted_vals[:, min(idx, V - 1)]
    
    # 14. 25th percentile
    p25 = percentile(sorted_probs, 0.25)
    
    # 15. 50th percentile (median)
    p50 = percentile(sorted_probs, 0.50)
    
    # 16. 75th percentile
    p75 = percentile(sorted_probs, 0.75)
    
    # 17. Inter-quartile range (robust spread measure)
    iqr = p75 - p25
    
    # 18. Effective vocabulary size (inverse Simpson concentration)
    eff_vocab = 1.0 / (probs ** 2).sum(dim=-1)
    
    # 19. Probability variance
    prob_var = probs.var(dim=-1)
    
    # 20. Negative log-prob of top-1 (cross-entropy of best prediction)
    nll_top1 = -torch.log(top1 + eps)
    
    # 21. Ratio of top-1 to mean prob (how much better than uniform)
    mean_prob = 1.0 / V
    top1_vs_mean = top1 / (mean_prob + eps)
    
    # 22. Gini coefficient (inequality measure)
    sorted_desc, _ = torch.sort(probs, dim=-1, descending=True)
    indices = torch.arange(1, V + 1, device=probs.device, dtype=torch.float)
    cumsum = torch.cumsum(sorted_desc, dim=-1)
    weighted_sum = (indices.unsqueeze(0) * sorted_desc).sum(dim=-1)
    gini = (2 * weighted_sum - (V + 1) * cumsum[:, -1]) / (V * cumsum[:, -1] + eps)
    
    # 23. Count of tokens with prob > 0.01
    n_active = (probs > 0.01).float().sum(dim=-1)
    
    # 24. Ratio of top-5 to top-50 (concentration in head vs broader body)
    top5_mass = topk_vals[:, :min(5, V)].sum(dim=-1)
    concentration_ratio = top5_mass / (top50_mass + eps)
    
    features = torch.stack([
        entropy,          # 1
        norm_entropy,     # 2
        perplexity,       # 3
        top1,             # 4
        margin,           # 5
        top3_mass,        # 6
        top10_mass,       # 7
        top50_mass,       # 8
        head_tail_ratio,  # 9
        logit_range,      # 10
        logit_std,        # 11
        logit_norm,       # 12
        logit_skew,       # 13
        p25,              # 14
        p50,              # 15
        p75,              # 16
        iqr,              # 17
        eff_vocab,        # 18
        prob_var,         # 19
        nll_top1,         # 20
        top1_vs_mean,     # 21
        gini,             # 22
        n_active,         # 23
        concentration_ratio,  # 24
    ], dim=-1)
    
    assert features.shape[-1] == num_features, (
        f"Expected {num_features} features, got {features.shape[-1]}"
    )
    
    return features


# ---------------------------------------------------------------------------
# Network Components
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-norm residual block with batch normalization."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.norm2(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return F.relu(h + residual)


class PoPHead(nn.Module):
    """Output head with optional intermediate layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# PoP v2 Network
# ---------------------------------------------------------------------------

class LLMErrorPredictorV2(nn.Module):
    """
    Improved meta-learning network for LLM error prediction.
    
    Architecture:
        Input: 24 distributional features
        → Feature normalization (LayerNorm)
        → Projection to hidden dim (Linear + BatchNorm + ReLU)
        → N residual blocks (pre-norm, with batch norm)
        → 3 output heads (error_magnitude, confidence, error_direction)
    
    Key differences from v1:
        - Residual blocks instead of plain Linear stacks
        - Batch normalization for training stability
        - 24 features instead of 16 (more discriminative)
        - Raw logits output (BCEWithLogitsLoss compatible)
        - Wider hidden dimension (512 default)
    """
    
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 24,
        hidden_dim: int = 512,
        num_residual_blocks: int = 3,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Projection to hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Residual backbone
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_residual_blocks)
        ])
        
        # Output heads (raw logits — apply sigmoid/tanh at inference)
        self.error_head = PoPHead(hidden_dim, head_hidden_dim)       # → sigmoid for error magnitude
        self.confidence_head = PoPHead(hidden_dim, head_hidden_dim)  # → sigmoid for confidence
        self.direction_head = PoPHead(hidden_dim, head_hidden_dim)   # → tanh for direction
    
    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            logits: Raw LLM logits, shape (B, V) or (V,)
            probs: LLM probability distribution, shape (B, V) or (V,)
        
        Returns:
            Dict with:
                - error_magnitude: sigmoid-squashed, shape (B,)
                - confidence: sigmoid-squashed, shape (B,)
                - error_direction: tanh-squashed, shape (B,)
                - error_logits: raw logits for loss computation, shape (B,)
                - confidence_logits: raw logits for loss computation, shape (B,)
                - direction_raw: raw value for loss computation, shape (B,)
                - features: extracted features, shape (B, 24)
        """
        # Extract features
        features = extract_features_vectorized(logits, probs, self.input_dim)
        
        # Normalize and project
        x = self.input_norm(features)
        x = self.projection(x)
        
        # Residual backbone
        x = self.residual_blocks(x)
        
        # Raw head outputs
        error_logits = self.error_head(x)
        confidence_logits = self.confidence_head(x)
        direction_raw = self.direction_head(x)
        
        return {
            # Activated outputs (for inference)
            "error_magnitude": torch.sigmoid(error_logits),
            "confidence": torch.sigmoid(confidence_logits),
            "error_direction": torch.tanh(direction_raw),
            # Raw logits (for training loss)
            "error_logits": error_logits,
            "confidence_logits": confidence_logits,
            "direction_raw": direction_raw,
            # Extras
            "features": features,
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class PoPLossV2(nn.Module):
    """
    Combined loss for PoP v2 multi-head output.
    
    - error_magnitude: BCEWithLogitsLoss (binary classification)
    - confidence: BCEWithLogitsLoss (bounded [0,1] prediction)
    - error_direction: SmoothL1Loss (regression, robust to outliers)
    """
    
    def __init__(self, error_weight: float = 1.0, conf_weight: float = 1.0, dir_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.huber = nn.SmoothL1Loss()
        self.error_weight = error_weight
        self.conf_weight = conf_weight
        self.dir_weight = dir_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_error = self.bce(predictions["error_logits"], targets["error_magnitude"])
        loss_conf = self.bce(predictions["confidence_logits"], targets["confidence"])
        loss_dir = self.huber(predictions["direction_raw"], targets["error_direction"])
        
        total = (
            self.error_weight * loss_error +
            self.conf_weight * loss_conf +
            self.dir_weight * loss_dir
        )
        
        return {
            "total": total,
            "error": loss_error,
            "confidence": loss_conf,
            "direction": loss_dir,
        }


# ---------------------------------------------------------------------------
# Training Wrapper (Backward-Compatible API)
# ---------------------------------------------------------------------------

@dataclass
class TrainingExampleV2:
    """Training example with raw logits/probs and labels."""
    logits: torch.Tensor          # (V,) raw LLM logits
    probs: torch.Tensor           # (V,) LLM probability distribution
    error_magnitude: float        # 0.0 or 1.0
    confidence: float             # 0.0–1.0
    error_direction: float        # -1.0–1.0


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"   # "cosine", "step", or "none"
    validation_split: float = 0.1
    log_every: int = 5
    checkpoint_dir: Optional[str] = None


class PoPLayerLLMV2:
    """
    PoP Layer v2 for LLM — improved meta-learning system.
    
    Drop-in replacement for PoPLayerLLM with the same core API:
        - predict(logits, probs) → dict
        - train_on_examples(examples) → history
        - get_params() → dict
    
    New capabilities:
        - train_batched(examples, config) → full training loop
        - save(path) / load(path) — checkpoint management
        - Proper validation and LR scheduling
    
    Example usage (backward-compatible):
        pop = PoPLayerLLMV2(vocab_size=50000)
        result = pop.predict(logits, probs)
        print(result["llm_likely_wrong"])
    
    Example usage (new batched training):
        config = TrainingConfig(epochs=50, batch_size=64)
        history = pop.train_batched(examples, config)
        pop.save("pop_checkpoint.pt")
    """
    
    def __init__(
        self,
        vocab_size: int,
        device: Optional[str] = None,
        hidden_dim: int = 512,
        num_residual_blocks: int = 3,
        learning_rate: float = 1e-3,
        input_dim: int = 24,
    ):
        """
        Initialize PoP Layer v2.
        
        Args:
            vocab_size: Size of the LLM vocabulary
            device: Device to use ('cpu', 'cuda', or None for auto)
            hidden_dim: Hidden layer dimension (default 512, up from 256)
            num_residual_blocks: Number of residual blocks (default 3)
            learning_rate: Initial learning rate
            input_dim: Number of input features (default 24)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        # Create network
        self.model = LLMErrorPredictorV2(
            vocab_size=vocab_size,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_residual_blocks=num_residual_blocks,
        ).to(self.device)
        
        logger.info(
            f"PoP v2 initialized: {self.model.count_parameters():,} parameters "
            f"on {self.device}"
        )
        
        self.criterion = PoPLossV2()
        self.is_trained = False
        self.training_history: List[Dict[str, Any]] = []
    
    # ── Inference ──────────────────────────────────────────────────────
    
    def predict(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Predict error likelihood for LLM output.
        
        Args:
            logits: LLM logits, shape (V,) or (B, V)
            probs: LLM probabilities, shape (V,) or (B, V)
        
        Returns:
            Dictionary with predictions (single-sample) or batch predictions.
        """
        if not self.is_trained:
            logger.warning("PoP v2 not trained yet — using untrained predictions")
        
        self.model.eval()
        
        # Move to device
        logits = logits.to(self.device)
        probs = probs.to(self.device)
        
        single_sample = logits.dim() == 1
        
        with torch.no_grad():
            outputs = self.model(logits, probs)
        
        if single_sample:
            error_mag = outputs["error_magnitude"].item()
            conf = outputs["confidence"].item()
            direction = outputs["error_direction"].item()
            
            should_correct = conf > 0.7 and error_mag > 0.3
            
            return {
                "error_magnitude": error_mag,
                "confidence": conf,
                "error_direction": direction,
                "should_correct": should_correct,
                "llm_likely_wrong": error_mag > 0.5,
                "llm_overconfident": direction > 0.3,
                "llm_underconfident": direction < -0.3,
            }
        else:
            return {
                "error_magnitude": outputs["error_magnitude"].cpu(),
                "confidence": outputs["confidence"].cpu(),
                "error_direction": outputs["error_direction"].cpu(),
                "should_correct": (outputs["confidence"] > 0.7) & (outputs["error_magnitude"] > 0.3),
                "llm_likely_wrong": outputs["error_magnitude"] > 0.5,
                "llm_overconfident": outputs["error_direction"] > 0.3,
                "llm_underconfident": outputs["error_direction"] < -0.3,
            }
    
    # ── Training (Backward-Compatible) ────────────────────────────────
    
    def train_step(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        error_magnitude: float,
        confidence: float,
        error_direction: float,
    ) -> Dict[str, float]:
        """
        Single training step (backward-compatible with v1 API).
        
        Args:
            logits: LLM logits (V,)
            probs: LLM probabilities (V,)
            error_magnitude: Target 0.0 or 1.0
            confidence: Target 0.0–1.0
            error_direction: Target -1.0–1.0
        
        Returns:
            Dict with loss values
        """
        self.model.train()
        
        logits = logits.to(self.device)
        probs = probs.to(self.device)
        
        # Create optimizer on-the-fly for single steps (like v1)
        if not hasattr(self, '_step_optimizer'):
            self._step_optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        
        self._step_optimizer.zero_grad()
        
        outputs = self.model(logits, probs)
        
        targets = {
            "error_magnitude": torch.tensor([error_magnitude], device=self.device),
            "confidence": torch.tensor([confidence], device=self.device),
            "error_direction": torch.tensor([error_direction], device=self.device),
        }
        
        losses = self.criterion(outputs, targets)
        losses["total"].backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self._step_optimizer.step()
        
        self.is_trained = True
        
        return {k: v.item() for k, v in losses.items()}
    
    def train_on_examples(
        self,
        examples: List[TrainingExampleV2],
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """
        Train on examples (improved v1 API).
        
        Now properly uses raw logits/probs per example instead of
        the broken v1 approach.
        
        Args:
            examples: List of TrainingExampleV2
            epochs: Number of epochs
        
        Returns:
            Training history
        """
        if not examples:
            return {"status": "No examples provided"}
        
        config = TrainingConfig(epochs=epochs, batch_size=min(32, len(examples)))
        return self.train_batched(examples, config)
    
    # ── Training (New Batched Interface) ──────────────────────────────
    
    def train_batched(
        self,
        examples: List[TrainingExampleV2],
        config: Optional[TrainingConfig] = None,
    ) -> Dict[str, Any]:
        """
        Full batched training loop with validation, LR scheduling, and
        gradient clipping.
        
        Args:
            examples: List of TrainingExampleV2
            config: Training configuration (uses defaults if None)
        
        Returns:
            Training history with train/val losses per epoch
        """
        if config is None:
            config = TrainingConfig()
        
        if not examples:
            return {"status": "No examples provided"}
        
        # ── Prepare data ──────────────────────────────────────────────
        all_logits = torch.stack([e.logits for e in examples])
        all_probs = torch.stack([e.probs for e in examples])
        all_error = torch.tensor([e.error_magnitude for e in examples], dtype=torch.float)
        all_conf = torch.tensor([e.confidence for e in examples], dtype=torch.float)
        all_dir = torch.tensor([e.error_direction for e in examples], dtype=torch.float)
        
        # Train/val split
        n = len(examples)
        n_val = max(1, int(n * config.validation_split))
        perm = torch.randperm(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        
        train_dataset = TensorDataset(
            all_logits[train_idx], all_probs[train_idx],
            all_error[train_idx], all_conf[train_idx], all_dir[train_idx],
        )
        val_dataset = TensorDataset(
            all_logits[val_idx], all_probs[val_idx],
            all_error[val_idx], all_conf[val_idx], all_dir[val_idx],
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        
        # ── Optimizer & Scheduler ─────────────────────────────────────
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        if config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs
            )
        elif config.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, config.epochs // 3), gamma=0.5
            )
        else:
            scheduler = None
        
        # ── Training loop ─────────────────────────────────────────────
        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        
        for epoch in range(1, config.epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                b_logits, b_probs, b_error, b_conf, b_dir = [
                    t.to(self.device) for t in batch
                ]
                
                optimizer.zero_grad()
                
                outputs = self.model(b_logits, b_probs)
                targets = {
                    "error_magnitude": b_error,
                    "confidence": b_conf,
                    "error_direction": b_dir,
                }
                
                losses = self.criterion(outputs, targets)
                losses["total"].backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=config.max_grad_norm
                )
                
                optimizer.step()
                train_loss += losses["total"].item()
                n_batches += 1
            
            avg_train_loss = train_loss / max(n_batches, 1)
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    b_logits, b_probs, b_error, b_conf, b_dir = [
                        t.to(self.device) for t in batch
                    ]
                    
                    outputs = self.model(b_logits, b_probs)
                    targets = {
                        "error_magnitude": b_error,
                        "confidence": b_conf,
                        "error_direction": b_dir,
                    }
                    
                    losses = self.criterion(outputs, targets)
                    val_loss += losses["total"].item()
                    n_val_batches += 1
            
            avg_val_loss = val_loss / max(n_val_batches, 1)
            
            # LR step
            if scheduler is not None:
                scheduler.step()
            
            # Logging
            current_lr = optimizer.param_groups[0]["lr"]
            
            epoch_record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
            }
            history.append(epoch_record)
            
            if epoch % config.log_every == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{config.epochs} — "
                    f"train_loss: {avg_train_loss:.4f}, "
                    f"val_loss: {avg_val_loss:.4f}, "
                    f"lr: {current_lr:.6f}"
                )
            
            # Checkpoint best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if config.checkpoint_dir:
                    self.save(os.path.join(config.checkpoint_dir, "best_model.pt"))
        
        self.is_trained = True
        self.training_history.extend(history)
        
        return {
            "status": "Training complete",
            "best_val_loss": best_val_loss,
            "final_train_loss": avg_train_loss,
            "total_epochs_trained": config.epochs,
            "history": history,
        }
    
    # ── Serialization ─────────────────────────────────────────────────
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "vocab_size": self.vocab_size,
            "device": self.device,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
            "model_config": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.residual_blocks[0].fc1.in_features
                    if len(self.model.residual_blocks) > 0 else 512,
            },
        }, path)
        logger.info(f"PoP v2 saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = checkpoint.get("is_trained", True)
        self.training_history = checkpoint.get("training_history", [])
        logger.info(f"PoP v2 loaded from {path}")
    
    # ── Introspection ─────────────────────────────────────────────────
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters and state."""
        return {
            "version": "v2",
            "device": self.device,
            "vocab_size": self.vocab_size,
            "num_parameters": self.model.count_parameters(),
            "is_trained": self.is_trained,
            "input_features": self.model.input_dim,
            "training_history_length": len(self.training_history),
        }
    
    def get_feature_importance(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute approximate feature importance via gradient magnitude.
        
        Useful for understanding which distributional features the model
        relies on most for error prediction.
        """
        self.model.eval()
        logits = logits.to(self.device).requires_grad_(False)
        probs = probs.to(self.device).requires_grad_(False)
        
        features = extract_features_vectorized(logits, probs).detach()
        features.requires_grad_(True)
        
        # Forward through rest of network
        x = self.model.input_norm(features)
        x = self.model.projection(x)
        x = self.model.residual_blocks(x)
        error_out = self.model.error_head(x).sum()
        
        error_out.backward()
        
        importance = features.grad.abs().mean(dim=0).cpu().tolist()
        
        feature_names = [
            "entropy", "norm_entropy", "perplexity", "top1", "margin",
            "top3_mass", "top10_mass", "top50_mass", "head_tail_ratio",
            "logit_range", "logit_std", "logit_norm", "logit_skew",
            "p25", "p50", "p75", "iqr", "eff_vocab", "prob_var",
            "nll_top1", "top1_vs_mean", "gini", "n_active", "concentration_ratio",
        ]
        
        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True,
        ))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pop_v2(
    vocab_size: int,
    device: Optional[str] = None,
    hidden_dim: int = 512,
) -> PoPLayerLLMV2:
    """Factory function — creates a PoP v2 layer."""
    return PoPLayerLLMV2(
        vocab_size=vocab_size,
        device=device,
        hidden_dim=hidden_dim,
    )


# Backward-compatible alias
create_pop_llm = create_pop_v2
