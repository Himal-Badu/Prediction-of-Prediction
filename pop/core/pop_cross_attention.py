"""
PoP Cross-Attention Fusion — Option B: Cross-Attention Expert Fusion with Adaptive Routing.

This module implements a new fusion architecture that replaces static alpha weighting
with cross-attention between specialists and an adaptive per-token routing gate.

Architecture:
    1. Feature Projectors: Project v1 (16-d) and v2 (24-d) features to a shared 64-d space
    2. Cross-Attention Block: Each specialist attends to the other's features
    3. Fusion MLP: Combines all representations into final predictions
    4. Adaptive Gate: Per-token decision between specialist-only vs full fusion

Key improvement over PoPFusion:
    - Static alpha → adaptive per-token gating
    - Independent specialists → cross-attention information sharing
    - Same predict() interface (drop-in compatible)

Usage:
    fusion = PoPCrossAttentionFusion(vocab_size=50257)
    result = fusion.predict(logits, probs)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
import math

from .pop_v2 import extract_features_vectorized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Extraction for v1 (16-d, matching pop_layer_llm.py)
# ---------------------------------------------------------------------------

def extract_features_v1(
    logits: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """
    Extract 16-d distributional features matching v1's LLMErrorPredictor.extract_features.
    Fully vectorized (no Python loops).
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    eps = 1e-10
    probs = probs.clamp(min=eps)

    # Entropy
    entropy = -(probs * torch.log(probs)).sum(dim=-1)

    # Top-1 probability
    top1_prob, _ = probs.max(dim=-1)

    # Top-3 mass
    top3_probs, _ = torch.topk(probs, min(3, probs.shape[-1]), dim=-1)
    top3_mass = top3_probs.sum(dim=-1)

    # Top-10 mass
    top10_probs, _ = torch.topk(probs, min(10, probs.shape[-1]), dim=-1)
    top10_mass = top10_probs.sum(dim=-1)

    # Logit range
    logit_range = logits.max(dim=-1).values - logits.min(dim=-1).values

    # Logit mean
    logit_mean = logits.mean(dim=-1)

    # Logit std
    logit_std = logits.std(dim=-1)

    # Number of tokens with prob > 0.01
    n_active = (probs > 0.01).float().sum(dim=-1)

    # Number of tokens with prob > 0.1
    n_confident = (probs > 0.1).float().sum(dim=-1)

    # Percentiles
    sorted_probs, _ = torch.sort(probs, dim=-1)
    B, V = probs.shape
    p25 = sorted_probs[:, int(0.25 * V)]
    p50 = sorted_probs[:, int(0.5 * V)]
    p75 = sorted_probs[:, int(0.75 * V)]

    # Probability variance
    prob_var = probs.var(dim=-1)

    # Gini coefficient
    sorted_desc, _ = torch.sort(probs, dim=-1, descending=True)
    indices = torch.arange(1, V + 1, device=probs.device, dtype=torch.float)
    cumsum = torch.cumsum(sorted_desc, dim=-1)
    weighted_sum = (indices.unsqueeze(0) * sorted_desc).sum(dim=-1)
    gini = (2 * weighted_sum - (V + 1) * cumsum[:, -1]) / (V * cumsum[:, -1] + eps)

    # Max/min ratio (log-scaled)
    min_prob, _ = probs.min(dim=-1)
    max_min_ratio = torch.log(top1_prob + eps) - torch.log(min_prob + eps)

    # Log-sum-exp
    log_sum_exp = torch.logsumexp(logits, dim=-1)

    features = torch.stack([
        entropy,          # 1
        top1_prob,        # 2
        top3_mass,        # 3
        top10_mass,       # 4
        logit_range,      # 5
        logit_mean,       # 6
        logit_std,        # 7
        n_active,         # 8
        n_confident,      # 9
        p25,              # 10
        p50,              # 11
        p75,              # 12
        prob_var,         # 13
        gini,             # 14
        max_min_ratio,    # 15
        log_sum_exp,      # 16
    ], dim=-1)

    return features


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention between two specialist feature sets.
    
    v1 attends to v2: query=v1_embed, key/value=v2_embed
    v2 attends to v1: query=v2_embed, key/value=v1_embed
    """
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 2):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # v1 attends to v2 (v1 queries, v2 keys/values)
        self.q1 = nn.Linear(embed_dim, embed_dim)
        self.k2 = nn.Linear(embed_dim, embed_dim)
        self.v2 = nn.Linear(embed_dim, embed_dim)
        self.out1 = nn.Linear(embed_dim, embed_dim)
        
        # v2 attends to v1 (v2 queries, v1 keys/values)
        self.q2 = nn.Linear(embed_dim, embed_dim)
        self.k1 = nn.Linear(embed_dim, embed_dim)
        self.v1 = nn.Linear(embed_dim, embed_dim)
        self.out2 = nn.Linear(embed_dim, embed_dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.scale = math.sqrt(self.head_dim)
    
    def _multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard multi-head attention.
        
        Args:
            query: (B, 1, embed_dim) — single token query
            key: (B, 1, embed_dim) — single token key
            value: (B, 1, embed_dim) — single token value
        
        Returns:
            (B, 1, embed_dim)
        """
        B = query.shape[0]
        
        # Project
        Q = query  # (B, 1, D)
        K = key    # (B, 1, D)
        V = value  # (B, 1, D)
        
        # Reshape to multi-head: (B, num_heads, 1, head_dim)
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, num_heads, 1, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention: (B, num_heads, 1, head_dim)
        attended = torch.matmul(attn_weights, V)
        
        # Concatenate heads: (B, 1, embed_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        
        return attended
    
    def forward(
        self,
        v1_embed: torch.Tensor,
        v2_embed: torch.Tensor,
    ) -> tuple:
        """
        Bidirectional cross-attention.
        
        Args:
            v1_embed: (B, embed_dim)
            v2_embed: (B, embed_dim)
        
        Returns:
            attended_v1: (B, embed_dim) — v1 after attending to v2
            attended_v2: (B, embed_dim) — v2 after attending to v1
        """
        # Add sequence dimension: (B, 1, D)
        q1 = v1_embed.unsqueeze(1)
        q2 = v2_embed.unsqueeze(1)
        k1 = v1_embed.unsqueeze(1)
        k2 = v2_embed.unsqueeze(1)
        
        # v1 attends to v2
        q1_proj = self.q1(q1)
        k2_proj = self.k2(k2)
        v2_proj = self.v2(k2)
        attended_v1 = self.out1(self._multi_head_attention(q1_proj, k2_proj, v2_proj))
        attended_v1 = self.norm1(attended_v1.squeeze(1) + v1_embed)  # residual + norm
        
        # v2 attends to v1
        q2_proj = self.q2(q2)
        k1_proj = self.k1(k1)
        v1_proj = self.v1(k1)
        attended_v2 = self.out2(self._multi_head_attention(q2_proj, k1_proj, v1_proj))
        attended_v2 = self.norm2(attended_v2.squeeze(1) + v2_embed)  # residual + norm
        
        return attended_v1, attended_v2


# ---------------------------------------------------------------------------
# Fusion MLP with Output Heads
# ---------------------------------------------------------------------------

class FusionMLP(nn.Module):
    """
    Fusion MLP that combines cross-attended representations.
    
    Input: concat(v1_embed, attended_v1, v2_embed, attended_v2) → 256-d
    Hidden: 256 → 128 → 64
    Output heads: error_magnitude, confidence, error_direction
    """
    
    def __init__(self, input_dim: int = 256, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.error_head = nn.Linear(hidden2, 1)
        self.confidence_head = nn.Linear(hidden2, 1)
        self.direction_head = nn.Linear(hidden2, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        
        return {
            "error_magnitude": torch.sigmoid(self.error_head(h)).squeeze(-1),
            "confidence": torch.sigmoid(self.confidence_head(h)).squeeze(-1),
            "error_direction": torch.tanh(self.direction_head(h)).squeeze(-1),
            "fused_repr": h,
        }


# ---------------------------------------------------------------------------
# Adaptive Gate
# ---------------------------------------------------------------------------

class AdaptiveGate(nn.Module):
    """
    Adaptive gating mechanism that decides per-token whether full fusion
    is needed or single expert suffices.
    
    Input: concat(v1_features, v2_features, fused_representation)
    Tiny MLP: input_dim → 32 → 1 → sigmoid → alpha
    
    alpha=1 → use fusion output
    alpha=0 → use specialist output
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) concatenated features
        
        Returns:
            alpha: (B,) gating values in [0, 1]
        """
        return self.gate(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

class PoPCrossAttentionFusion:
    """
    Cross-Attention Expert Fusion with Adaptive Routing (Option B).
    
    Drop-in replacement for PoPFusion with the same predict() interface.
    
    Improvements over static PoPFusion:
    - Cross-attention: specialists see each other's features before prediction
    - Adaptive gating: per-token decision on fusion intensity
    - Richer fusion: 4-way concatenation after bidirectional attention
    """
    
    def __init__(
        self,
        vocab_size: int,
        device: Optional[str] = None,
        hidden_dim: int = 64,
        num_heads: int = 2,
    ):
        """
        Args:
            vocab_size: LLM vocabulary size
            device: 'cpu' or 'cuda'
            hidden_dim: Embedding dimension for cross-attention (default 64)
            num_heads: Number of attention heads (default 2)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Feature projectors: map different-dim features to shared space
        self.v1_projector = nn.Linear(16, hidden_dim)  # v1 has 16 features
        self.v2_projector = nn.Linear(24, hidden_dim)  # v2 has 24 features
        
        # Cross-attention block
        self.cross_attention = CrossAttentionBlock(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Fusion MLP: concat of 4 × hidden_dim = 256
        fusion_input_dim = 4 * hidden_dim
        self.fusion_mlp = FusionMLP(input_dim=fusion_input_dim)
        
        # Adaptive gate: input = v1_features (16) + v2_features (24) + fused_repr (64)
        gate_input_dim = 16 + 24 + hidden_dim
        self.adaptive_gate = AdaptiveGate(input_dim=gate_input_dim)
        
        # Move everything to device
        self.v1_projector.to(self.device)
        self.v2_projector.to(self.device)
        self.cross_attention.to(self.device)
        self.fusion_mlp.to(self.device)
        self.adaptive_gate.to(self.device)
        
        # Layer norms for projected features
        self.v1_norm = nn.LayerNorm(hidden_dim).to(self.device)
        self.v2_norm = nn.LayerNorm(hidden_dim).to(self.device)
        
        logger.info(
            f"PoP Cross-Attention Fusion initialized: "
            f"hidden_dim={hidden_dim}, num_heads={num_heads}, device={self.device}"
        )
    
    def forward_features(
        self,
        v1_features: torch.Tensor,
        v2_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Raw feature path — forward pass through the cross-attention architecture.
        
        Args:
            v1_features: (B, 16) v1 distributional features
            v2_features: (B, 24) v2 distributional features
        
        Returns:
            Dict with:
                - error_magnitude: (B,)
                - confidence: (B,)
                - error_direction: (B,)
                - alpha: (B,) adaptive gate values
                - attended_v1: (B, 64)
                - attended_v2: (B, 64)
                - fused_repr: (B, 64)
        """
        # Project to shared embedding space
        v1_embed = F.relu(self.v1_projector(v1_features))  # (B, 64)
        v2_embed = F.relu(self.v2_projector(v2_features))  # (B, 64)
        
        # Normalize
        v1_embed = self.v1_norm(v1_embed)
        v2_embed = self.v2_norm(v2_embed)
        
        # Cross-attention
        attended_v1, attended_v2 = self.cross_attention(v1_embed, v2_embed)
        
        # Fusion: concatenate all representations
        fused_input = torch.cat([v1_embed, attended_v1, v2_embed, attended_v2], dim=-1)
        fusion_output = self.fusion_mlp(fused_input)
        
        # Adaptive gate
        gate_input = torch.cat([v1_features, v2_features, fusion_output["fused_repr"]], dim=-1)
        alpha = self.adaptive_gate(gate_input)
        
        return {
            "error_magnitude": fusion_output["error_magnitude"],
            "confidence": fusion_output["confidence"],
            "error_direction": fusion_output["error_direction"],
            "alpha": alpha,
            "attended_v1": attended_v1,
            "attended_v2": attended_v2,
            "fused_repr": fusion_output["fused_repr"],
        }
    
    def predict(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Predict error likelihood — same interface as PoPFusion.predict().
        
        Runs both feature extractors, applies cross-attention fusion,
        and uses adaptive gating to blend specialist and fusion outputs.
        
        Args:
            logits: LLM logits, shape (V,) or (B, V)
            probs: LLM probabilities, shape (V,) or (B, V)
        
        Returns:
            Dict with predictions (same keys as PoPFusion):
                - error_magnitude, confidence, error_direction
                - should_correct, llm_likely_wrong, llm_overconfident, llm_underconfident
                - model_type, alpha (replaces v1_weight)
        """
        # Ensure 2D
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        logits = logits.to(self.device)
        probs = probs.to(self.device)
        
        # Set all submodules to eval mode for deterministic behavior
        self.v1_projector.eval()
        self.v2_projector.eval()
        self.cross_attention.eval()
        self.fusion_mlp.eval()
        self.adaptive_gate.eval()
        self.v1_norm.eval()
        self.v2_norm.eval()
        
        with torch.no_grad():
            # Extract features from both specialists
            v1_features = extract_features_v1(logits, probs)   # (B, 16)
            v2_features = extract_features_vectorized(logits, probs)  # (B, 24)
            
            # Forward through cross-attention architecture
            result = self.forward_features(v1_features, v2_features)
        
        # For single-sample, extract scalars
        if result["alpha"].dim() == 0 or result["alpha"].numel() == 1:
            alpha_val = result["alpha"].item()
            error_mag = result["error_magnitude"].item()
            confidence = result["confidence"].item()
            direction = result["error_direction"].item()
        else:
            # Batch: use first sample
            alpha_val = result["alpha"][0].item()
            error_mag = result["error_magnitude"][0].item()
            confidence = result["confidence"][0].item()
            direction = result["error_direction"][0].item()
        
        # Decision logic
        should_correct = confidence > 0.7 and error_mag > 0.3
        
        return {
            "error_magnitude": error_mag,
            "confidence": confidence,
            "error_direction": direction,
            "should_correct": should_correct,
            "llm_likely_wrong": error_mag > 0.5,
            "llm_overconfident": direction > 0.3,
            "llm_underconfident": direction < -0.3,
            "model_type": "cross_attention_fusion",
            "alpha": alpha_val,
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters and architecture info."""
        total_params = (
            sum(p.numel() for p in self.v1_projector.parameters()) +
            sum(p.numel() for p in self.v2_projector.parameters()) +
            sum(p.numel() for p in self.cross_attention.parameters()) +
            sum(p.numel() for p in self.fusion_mlp.parameters()) +
            sum(p.numel() for p in self.adaptive_gate.parameters()) +
            sum(p.numel() for p in self.v1_norm.parameters()) +
            sum(p.numel() for p in self.v2_norm.parameters())
        )
        
        return {
            "model_type": "cross_attention_fusion",
            "vocab_size": self.vocab_size,
            "device": self.device,
            "hidden_dim": self.hidden_dim,
            "total_parameters": total_params,
            "architecture": {
                "v1_projector": "16 → 64",
                "v2_projector": "24 → 64",
                "cross_attention": f"2-head, {self.hidden_dim}-d",
                "fusion_mlp": "256 → 128 → 64 → 3 heads",
                "adaptive_gate": f"{16 + 24 + self.hidden_dim} → 32 → 1",
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pop_cross_attention_fusion(
    vocab_size: int,
    device: Optional[str] = None,
) -> PoPCrossAttentionFusion:
    """Factory function — creates a PoP Cross-Attention Fusion layer."""
    return PoPCrossAttentionFusion(vocab_size=vocab_size, device=device)
