"""
PoP v3: Logit Signal Transformer (LST)

New framework for hallucination detection that:
1. Uses RAW logit distributions (not hand-crafted scalar summaries)
2. Multi-scale entropy analysis (temperature sweep)
3. Token-position-aware features
4. Self-consistency sampling signal
5. Meta-learning (MAML) for cross-domain adaptation

This replaces the 24 hand-crafted features approach entirely.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LSTConfig:
    """Configuration for Logit Signal Transformer."""
    vocab_size: int = 32000        # Vocabulary size (compressed)
    logit_compression_dim: int = 256  # Compress vocab -> this
    n_temperature_scales: int = 5    # Number of temperature scales
    temperatures: list = None        # Temperature values
    n_samples: int = 10              # Self-consistency samples
    hidden_dim: int = 128            # Hidden dimension
    n_attention_heads: int = 4       # Attention heads
    n_layers: int = 3                # Transformer layers
    dropout: float = 0.1
    max_seq_len: int = 512           # Maximum sequence length
    
    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = [0.3, 0.5, 1.0, 1.5, 2.0]


class LogitCompressor(nn.Module):
    """
    Compress full logit distribution into dense representation.
    Instead of hand-crafted features, LEARN the compression.
    """
    def __init__(self, vocab_size: int, compressed_dim: int):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(vocab_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, compressed_dim),
            nn.LayerNorm(compressed_dim),
            nn.GELU(),
        )
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, vocab_size) raw logit vector
        Returns:
            (batch, compressed_dim) dense representation
        """
        return self.compressor(logits)


class MultiScaleEntropy(nn.Module):
    """
    Compute entropy at multiple temperature scales.
    Different temperatures reveal different aspects of the distribution.
    - Low temp (0.3): what the model would greedily pick (sharp distribution)
    - High temp (2.0): what the model considers possible (flattened distribution)
    The RELATIONSHIP between these scales is the signal.
    """
    def __init__(self, temperatures: list):
        super().__init__()
        self.temperatures = temperatures
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, vocab_size)
        Returns:
            (batch, n_temps) entropy at each temperature
        """
        entropies = []
        for T in self.temperatures:
            scaled_logits = logits / T
            probs = F.softmax(scaled_logits, dim=-1)
            # Clamp for numerical stability
            probs = probs.clamp(min=1e-10)
            entropy = -(probs * probs.log()).sum(dim=-1)
            entropies.append(entropy)
        
        return torch.stack(entropies, dim=-1)  # (batch, n_temps)


class TemperatureGradient(nn.Module):
    """
    Compute how entropy changes across temperatures.
    This captures SHAPE of the distribution, not just level.
    Steep gradient = distribution collapses quickly (confident)
    Flat gradient = distribution stays spread (uncertain)
    """
    def __init__(self, temperatures: list):
        super().__init__()
        self.temperatures = temperatures
    
    def forward(self, multi_scale_entropy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_scale_entropy: (batch, n_temps)
        Returns:
            (batch, n_temps-1) entropy gradients between consecutive temps
        """
        gradients = []
        for i in range(len(self.temperatures) - 1):
            grad = multi_scale_entropy[:, i+1] - multi_scale_entropy[:, i]
            gradients.append(grad)
        return torch.stack(gradients, dim=-1)


class SelfConsistencySampler(nn.Module):
    """
    Sample multiple tokens from the same distribution.
    Agreement between samples = consistency signal.
    
    Key insight: If we sample 10 times and 9/10 give the same token,
    the model is confident. If samples are all different, the model
    is uncertain. This is a SEMANTIC signal, not statistical.
    """
    def __init__(self, n_samples: int = 10):
        super().__init__()
        self.n_samples = n_samples
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, vocab_size)
        Returns:
            (batch, 3) consistency features:
              - agreement_ratio: fraction of samples that match the top sample
              - unique_count: number of unique tokens in samples
              - top_prob_vs_agreement: difference between top prob and empirical agreement
        """
        probs = F.softmax(logits, dim=-1)
        batch_size = logits.shape[0]
        
        # Sample n_samples tokens from the distribution
        samples = torch.multinomial(probs, self.n_samples, replacement=True)  # (batch, n_samples)
        
        features = []
        for b in range(batch_size):
            sampled_tokens = samples[b]
            top_token = sampled_tokens[0]  # First sample as reference
            
            # Agreement ratio: how many samples match the first?
            agreement = (sampled_tokens == top_token).float().mean()
            
            # Unique count: how many distinct tokens were sampled?
            unique_count = len(torch.unique(sampled_tokens).float())
            unique_ratio = float(unique_count) / self.n_samples
            
            # Top probability vs empirical agreement
            top_prob = probs[b].max().item()
            agreement_gap = agreement.item() - top_prob
            
            features.append(torch.tensor([agreement.item(), unique_ratio, agreement_gap]))
        
        return torch.stack(features, dim=0)  # (batch, 3)


class PositionalEncoder(nn.Module):
    """
    Encode token position information.
    Model confidence at token 1 means something different than at token 50.
    """
    def __init__(self, max_len: int = 512, d_model: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch,) token positions in sequence
        Returns:
            (batch, d_model) positional encoding
        """
        return self.embedding(positions.clamp(0, 511))


class LogitSignalTransformer(nn.Module):
    """
    PoP v3: Logit Signal Transformer (LST)
    
    Takes raw logit distributions and extracts learned features
    for hallucination detection. No hand-crafted features.
    
    Architecture:
    1. Logit Compressor: vocab_size -> dense vector (LEARNED)
    2. Multi-Scale Entropy: entropy at 5 temperature scales
    3. Temperature Gradient: how entropy changes across scales
    4. Self-Consistency: sampling-based agreement signal
    5. Positional Encoder: where in the sequence
    6. Fusion: combine all signals
    7. Transformer: learn interactions between signals
    8. Prediction Head: hallucination probability
    """
    
    def __init__(self, config: LSTConfig):
        super().__init__()
        self.config = config
        
        # Signal extractors
        self.compressor = LogitCompressor(config.vocab_size, config.logit_compression_dim)
        self.multi_scale = MultiScaleEntropy(config.temperatures)
        self.temp_gradient = TemperatureGradient(config.temperatures)
        self.consistency = SelfConsistencySampler(config.n_samples)
        self.pos_encoder = PositionalEncoder(config.max_seq_len, d_model=16)
        
        # Total features from all extractors:
        # compressor: 256
        # multi_scale: 5 (one per temperature)
        # temp_gradient: 4 (gradients between temps)
        # consistency: 3
        # position: 16
        total_features = (config.logit_compression_dim + 
                         config.n_temperature_scales + 
                         config.n_temperature_scales - 1 + 
                         3 + 16)
        
        # Fusion layer: combine all signals into one representation
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Self-attention: learn which signals matter for each sample
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Prediction heads
        self.hallucination_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )
        
        # Auxiliary: predict temperature (forces model to understand uncertainty scale)
        self.scale_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, config.n_temperature_scales),
        )
    
    def extract_all_features(
        self, 
        logits: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract all signals from raw logits.
        
        Args:
            logits: (batch, vocab_size) raw logit vectors
            positions: (batch,) token positions (optional, defaults to 0)
        
        Returns:
            (batch, total_features) fused feature vector
        """
        batch_size = logits.shape[0]
        
        if positions is None:
            positions = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # 1. Learned compression
        compressed = self.compressor(logits)  # (batch, 256)
        
        # 2. Multi-scale entropy
        entropy = self.multi_scale(logits)  # (batch, 5)
        
        # 3. Temperature gradients
        gradients = self.temp_gradient(entropy)  # (batch, 4)
        
        # 4. Self-consistency
        consistency = self.consistency(logits)  # (batch, 3)
        
        # 5. Positional encoding
        pos = self.pos_encoder(positions)  # (batch, 16)
        
        # Concatenate all signals
        combined = torch.cat([compressed, entropy, gradients, consistency, pos], dim=-1)
        
        return combined
    
    def forward(
        self,
        logits: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            logits: (batch, vocab_size) or (batch, seq_len, vocab_size)
            positions: (batch,) or (batch, seq_len)
        
        Returns:
            Dict with hallucination logits, scale predictions, features
        """
        if logits.dim() == 3:
            # Sequence of logit vectors — process each, then attend
            batch_size, seq_len, vocab_size = logits.shape
            all_features = []
            for t in range(seq_len):
                pos_t = positions[:, t] if positions is not None else torch.full((batch_size,), t, dtype=torch.long, device=logits.device)
                feat = self.extract_all_features(logits[:, t, :], pos_t)
                all_features.append(feat)
            
            # Stack as sequence: (batch, seq_len, hidden_dim)
            stacked = torch.stack(all_features, dim=1)
            fused = self.fusion(stacked)
            
            # Self-attention over the sequence
            encoded = self.encoder(fused)
            
            # Pool: use mean of encoded representations
            pooled = encoded.mean(dim=1)  # (batch, hidden_dim)
        else:
            # Single logit vector
            features = self.extract_all_features(logits, positions)
            fused = self.fusion(features)  # (batch, hidden_dim)
            # Add sequence dimension for transformer
            fused = fused.unsqueeze(1)  # (batch, 1, hidden_dim)
            encoded = self.encoder(fused)
            pooled = encoded.squeeze(1)  # (batch, hidden_dim)
        
        # Predictions
        hallucination_logit = self.hallucination_head(pooled).squeeze(-1)
        scale_pred = self.scale_head(pooled)
        
        return {
            'hallucination_logit': hallucination_logit,
            'hallucination_prob': torch.sigmoid(hallucination_logit),
            'scale_prediction': scale_pred,
            'pooled_features': pooled,
        }


class PoPv3Loss(nn.Module):
    """
    Multi-task loss for LST.
    Main task: hallucination detection
    Auxiliary: temperature scale prediction (forces uncertainty awareness)
    """
    def __init__(self, scale_weight: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.scale_weight = scale_weight
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        scale_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: model output dict
            labels: (batch,) binary labels (0=correct, 1=hallucination)
            scale_labels: (batch, n_scales) optional temperature scale labels
        """
        # Main loss: hallucination detection
        hall_loss = self.bce(predictions['hallucination_logit'], labels)
        
        total_loss = hall_loss
        losses = {'hallucination': hall_loss.item()}
        
        # Auxiliary loss: predict entropy at each temperature scale
        if scale_labels is not None:
            scale_loss = self.mse(predictions['scale_prediction'], scale_labels)
            total_loss = total_loss + self.scale_weight * scale_loss
            losses['scale'] = scale_loss.item()
        
        losses['total'] = total_loss.item()
        return total_loss, losses


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing Logit Signal Transformer (LST)...")
    
    config = LSTConfig(vocab_size=1000, logit_compression_dim=64, hidden_dim=64)
    model = LogitSignalTransformer(config)
    
    # Test with single logit vector
    batch_size = 16
    logits = torch.randn(batch_size, 1000)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    output = model(logits)
    print(f"  Single vector: hall_prob shape = {output['hallucination_prob'].shape}")
    print(f"  Sample probs: {output['hallucination_prob'][:5].detach().numpy()}")
    
    # Test with sequence of logit vectors
    seq_len = 5
    logits_seq = torch.randn(batch_size, seq_len, 1000)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    output_seq = model(logits_seq, positions)
    print(f"  Sequence: hall_prob shape = {output_seq['hallucination_prob'].shape}")
    
    # Test loss
    criterion = PoPv3Loss()
    loss, loss_dict = criterion(output, labels)
    print(f"  Loss: {loss.item():.4f} ({loss_dict})")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    print("\n✅ LST architecture working!")
