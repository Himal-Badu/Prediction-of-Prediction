"""
LLM Integration - Base LLM using HuggingFace transformers.
"""
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMBase:
    """
    Base LLM using HuggingFace transformers.
    Provides probability distributions from the prediction layer.
    """
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[str] = None
    ):
        """
        Initialize the LLM.
        
        Args:
            model_name: Name of the model to load (distilgpt2, gpt2)
            device: Device to use ('cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing LLM: {model_name} on {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.vocab_size = None
        self.is_loaded = False
        
    def load(self) -> "LLMBase":
        """Load the model and tokenizer."""
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        self.vocab_size = self.model.config.vocab_size
        self.is_loaded = True
        
        logger.info(f"Model loaded. Vocab size: {self.vocab_size}")
        return self
    
    def _get_input_ids(self, text: str) -> torch.Tensor:
        """Convert text to input IDs."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        return inputs["input_ids"].to(self.device)
    
    def predict_next_token(
        self,
        text: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Predict next token probabilities.
        
        Args:
            text: Input text
            top_k: Number of top tokens to return
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction")
        
        input_ids = self._get_input_ids(text)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last token predictions
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get top k
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            
            # Decode tokens
            top_tokens = [
                self.tokenizer.decode(idx.item()) 
                for idx in top_indices[0]
            ]
            
            return {
                "input_text": text,
                "top_tokens": top_tokens,
                "top_probs": top_probs[0].cpu().numpy().tolist(),
                "top_indices": top_indices[0].cpu().numpy().tolist(),
                "full_probs": probs[0].cpu().numpy()
            }
    
    def get_probability_distribution(
        self,
        text: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get full probability distribution over vocabulary.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (probabilities array, token list)
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction")
        
        input_ids = self._get_input_ids(text)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
        # Get all vocab
        vocab_ids = list(range(self.vocab_size))
        tokens = [self.tokenizer.decode(i) for i in vocab_ids]
        
        return probs[0].cpu().numpy(), tokens
    
    def generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """
        Generate text continuation.
        
        Args:
            text: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated text
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction")
        
        input_ids = self._get_input_ids(text)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=torch.ones_like(input_ids)
            )
        
        return self.tokenizer.decode(outputs[0])
    
    def get_logits(self, text: str) -> torch.Tensor:
        """
        Get raw logits for a given text.
        
        Args:
            text: Input text
            
        Returns:
            Raw logits tensor
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before prediction")
        
        input_ids = self._get_input_ids(text)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            
        return logits[0]
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "vocab_size": self.vocab_size,
            "is_loaded": self.is_loaded
        }


def create_llm(model_name: str = "distilgpt2", device: Optional[str] = None) -> LLMBase:
    """Factory function to create an LLM."""
    llm = LLMBase(model_name=model_name, device=device)
    return llm.load()