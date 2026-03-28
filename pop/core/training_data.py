"""
Balanced training data for PoP.
50% examples where LLM is RIGHT, 50% where LLM is WRONG.
This is critical — PoP needs contrast to learn the difference.
"""
from typing import List, Dict


def get_llm_correct_prompts() -> List[Dict[str, str]]:
    """
    Completion-style prompts where DistilGPT-2 usually gets it right.
    These teach PoP: "this distribution pattern = LLM is probably right"
    """
    return [
        {"prompt": "She opened the", "answer": " door"},
        {"prompt": "He picked up the", "answer": " phone"},
        {"prompt": "The cat sat on the", "answer": " floor"},
        {"prompt": "I went to the", "answer": " hospital"},
        {"prompt": "I have a", "answer": " lot"},
        {"prompt": "The movie was", "answer": " a"},
        {"prompt": "I think it is", "answer": " a"},
        {"prompt": "He went to the", "answer": " hospital"},
        {"prompt": "The book is", "answer": " a"},
        {"prompt": "We went to the", "answer": " hospital"},
        {"prompt": "She is very", "answer": " good"},
        {"prompt": "The food was", "answer": " good"},
        {"prompt": "I need to", "answer": " go"},
        {"prompt": "The children were", "answer": " playing"},
        {"prompt": "She looked at the", "answer": " door"},
        {"prompt": "The dog ran", "answer": " away"},
        {"prompt": "I like to", "answer": " play"},
        {"prompt": "The train arrived", "answer": " at"},
        {"prompt": "It was a", "answer": " great"},
        {"prompt": "He said it was", "answer": " a"},
        {"prompt": "They went to the", "answer": " store"},
        {"prompt": "She said the", "answer": " same"},
        {"prompt": "The water is", "answer": " a"},
        {"prompt": "I saw a", "answer": " man"},
        {"prompt": "The house was", "answer": " a"},
        {"prompt": "He looked at the", "answer": " door"},
        {"prompt": "The car was", "answer": " a"},
        {"prompt": "She gave him a", "answer": " hug"},
        {"prompt": "We had a", "answer": " great"},
        {"prompt": "The room was", "answer": " a"},
        {"prompt": "It is a", "answer": " great"},
        {"prompt": "I found a", "answer": " way"},
        {"prompt": "The man was", "answer": " a"},
        {"prompt": "She has a", "answer": " lot"},
        {"prompt": "He had a", "answer": " great"},
        {"prompt": "They had a", "answer": " great"},
        {"prompt": "The dog was", "answer": " a"},
        {"prompt": "I was a", "answer": " bit"},
        {"prompt": "The light was", "answer": " a"},
        {"prompt": "She was a", "answer": " great"},
    ]


def get_llm_wrong_prompts() -> List[Dict[str, str]]:
    """
    Factual prompts where DistilGPT-2 is usually WRONG.
    These teach PoP: "this distribution pattern = LLM is probably wrong"
    """
    return [
        {"prompt": "The capital of France is", "answer": " Paris"},
        {"prompt": "The chemical symbol for gold is", "answer": " Au"},
        {"prompt": "The largest planet is", "answer": " Jupiter"},
        {"prompt": "Shakespeare wrote", "answer": " Hamlet"},
        {"prompt": "World War II ended in", "answer": " 1945"},
        {"prompt": "The pyramids are in", "answer": " Egypt"},
        {"prompt": "The Mona Lisa was painted by", "answer": " Leonardo"},
        {"prompt": "Newton's first law is about", "answer": " inertia"},
        {"prompt": "The square root of 144 is", "answer": " 12"},
        {"prompt": "The opposite of hot is", "answer": " cold"},
        {"prompt": "The speed of light is approximately", "answer": " 3"},
        {"prompt": "The atomic number of carbon is", "answer": " 6"},
        {"prompt": "The first man on the moon was", "answer": " Neil"},
        {"prompt": "The mitochondria is the", "answer": " powerhouse"},
        {"prompt": "Albert Einstein was born in", "answer": " 1879"},
        {"prompt": "The largest country by area is", "answer": " Russia"},
        {"prompt": "The longest river is the", "answer": " Nile"},
        {"prompt": "Pi is approximately", "answer": " 3"},
        {"prompt": "Water boils at", "answer": " 100"},
        {"prompt": "The chemical symbol for iron is", "answer": " Fe"},
        {"prompt": "The inventor of the telephone was", "answer": " Alexander"},
        {"prompt": "Mount Everest is in", "answer": " Nepal"},
        {"prompt": "The Great Wall of", "answer": " China"},
        {"prompt": "The internet was invented in", "answer": " 1969"},
        {"prompt": "The theory of relativity was by", "answer": " Albert"},
        {"prompt": "The Amazon is the largest", "answer": " river"},
        {"prompt": "The periodic table has", "answer": " 118"},
        {"prompt": "The closest star to Earth is", "answer": " the"},
        {"prompt": "A triangle has", "answer": " three"},
        {"prompt": "The human heart has", "answer": " four"},
        {"prompt": "Light year is a unit of", "answer": " distance"},
        {"prompt": "World War I started in", "answer": " 1914"},
        {"prompt": "The chemical symbol for water is", "answer": " H"},
        {"prompt": "The tallest mountain is", "answer": " Mount"},
        {"prompt": "The sun is a", "answer": " star"},
        {"prompt": "DNA stands for", "answer": " de"},
        {"prompt": "2 + 2 equals", "answer": " 4"},
        {"prompt": "The color of grass is", "answer": " green"},
        {"prompt": "The sky is", "answer": " blue"},
        {"prompt": "The smallest country is", "answer": " Vatican"},
    ]


def get_balanced_facts() -> List[Dict[str, str]]:
    """Get balanced training data — equal LLM-right and LLM-wrong examples."""
    correct = get_llm_correct_prompts()
    wrong = get_llm_wrong_prompts()
    # Balance: use min of both
    n = min(len(correct), len(wrong))
    return correct[:n] + wrong[:n]


def get_all_facts() -> List[Dict[str, str]]:
    """Get all training data combined."""
    return get_balanced_facts()
