import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')
import torch
import numpy as np
from pop.core.llm_base import create_llm

print("Loading DistilGPT2...")
llm = create_llm('distilgpt2', device='cpu')

# 200 diverse prompts
prompts = [
    # Factual
    "The capital of France is", "The chemical symbol for gold is", "The largest planet is",
    "Water boils at", "The speed of light is", "The atomic number of carbon is",
    "The first man on the moon was", "The pyramids are in", "Newton's first law is about",
    "The square root of 144 is", "World War II ended in", "Albert Einstein was born in",
    "The largest country by area is", "The longest river is", "Pi is approximately",
    "The chemical symbol for iron is", "Mount Everest is in", "The Great Wall of",
    "The periodic table has", "The closest star to Earth is", "A triangle has",
    "The human heart has", "Light year is a unit of", "World War I started in",
    "The chemical symbol for water is", "The tallest mountain is", "The sun is a",
    "DNA stands for", "The smallest country is", "The inventor of the telephone",
    # Common phrases
    "She opened the", "He picked up the", "The cat sat on the", "I went to the",
    "I have a", "The movie was", "I think it is", "He went to the",
    "The book is", "We went to the", "She is very", "The food was",
    "I need to", "The children were", "She looked at the", "The dog ran",
    "I like to", "The train arrived", "It was a", "He said it was",
    "They went to the", "She said the", "The water is", "I saw a",
    "The house was", "He looked at the", "The car was", "She gave him a",
    "We had a", "The room was", "It is a", "I found a",
    "The man was", "She has a", "He had a", "They had a",
    "The dog was", "I was a", "The light was", "She was a",
    # Science
    "Photosynthesis converts", "The mitochondria is the", "Evolution is driven by",
    "Gravity is a", "Atoms are made of", "The speed of sound is",
    "Electricity flows through", "The ozone layer", "DNA replication",
    "Thermodynamics studies", "Quantum mechanics", "The theory of relativity",
    "Black holes are", "The Higgs boson", "Neural networks",
    "Machine learning", "Deep learning", "Natural language processing",
    # Creative
    "Once upon a time", "The adventure began when", "In a galaxy far away",
    "The hero discovered", "The mystery deepened as", "It was a dark and stormy",
    "The end of the world", "Time travel is", "The last human on Earth",
    "A robot walked into", "The future of humanity", "In the year 3000",
    # More factual
    "The Eiffel Tower is in", "The Amazon river", "Shakespeare wrote",
    "The Mona Lisa was painted by", "The internet was invented", "The theory of evolution",
    "The speed of light", "Water is composed of", "The human brain has",
    "Photosynthesis occurs in", "The first computer was", "The moon landing was in",
    # Math
    "2 plus 2 equals", "The square of 5 is", "10 divided by 2 is",
    "The derivative of x squared is", "Pi times the radius squared", "The integral of 1 is",
    "A prime number is", "The fibonacci sequence", "E equals mc",
    # More common
    "Today is a", "The best way to", "In my opinion", "The most important thing",
    "Life is about", "The meaning of", "Success comes from", "Knowledge is",
    "The future looks", "Technology will", "Artificial intelligence", "The internet of things",
    "Climate change is", "The economy is", "Education should be",
    "Health is", "Music makes", "Art expresses", "Love is",
    "Friendship means", "Happiness comes from", "Dreams are",
    "The world needs", "Humanity should", "Progress requires",
    "Innovation drives", "Creativity is", "Leadership means",
    "Teamwork makes", "Communication is", "Patience is",
    "Courage means", "Wisdom comes from", "Truth is",
    "Justice requires", "Peace is built", "Freedom means",
    # Tech
    "Python is a", "JavaScript runs", "The cloud is",
    "Blockchain technology", "5G networks", "Quantum computing",
    "Cybersecurity protects", "Big data", "IoT devices",
    "Virtual reality", "Augmented reality", "Self-driving cars",
    # Additional
    "The answer to life", "42 is", "Hello world",
    "The quick brown fox", "To be or not", "All your base",
    "May the force be", "Live long and", "I'll be back",
    "Houston we have", "One small step", "That's one small step",
    "Ask not what your", "I have a dream", "Four score and seven",
    "We hold these truths", "When in the course", "Give me liberty",
]

print(f"Generating training data from {len(prompts)} prompts...")

features_list = []
labels_list = []
prompts_out = []
tokens_out = []

for i, prompt in enumerate(prompts):
    try:
        logits = llm.get_logits(prompt)
        probs = torch.softmax(logits, dim=-1)
        
        # Good example (LLM's actual prediction)
        top1_prob = torch.max(probs).item()
        top1_idx = torch.argmax(probs).item()
        top1_token = llm.tokenizer.decode(top1_idx)
        
        # Extract 16 features (same as pop_layer_llm.py)
        feat = []
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        feat.append(entropy)
        feat.append(top1_prob)
        
        top3, _ = torch.topk(probs, 3)
        feat.append(torch.sum(top3).item())
        
        top10, _ = torch.topk(probs, 10)
        feat.append(torch.sum(top10).item())
        
        feat.append((torch.max(logits) - torch.min(logits)).item())
        feat.append(torch.mean(logits).item())
        feat.append(torch.std(logits).item())
        
        feat.append(float(torch.sum(probs > 0.01).item()))
        feat.append(float(torch.sum(probs > 0.1).item()))
        
        sorted_probs, _ = torch.sort(probs)
        n = len(sorted_probs)
        feat.append(sorted_probs[int(0.25 * n)].item())
        feat.append(sorted_probs[int(0.5 * n)].item())
        feat.append(sorted_probs[int(0.75 * n)].item())
        
        feat.append(torch.var(probs).item())
        
        # Gini
        sorted_desc, _ = torch.sort(probs, descending=True)
        rank = torch.arange(1, n+1, dtype=torch.float32)
        cumsum = torch.cumsum(sorted_desc, dim=0)
        gini = (2 * torch.sum(rank * sorted_desc) - (n+1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)
        feat.append(gini.item())
        
        # Log max/min ratio
        min_prob = torch.min(probs).item()
        feat.append(float(np.log(top1_prob + 1e-10) - np.log(min_prob + 1e-10)))
        
        # Log-sum-exp
        feat.append(torch.logsumexp(logits, dim=-1).item())
        
        features_list.append(feat)
        labels_list.append(0)  # good
        prompts_out.append(prompt)
        tokens_out.append(top1_token)
        
        # Bad example (random low-prob token)
        sorted_probs_idx = torch.argsort(probs, descending=True)
        bad_idx = sorted_probs_idx[np.random.randint(20, min(100, len(sorted_probs_idx)))]
        bad_token = llm.tokenizer.decode(bad_idx.item())
        bad_prob = probs[bad_idx].item()
        
        # Same features but with confused context
        feat_bad = list(feat)
        feat_bad[1] = bad_prob  # lower top-1 prob
        feat_bad[0] = entropy + np.random.uniform(0.5, 2.0)  # higher entropy
        feat_bad[13] = feat_bad[13] * 1.5  # more variance
        
        features_list.append(feat_bad)
        labels_list.append(1)  # bad
        prompts_out.append(prompt)
        tokens_out.append(bad_token)
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(prompts)} prompts")
    except Exception as e:
        print(f"  Error on prompt '{prompt}': {e}")

features = np.array(features_list, dtype=np.float32)
labels = np.array(labels_list, dtype=np.int32)

# Save full dataset
np.save('/root/.openclaw/workspace-main/pop-repo/training_data_large.npy', {
    'features': features,
    'labels': labels,
    'prompts': prompts_out,
    'predicted_tokens': tokens_out,
})

# Train/val split (80/20)
n = len(features)
indices = np.random.permutation(n)
train_idx = indices[:int(0.8 * n)]
val_idx = indices[int(0.8 * n):]

np.save('/root/.openclaw/workspace-main/pop-repo/training_data_train.npy', {
    'features': features[train_idx],
    'labels': labels[train_idx],
})
np.save('/root/.openclaw/workspace-main/pop-repo/training_data_val.npy', {
    'features': features[val_idx],
    'labels': labels[val_idx],
})

print(f"\nDone! Generated {len(features)} samples")
print(f"  Good (0): {np.sum(labels == 0)}")
print(f"  Bad (1): {np.sum(labels == 1)}")
print(f"  Features shape: {features.shape}")
print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
