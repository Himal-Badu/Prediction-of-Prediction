"""
PoP Smart Demo — balanced training, both modes.
Shows PoP working in passive (warn) and active (correct) modes.
"""
import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')
import torch
import numpy as np
from pop.core.llm_base import create_llm
from pop.core.pop_layer_llm import create_pop_llm
from pop.core.debugger import PoPDebugger
from pop.core.training_data import get_balanced_facts, get_llm_correct_prompts, get_llm_wrong_prompts

print("Loading DistilGPT-2...")
llm = create_llm('distilgpt2', device='cpu')
pop = create_pop_llm(vocab_size=llm.vocab_size, device='cpu')

# ── Check LLM accuracy on balanced data ──
correct_prompts = get_llm_correct_prompts()
wrong_prompts = get_llm_wrong_prompts()

llm_right = 0
for tc in correct_prompts:
    r = llm.predict_next_token(tc['prompt'], top_k=1)
    if r['top_tokens'][0].strip().lower() == tc['answer'].strip().lower():
        llm_right += 1

llm_wrong_count = 0
for tc in wrong_prompts:
    r = llm.predict_next_token(tc['prompt'], top_k=1)
    if r['top_tokens'][0].strip().lower() != tc['answer'].strip().lower():
        llm_wrong_count += 1

print(f"\nLLM stats on training data:")
print(f"  Should-be-correct set: {llm_right}/{len(correct_prompts)} actually correct")
print(f"  Should-be-wrong set:   {llm_wrong_count}/{len(wrong_prompts)} actually wrong")

# ── Train PoP with balanced data ──
facts = get_balanced_facts()
print(f"\nTraining on {len(facts)} balanced examples (50/50 right/wrong)...")

EPOCHS = 15
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    errors_seen = 0
    for f in facts:
        prompt = f['prompt']
        correct = f['answer']

        logits = llm.get_logits(prompt)
        probs = torch.softmax(logits, dim=-1)
        llm_result = llm.predict_next_token(prompt, top_k=1)

        predicted = llm_result['top_tokens'][0]
        pred_prob = llm_result['top_probs'][0]

        is_wrong = predicted.strip().lower() != correct.strip().lower()

        if correct in llm_result.get('top_tokens', []):
            correct_prob = llm_result['top_probs'][llm_result['top_tokens'].index(correct)]
        else:
            full = probs.cpu().numpy()
            ids = llm.tokenizer.encode(correct)
            correct_prob = float(full[ids[0]]) if ids else 0.0

        error_mag = 1.0 if is_wrong else 0.0
        error_dir = (pred_prob - correct_prob) if is_wrong else 0.0

        loss = pop.train_step(logits.unsqueeze(0), probs.unsqueeze(0), error_mag, pred_prob, error_dir)
        epoch_loss += loss['loss']
        if is_wrong:
            errors_seen += 1

    avg = epoch_loss / len(facts)
    if (epoch + 1) % 3 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg:.4f} | errors: {errors_seen}/{len(facts)}")

print("✓ Training complete!")

# ── Test: curated mix with known answers (held-out — not in training_data.py) ──
test_cases = [
    # Easy (LLM likely correct)
    {"prompt": "She opened the", "answer": " door"},
    {"prompt": "He picked up the", "answer": " phone"},
    {"prompt": "The dog ran", "answer": " away"},
    {"prompt": "I have a", "answer": " lot"},
    {"prompt": "She is very", "answer": " good"},
    # Hard (LLM likely wrong) — held-out, not in training_data.py
    {"prompt": "The inventor of the telephone was", "answer": " Alexander"},
    {"prompt": "Mount Everest is in", "answer": " Nepal"},
    {"prompt": "The internet was invented in", "answer": " 1969"},
    {"prompt": "Photosynthesis converts sunlight into", "answer": " energy"},
    {"prompt": "DNA stands for deoxyribonucleic", "answer": " acid"},
    {"prompt": "The Roman Empire fell in the year", "answer": " 476"},
    {"prompt": "The French Revolution began in", "answer": " 1789"},
    {"prompt": "The printing press was invented by", "answer": " Gutenberg"},
    {"prompt": "An even number is divisible by", "answer": " two"},
    {"prompt": "E equals mc", "answer": " squared"},
]

print("\n" + "="*60)
print("ACTIVE MODE — PoP corrects when error > 0.5")
print("="*60)

debugger_active = PoPDebugger(verbose=True)

for tc in test_cases:
    prompt = tc['prompt']
    correct = tc['answer']

    llm_result = llm.predict_next_token(prompt, top_k=20)
    top_token = llm_result['top_tokens'][0]
    top_prob = llm_result['top_probs'][0]

    logits = llm.get_logits(prompt)
    probs = torch.softmax(logits, dim=-1)
    pop_result = pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))

    should_correct = pop_result['error_magnitude'] > 0.5

    if should_correct:
        alt_tokens = llm_result['top_tokens'][1:20]
        alt_probs = llm_result['top_probs'][1:20]
        best_idx = int(np.argmax(alt_probs))
        final_token = alt_tokens[best_idx]
        final_prob = alt_probs[best_idx]
        decision = 'CORRECTED'
    else:
        final_token = top_token
        final_prob = top_prob
        decision = 'TRUST_LLM'

    debugger_active.log_prediction(
        input_text=prompt,
        llm_token=top_token,
        llm_prob=top_prob,
        llm_top5=[{'token': t, 'prob': p} for t, p in zip(llm_result['top_tokens'][:5], llm_result['top_probs'][:5])],
        pop_error_magnitude=pop_result['error_magnitude'],
        pop_confidence=pop_result['confidence'],
        pop_direction=pop_result['error_direction'],
        decision=decision,
        final_token=final_token,
        final_prob=final_prob,
        correct_token=correct
    )

debugger_active.print_summary()

# Save
debugger_active.to_json('/root/.openclaw/workspace-main/pop-repo/pop_debug_log.json')
print("✓ Debug log saved!")
