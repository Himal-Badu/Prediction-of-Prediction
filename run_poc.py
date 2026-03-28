"""Run the full PoP PoC end-to-end."""
import sys
sys.path.insert(0, '/root/.openclaw/workspace-main/pop-repo')

import torch
import numpy as np
from pop.core.llm_base import create_llm
from pop.core.pop_layer_llm import create_pop_llm
from pop.core.debugger import PoPDebugger
from pop.core.training_data import get_all_facts

# ── 1. Load LLM ──
print("Loading DistilGPT-2...")
llm = create_llm('distilgpt2', device='cpu')
print(f"✓ Model: {llm.model_name} | Vocab: {llm.vocab_size}\n")

# ── 2. Baseline: Untrained PoP ──
pop = create_pop_llm(vocab_size=llm.vocab_size, device='cpu')
print(f"✓ PoP layer created (untrained)\n")

# ── 3. Training ──
facts = get_all_facts()
print(f"Training on {len(facts)} examples...")

EPOCHS = 5
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    errors_found = 0

    for f in facts:
        prompt = f['prompt']
        correct = f['answer']

        logits = llm.get_logits(prompt)
        probs = torch.softmax(logits, dim=-1)
        llm_result = llm.predict_next_token(prompt, top_k=1)

        predicted = llm_result['top_tokens'][0]
        pred_prob = llm_result['top_probs'][0]

        is_wrong = predicted.strip().lower() != correct.strip().lower()

        # Get correct token prob
        if correct in llm_result.get('top_tokens', []):
            idx = llm_result['top_tokens'].index(correct)
            correct_prob = llm_result['top_probs'][idx]
        else:
            full_probs = probs.cpu().numpy()
            ids = llm.tokenizer.encode(correct)
            correct_prob = float(full_probs[ids[0]]) if ids else 0.0

        error_mag = 1.0 if is_wrong else 0.0
        error_dir = (pred_prob - correct_prob) if is_wrong else 0.0

        loss = pop.train_step(logits.unsqueeze(0), probs.unsqueeze(0), error_mag, pred_prob, error_dir)
        epoch_loss += loss['loss']
        if is_wrong:
            errors_found += 1

    avg_loss = epoch_loss / len(facts)
    print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f} | LLM errors: {errors_found}/{len(facts)}")

print("\n✓ Training complete!\n")

# ── 4. Test with Debugger (held-out prompts — none in training_data.py) ──
test_cases = [
    {'prompt': 'The inventor of the telephone was', 'answer': ' Alexander'},
    {'prompt': 'Mount Everest is in', 'answer': ' Nepal'},
    {'prompt': 'The internet was invented in', 'answer': ' 1969'},
    {'prompt': 'Photosynthesis converts sunlight into', 'answer': ' energy'},
    {'prompt': 'DNA stands for deoxyribonucleic', 'answer': ' acid'},
    {'prompt': 'The Roman Empire fell in the year', 'answer': ' 476'},
    {'prompt': 'The French Revolution began in', 'answer': ' 1789'},
    {'prompt': 'The printing press was invented by', 'answer': ' Gutenberg'},
    {'prompt': 'An even number is divisible by', 'answer': ' two'},
    {'prompt': 'E equals mc', 'answer': ' squared'},
    {'prompt': 'The Renaissance started in', 'answer': ' Italy'},
    {'prompt': 'Christopher Columbus sailed to America in', 'answer': ' 1492'},
    {'prompt': 'The Industrial Revolution began in', 'answer': ' England'},
    {'prompt': 'A prime number has exactly', 'answer': ' two'},
    {'prompt': 'The derivative of x squared is', 'answer': ' 2x'},
]

debugger = PoPDebugger(verbose=True)

for tc in test_cases:
    prompt = tc['prompt']
    correct = tc['answer']

    llm_result = llm.predict_next_token(prompt, top_k=5)
    top_token = llm_result['top_tokens'][0]
    top_prob = llm_result['top_probs'][0]

    logits = llm.get_logits(prompt)
    probs = torch.softmax(logits, dim=-1)
    pop_result = pop.predict(logits.unsqueeze(0), probs.unsqueeze(0))

    # Safety guard: PoP says error is HIGH → correct
    should_correct = pop_result['error_magnitude'] > 0.5

    if should_correct:
        final_token = llm_result['top_tokens'][1] if len(llm_result['top_tokens']) > 1 else top_token
        final_prob = llm_result['top_probs'][1] if len(llm_result['top_probs']) > 1 else top_prob
        decision = 'CORRECTED'
    else:
        final_token = top_token
        final_prob = top_prob
        decision = 'TRUST_LLM'

    debugger.log_prediction(
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

# ── 5. Summary ──
debugger.print_summary()

# ── 6. Missed & false corrections ──
missed = debugger.get_missed_corrections()
bad = debugger.get_false_corrections()

print(f"\n MISSED CORRECTIONS ({len(missed)}):")
for e in missed:
    print(f"  \"{e.input_text}\" → LLM said \"{e.llm_token}\" but answer was \"{e.correct_token}\"")

print(f"\n FALSE CORRECTIONS ({len(bad)}):")
for e in bad:
    print(f"  \"{e.input_text}\" → LLM had \"{e.llm_token}\" (correct!) but PoP changed to \"{e.final_token}\"")

# Save debug log
debugger.to_json('/root/.openclaw/workspace-main/pop-repo/pop_debug_log.json')
print("\n✓ Debug log saved!")
