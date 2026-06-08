# CI Workflow Diagnosis - PoP Project

## Problem
PR #16 targets `experiment/nli-attention-analysis` but CI workflows only trigger on `main`:
- `.github/workflows/ci.yml` — triggers: `push/pull_request branches: [main]`
- `.github/workflows/meta-ensemble.yml` — triggers: `push/pull_request branches: [main]`

## Evidence
- `gh pr checks 16` returns empty `[]`
- PR merge state: `CLEAN` (mergeable but unvalidated)
- No recent commits since May 2, 2026

## Solution Options
1. **Merge PR #16 into `main`** — Will trigger full CI validation
2. **Modify workflows** — Add `experiment/nli-attention-analysis` to branch list if it's intentional target