# Spec 06 — Evaluation (Ragas, CI-gated)

## Goal
Measure retrieval and answer quality on a fixed golden set so component swaps (embedder dimension, reranker choice, top-k) are decided by numbers, not vibes.

## Module
`eval/`

## Interface
```python
# eval/run_eval.py
def run_eval(dataset_path: str) -> EvalReport: ...   # prints + writes JSON, exits non-zero on regression
```

Golden dataset item:
```json
{
  "question": "...",
  "ground_truth": "...",
  "expected_source_ids": ["doc_12", "doc_5"]
}
```

## Metrics
- **Retrieval:** recall@k and NDCG@k against `expected_source_ids` — measured with and without the reranker, so you can see the reranker's actual lift.
- **Answer (Ragas):** faithfulness (no unsupported claims), answer relevance, context precision, context recall.
- **Abstention:** on a held-out set of out-of-corpus questions, fraction correctly abstained.

## Constraints & decisions
- **Retriever recall is the gate.** A reranker can lift NDCG without lifting answer faithfulness — track both; if retrieval recall@100 is low, fix retrieval before touching the reranker.
- **Commit the eval to CI.** A PR that drops faithfulness or recall below the committed baseline fails. Store baselines in `eval/baselines.json`.
- **Small but real.** 30–50 hand-written Q/A with known sources beats a large noisy set. Include exact-identifier questions and at least a few cross-modal ones (answer lives in an image/audio chunk).
- **Reranker bake-off lives here.** Run Cohere Rerank 4 vs Zerank 2 vs Voyage 2.5 against this set; the winner becomes the default in spec 04.

## Dependencies
`ragas`, `datasets`. Uses the real pipeline end-to-end (or a recorded fixture corpus).

## Acceptance criteria
- `run_eval.py` produces a JSON report with all metrics above and a pass/fail against baselines.
- Toggling the reranker off changes the reported retrieval-lift numbers.
- CI fails on a synthetic regression (e.g. faithfulness baseline set artificially high).

## Claude Code notes
Build the harness before the reranker bake-off — you need the scoreboard to pick the reranker. Seed `eval/datasets/` with ~10 items to start; expand later.
