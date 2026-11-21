# Spec 05 — Generation (grounded synthesis + citations)

## Goal
Turn the reranked top-k into a grounded answer with source citations, or an explicit abstention when the context does not support an answer.

## Module
`backend/app/generation/`

## Interface
```python
# schemas.py
class Citation(BaseModel):
    chunk_id: str
    source_id: str
    locator: str          # "p. 4", "12:30–13:10", bbox, char span
    modality: str

class Answer(BaseModel):
    text: str
    citations: list[Citation]
    used_chunk_ids: list[str]
    abstained: bool

# generation/generate.py
def generate(query: str, items: list[RetrievedItem]) -> Answer: ...
def generate_stream(query: str, items: list[RetrievedItem]) -> Iterator[str]: ...
```

## Constraints & decisions
- **Model:** current Gemini Pro tier (Gemini 3.x). Verify the exact model ID at `ai.google.dev`; Flash tier as a config-flag fallback for latency/cost.
- **Grounded or abstain.** The system prompt instructs: answer only from the provided context, cite the chunk backing each claim, and if the context is insufficient say so plainly (`abstained = True`). No outside knowledge, no filling gaps.
- **Citations are mandatory and machine-checkable.** Every claim maps to a `chunk_id` that was actually in `items`. Reject/repair answers that cite chunk IDs not in the input.
- **Context assembly:** pass the reranked chunks with their locators so the model can reference them; include the text surrogate for media chunks plus a note of the modality and source.
- **Streaming:** `/query/stream` streams tokens; citations are resolved and attached at the end from `used_chunk_ids`.
- **Determinism knobs** in `config.py`: temperature (low for factual domains), max output tokens.

## Dependencies
`google-genai`, Langfuse SDK for tracing the full retrieval→rerank→generation span.

## Acceptance criteria
- An answerable query returns non-empty `text`, `abstained = False`, and ≥1 citation whose `chunk_id` is in the input set.
- A query with no supporting context returns `abstained = True` and does not fabricate.
- Every `chunk_id` in `citations` also appears in `used_chunk_ids` and in the input `items`.
- The full pipeline emits one Langfuse trace with child spans for retrieve, rerank, generate.

## Claude Code notes
Write the citation-integrity check as a hard post-processing step, not a hope about model behavior. Test abstention with an out-of-corpus question.
