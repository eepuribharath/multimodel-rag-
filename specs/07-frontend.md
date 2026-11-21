# Spec 07 — Frontend (clean, minimal)

## Goal
A single-screen interface for asking questions and reading grounded, cited answers, with the retrieved evidence one glance away. Clean and minimal — one primary action, no chrome that isn't earning its place.

## Scope note
This spec defines **structure, state, data contracts, and interactions**. The **visual layer — layout polish, type scale, spacing, color, motion — is built with the Impeccable skill**, not prescribed here. Keep this spec functional; let Impeccable own the aesthetics.

## Stack
Vite + React + Tailwind. Talks to the FastAPI backend (`/query/stream`, `/ingest`).

## Screens / components
One screen, three regions:

1. **Query bar** (primary, centered until first query, then docked top)
   - Multimodal input: a text field plus an attach control (query can be text or media — the embedder supports it).
   - Single submit action. Enter to send.
2. **Answer panel**
   - Streams the answer token-by-token.
   - Inline citation markers (e.g. `[1]`, `[2]`) that map to evidence items; hovering/tapping one highlights the corresponding source.
   - Explicit **abstention state**: when `abstained = true`, show a plain "not enough in the sources to answer" message — never a fabricated answer styled as confident.
3. **Evidence panel** (collapsible, secondary)
   - One card per cited source: modality badge (text/image/audio/video/pdf), source title, locator ("p. 4", "12:30–13:10"), and a snippet/thumbnail/player appropriate to the modality.
   - Optional **retrieval inspector** toggle (dev/power feature): shows the candidate list pre- vs post-rerank with scores, for debugging retrieval quality. Off by default.

## Data contracts
Consume the backend shapes directly — do not reshape on the client:
- Stream: text deltas over SSE/chunked from `/query/stream`.
- On completion: `Answer { text, citations[], used_chunk_ids[], abstained }` (see spec 05). Each `Citation` has `{ chunk_id, source_id, locator, modality }` and resolves to an evidence card.

## States (design all of them)
`idle` · `retrieving` · `reranking` · `generating (streaming)` · `answered` · `abstained` · `empty result` · `error`. Show lightweight stage progress (retrieving → reranking → generating) rather than a single opaque spinner — it reflects the real pipeline and reads as fast.

## Interactions
- Citation ↔ evidence highlighting is bidirectional.
- Media evidence renders inline (image thumb, audio/video mini-player seeked to the cited timestamp).
- Keyboard: Enter submits, Esc clears, `/` focuses the query bar.

## Accessibility
- Every state has a text equivalent (screen-reader announces stage changes and abstention).
- Modality badges pair an icon with a text label, never color alone.
- Streaming region uses `aria-live="polite"`.

## Acceptance criteria
- Asking a question streams an answer with working, clickable citations that highlight evidence cards.
- An out-of-corpus question shows the abstention state, not a fake answer.
- A cited audio/video source plays from the cited timestamp.
- All eight states are reachable and visually distinct.

## Claude Code notes
Build against a mocked backend response first (fixture `Answer` + SSE stream) so the UI and states are complete before the real pipeline is wired. Then run the Impeccable skill over the finished structure for the visual pass.
