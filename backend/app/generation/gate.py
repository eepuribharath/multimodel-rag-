"""Pre-retrieval query gate: chitchat detection + follow-up rewriting.

Three tiers, cheapest first:
1. regex for unmistakably social messages → chitchat, no API call
2. no history and not social → search the question as-is, no API call
3. history present → one small flash call that either flags chitchat or
   rewrites the follow-up into a standalone query (resolves "it", "that").
   Any failure falls back to searching the original question — the gate can
   only improve a query, never block one.
"""

import re
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai import types

CHITCHAT_SENTINEL = "CHITCHAT"

_SOCIAL_RE = re.compile(
    r"^(hi+|hiya|hello+|hey+|yo+|sup|howdy|good\s+(morning|afternoon|evening)|"
    r"thanks|thank\s+you|thx|ty|ok(ay)?|cool|nice|great|awesome|"
    r"bye|goodbye|see\s+ya|later|"
    r"how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|"
    r"who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|help)"
    r"[\s!?.,:;)…]*$",
    re.IGNORECASE,
)

_FUTURE_UPLOAD_RE = re.compile(
    r"\b(i\s+will|i'll|i\s+want\s+to|i'd\s+like\s+to|i\s+am\s+going\s+to|i\s+plan\s+to)\s+.*\b(attach|upload|send|provide|give)\b",
    re.IGNORECASE,
)
_HOWTO_UPLOAD_RE = re.compile(
    r"^(how\s+do\s+i|how\s+can\s+i)\s+.*\b(attach|upload|send|import)\b",
    re.IGNORECASE,
)
_GENERAL_CAPABILITY_RE = re.compile(
    r"^(can\s+you|could\s+you|will\s+you|are\s+you\s+able\s+to)\s+"
    r"(help(\s+me)?|explain(\s+to\s+me)?|read|analyze|summarize|assist)"
    r"(\s+(a|an|the|my)?\s*(pdf|file|document|doc|image|audio|video)s?(\s+files?)?)?"
    r"[\s!?.,:;)…]*$",
    re.IGNORECASE,
)

REWRITE_INSTRUCTION = f"""\
You prepare a user's chat message for document search. Given the previous
conversation and the new message, reply with ONLY one of:
- {CHITCHAT_SENTINEL} — if the message is a greeting, thanks, small talk, or a
  question about the assistant itself rather than a request for information.
- Otherwise: the message rewritten as a standalone search query, resolving
  pronouns and references from the conversation ("it", "that one", "them").
  If the message is already standalone, return it unchanged.
Never answer the message. Reply with the query or {CHITCHAT_SENTINEL} only.
"""


@dataclass
class GatePlan:
    kind: Literal["chitchat", "search"]
    standalone: str  # the query retrieval should run (search only)


def looks_social(message: str) -> bool:
    clean = message.strip()
    if len(clean) > 120:
        return False
    if _SOCIAL_RE.match(clean):
        return True
    if _FUTURE_UPLOAD_RE.search(clean):
        return True
    if _HOWTO_UPLOAD_RE.search(clean):
        return True
    if _GENERAL_CAPABILITY_RE.search(clean):
        return True
    return False


class QueryGate:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: genai.Client | None = None,
    ):
        self.model = model
        # no key and no client → tier 3 disabled; tiers 1–2 still work
        self._client = client or (genai.Client(api_key=api_key) if api_key else None)

    def plan(self, question: str, history: list[tuple[str, str]]) -> GatePlan:
        if looks_social(question):
            return GatePlan(kind="chitchat", standalone=question)
        if not history or self._client is None:
            return GatePlan(kind="search", standalone=question)
        try:
            transcript = "\n".join(
                f"User: {q}\nAssistant: {a}" for q, a in history
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=f"Previous conversation:\n{transcript}\n\nNew message: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=REWRITE_INSTRUCTION,
                    temperature=0.0,
                    max_output_tokens=100,
                ),
            )
            reply = (response.text or "").strip()
        except Exception:
            return GatePlan(kind="search", standalone=question)
        if not reply:
            return GatePlan(kind="search", standalone=question)
        if reply.upper().startswith(CHITCHAT_SENTINEL):
            return GatePlan(kind="chitchat", standalone=question)
        return GatePlan(kind="search", standalone=reply)
