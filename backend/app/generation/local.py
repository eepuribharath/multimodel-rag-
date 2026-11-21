"""Local Hugging Face transformers generator (no API key required)."""

import re
import threading
from typing import Iterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.generation.generate import (
    ABSTAIN_SENTINEL,
    ABSTAIN_TEXT,
    CHITCHAT_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    build_context,
    chitchat_answer,
    resolve_answer,
)
from app.schemas import Answer, RetrievedItem


def _history_block(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return ""
    transcript = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
    return f"Previous conversation (for context only, not a source):\n{transcript}\n\n"


LOCAL_SYSTEM_INSTRUCTION = """\
You are a helpful knowledge assistant. You answer questions using ONLY the numbered context blocks provided.
Rules:
1. For image or media blocks, their Visual Description and Title state what is depicted. Use this information to describe images and answer questions about them.
2. Every claim should cite the context block number inline as [n] (e.g. [1]).
3. Be direct, clear, and informative.
4. If none of the context blocks contain information relevant to the question, reply with: INSUFFICIENT_CONTEXT
"""

ONE_SHOT_EXAMPLE = """\
Context blocks:
[1] source: landscape.jpg | modality: image | locator: §0
Title: Mountain Lake
Visual Description: a calm lake surrounded by pine trees and mountains

Question: What does the image depict?
Answer: The image depicts a calm lake surrounded by pine trees and mountains [1].
"""


class LocalGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        temperature: float = 0.1,
        max_output_tokens: int = 512,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
        )

    def _format_prompt(
        self,
        query: str,
        items: list[RetrievedItem],
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        context = build_context(items)
        user_content = (
            f"{ONE_SHOT_EXAMPLE}\n\n"
            f"{_history_block(history)}"
            f"Context blocks:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a complete, helpful answer in English based ONLY on the context blocks above. "
            "Always cite the block number inline as [n] (e.g. [1]). "
            "If the context lacks sufficient information, reply with: INSUFFICIENT_CONTEXT"
        )
        messages = [
            {"role": "system", "content": LOCAL_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        query: str,
        items: list[RetrievedItem],
        history: list[tuple[str, str]] | None = None,
    ) -> Answer:
        if not items:
            return Answer(
                text=ABSTAIN_TEXT,
                citations=[],
                used_chunk_ids=[],
                abstained=True,
                kind="abstained",
            )
        full_text = "".join(self.generate_stream(query, items, history))
        if (
            items
            and not re.search(r"\[\d+\]", full_text)
            and "INSUFFICIENT_CONTEXT" not in full_text.upper()
            and not full_text.startswith(ABSTAIN_TEXT)
        ):
            full_text = f"{full_text.strip()} [1]"
        return resolve_answer(full_text, items)

    def generate_stream(
        self,
        query: str,
        items: list[RetrievedItem],
        history: list[tuple[str, str]] | None = None,
    ) -> Iterator[str]:
        if not items:
            yield ABSTAIN_TEXT
            return

        prompt = self._format_prompt(query, items, history)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        do_sample = self.temperature > 0.05
        gen_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=self.max_output_tokens,
            temperature=max(self.temperature, 0.01) if do_sample else None,
            do_sample=do_sample,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        saw_any = False
        saw_citation = False
        accumulated = []
        for delta in streamer:
            if delta:
                saw_any = True
                accumulated.append(delta)
                if "[" in delta:
                    saw_citation = True
                yield delta
        thread.join()

        full_streamed = "".join(accumulated).strip().upper()
        saw_abstain = "INSUFFICIENT_CONTEXT" in full_streamed

        # If model finished without citation, yield citation marker for the top evidence block (unless abstaining)
        if saw_any and not saw_citation and not saw_abstain and items:
            yield " [1]"

    def generate_chitchat_stream(
        self, query: str, history: list[tuple[str, str]] | None = None
    ) -> Iterator[str]:
        messages = [
            {"role": "system", "content": CHITCHAT_INSTRUCTION},
            {"role": "user", "content": query},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for delta in streamer:
            if delta:
                yield delta
        thread.join()

    def generate_chitchat(
        self, query: str, history: list[tuple[str, str]] | None = None
    ) -> Answer:
        return chitchat_answer("".join(self.generate_chitchat_stream(query, history)))
