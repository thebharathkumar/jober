"""LLM provider abstraction.

Two implementations ship:

* ``AnthropicProvider`` — the real thing, used when ``ANTHROPIC_API_KEY`` is set
  and the ``anthropic`` SDK is installed.
* ``ExtractiveProvider`` — a deterministic, offline fallback that answers by
  selecting the sentences from the retrieved evidence that best match the
  question. It is intentionally simple, but it makes the *entire* pipeline —
  including the eval harness — runnable with zero credentials and zero network.
  That means CI can measure the retrieval+grounding stack in isolation from the
  model, which is exactly what you want when a number has to be trustworthy.

Both satisfy the same ``complete(system, prompt, contexts)`` contract.
"""

from __future__ import annotations

import re
from typing import Protocol

from bus_factor.config import Settings

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9_]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str, contexts: list[str]) -> str: ...


class ExtractiveProvider:
    """Offline, deterministic. Answers from the evidence, never invents."""

    name = "extractive-offline"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def complete(self, system: str, prompt: str, contexts: list[str]) -> str:
        question = prompt
        q_words = _words(question)
        if not contexts or not q_words:
            return "I don't have enough recorded knowledge to answer that confidently."

        # Trust the retriever: answer from the single highest-ranked document
        # that yields usable content, rather than re-competing sentences across
        # all of them. Retrieval already ranked the right source first (often via
        # a strong title match the body doesn't echo), so drawing from a
        # lower-ranked neighbour just because a sentence shares a common word is
        # exactly the top-k pollution to avoid.
        for ctx in contexts:  # in retrieval-rank order
            picked = self._best_sentences(ctx, q_words)
            if picked:
                return " ".join(picked)
        return "I don't have enough recorded knowledge to answer that confidently."

    def _best_sentences(self, context: str, q_words: set[str]) -> list[str]:
        scored: list[tuple[float, str]] = []
        for sent in _SENT_RE.split(context):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            s_words = _words(sent)
            if not s_words:
                continue
            # Drop sentences that are almost entirely question words — those are
            # echoes of the prompt (e.g. a "Re: <question>" title), not answers.
            novelty_ratio = len(s_words - q_words) / len(s_words)
            if novelty_ratio < 0.3:
                continue
            # Rank by question overlap, but keep overlap-0 sentences in play (the
            # "+1") so the top document's informative lead is still returned even
            # when its wording doesn't lexically echo the question.
            overlap = len(q_words & s_words)
            scored.append(((overlap + 1) * (1 + novelty_ratio), sent))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen: list[str] = []
        seen: set[str] = set()
        for _score, sent in scored:
            if sent not in seen:
                seen.add(sent)
                chosen.append(sent)
            if len(chosen) >= self.max_sentences:
                break
        return chosen


class AnthropicProvider:
    """Grounded answerer backed by a Claude model."""

    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.model
        self._max_tokens = settings.max_tokens

    def complete(self, system: str, prompt: str, contexts: list[str]) -> str:
        evidence = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        user = (
            f"Question:\n{prompt}\n\n"
            f"Recorded knowledge (your only source of truth):\n{evidence}\n\n"
            "Answer the question using ONLY the recorded knowledge above. "
            "If it does not contain the answer, say so plainly. Do not invent."
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


def get_provider(settings: Settings) -> LLMProvider:
    """Pick the best available provider without ever failing hard."""
    if settings.has_llm:
        try:
            return AnthropicProvider(settings)
        except Exception:
            pass
    return ExtractiveProvider()
