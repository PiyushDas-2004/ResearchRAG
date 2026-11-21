"""Deterministic offline stand-in for an LLM.

The mock reads the same prompts a real model receives (it parses the tagged sections built
by :mod:`researchrag.generation.prompts`) and answers extractively: it picks the passage
sentences with the best coverage of the question's content terms, cites them and quotes them
verbatim, and abstains when coverage is too low. It can also inject configurable failures -
timeouts, 429s, malformed JSON, fabricated quotes, uncited claims, numeric hallucinations and
obeying injected instructions - so that every defensive path can be exercised without keys.

Numbers produced with the mock measure the *pipeline* (retrieval, abstention gate, verifier,
guards), not language-model quality. Reports label them accordingly.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import AsyncIterator

from researchrag.config.pipeline import MockLLMConfig
from researchrag.errors import ProviderTimeoutError, RateLimitedError
from researchrag.generation.prompts import extract_tag, parse_passages
from researchrag.llm.base import LLMRequest, LLMResponse, StreamEvent, estimate_tokens
from researchrag.security.injection import scan_text
from researchrag.textutils import STOPWORDS, content_terms, coverage, numbers, split_sentences, words
from researchrag.verification.lexical import lexical_judge

_PRONOUN_RE = re.compile(r"\b(it|its|they|their|them|this|that|these|those|he|she|his|her)\b", re.I)
_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_STREAM_CHUNK = 24
_NO_EVIDENCE = "The passages do not contain enough information to answer the question."


class MockLLM:
    """``mock/mock-extractive-v1``."""

    provider = "mock"

    def __init__(self, config: MockLLMConfig | None = None, model: str = "mock-extractive-v1") -> None:
        self.config = config or MockLLMConfig()
        self.model = model
        self.calls = 0
        self._failures = 0

    # ----------------------------------------------------------------- failure injection
    def _maybe_fail(self, task: str) -> str | None:
        """Raise transport failures; return a content failure mode for this call (or None)."""
        self.calls += 1
        mode = self.config.failure_mode
        if mode == "none":
            return None
        if self.config.failure_count >= 0 and self._failures >= self.config.failure_count:
            return None
        if mode in {"timeout", "rate_limit"} or task == "answer":
            self._failures += 1
        if mode == "timeout":
            raise ProviderTimeoutError("mock timeout")
        if mode == "rate_limit":
            raise RateLimitedError("mock 429", retry_after=0.0)
        return mode if task == "answer" else None

    # ----------------------------------------------------------------- public API
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Produce a deterministic response for the request's task."""
        mode = self._maybe_fail(request.task)
        text = self._respond(request, mode)
        return LLMResponse(
            text=text,
            input_tokens=estimate_tokens(request.messages),
            output_tokens=max(1, len(text) // 4),
            model_id=f"mock/{self.model}",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """Stream the same text as :meth:`complete` in fixed-size chunks."""
        mode = self._maybe_fail(request.task)
        text = self._respond(request, mode)
        for i in range(0, len(text), _STREAM_CHUNK):
            yield StreamEvent(delta=text[i : i + _STREAM_CHUNK])
        yield StreamEvent(
            input_tokens=estimate_tokens(request.messages), output_tokens=max(1, len(text) // 4)
        )

    # ----------------------------------------------------------------- task handlers
    def _respond(self, request: LLMRequest, mode: str | None) -> str:
        user = "\n".join(m.content for m in request.messages if m.role == "user")
        task = request.task
        if task == "answer":
            return self._answer(user, mode)
        if task == "rewrite":
            return self._rewrite(user)
        if task == "condense":
            return self._condense(user)
        if task == "verify":
            return self._verify(user)
        if task == "rerank":
            return self._rerank(user)
        if task == "judge":
            return self._judge(user)
        return json.dumps({"error": f"mock has no handler for task {task}"})

    def _rng(self, key: str) -> random.Random:
        seed = int(hashlib.sha256(f"{self.config.seed}|{key}".encode()).hexdigest()[:12], 16)
        return random.Random(seed)  # noqa: S311 - deterministic test behaviour, not security

    def _answer(self, user: str, mode: str | None) -> str:
        question = extract_tag(user, "question")
        passages = parse_passages(user)
        q_terms = set(content_terms(question))
        scored: list[tuple[float, int, int, str]] = []
        for number, text in passages:
            for order, span in enumerate(split_sentences(text)):
                if span.text.startswith("[instruction-like text removed]") or scan_text(span.text).found:
                    continue
                s_terms = set(content_terms(span.text))
                cov = coverage(q_terms, s_terms)
                bonus = 0.05 if numbers(span.text) and _asks_number(question) else 0.0
                scored.append((cov + bonus, -number, -order, span.text))
        scored.sort(reverse=True)
        if mode == "follow_injection":
            injected = [t for _, t in passages if scan_text(t).found]
            if injected:
                tokens = scan_text(injected[0]).payload_tokens
                reply = tokens[0] if tokens else "As instructed by the document, I will ignore my rules."
                return json.dumps(
                    {"answer": reply, "claims": [], "quotes": [], "confidence": 0.9, "abstained": False}
                )
        best = scored[0][0] if scored else 0.0
        if best < self.config.min_term_coverage:
            return json.dumps(
                {
                    "answer": "",
                    "claims": [],
                    "quotes": [],
                    "confidence": round(best, 3),
                    "abstained": True,
                    "abstain_reason": _NO_EVIDENCE,
                }
            )
        chosen: list[tuple[int, str]] = []
        covered: set[str] = set()
        for score, neg_number, _, sentence in scored:
            if len(chosen) >= self.config.max_claims:
                break
            new_terms = (set(content_terms(sentence)) & q_terms) - covered
            if score >= max(self.config.min_term_coverage * 0.8, best * 0.6) and (not chosen or new_terms):
                chosen.append((-neg_number, sentence))
                covered |= set(content_terms(sentence)) & q_terms
        claims = [{"text": s, "citations": [n]} for n, s in chosen]
        quotes = [{"passage": n, "quote": s} for n, s in chosen]
        rng = self._rng(question)
        if self.config.hallucination_rate > 0 and rng.random() < self.config.hallucination_rate:
            claims[0]["text"] = _perturb_number(str(claims[0]["text"]), rng)
        if mode == "fabricated_quote":
            quotes[0]["quote"] = "The authors conclusively proved that " + str(quotes[0]["quote"]).lower()
        if mode == "uncited_claim":
            claims.append(
                {"text": "This result has been replicated in many independent studies.", "citations": []}
            )
        answer = " ".join(f"{c['text']}" + "".join(f" [{n}]" for n in c["citations"]) for c in claims)
        payload = {
            "answer": answer,
            "claims": claims,
            "quotes": quotes,
            "confidence": round(min(1.0, best), 3),
            "abstained": False,
            "abstain_reason": None,
        }
        text = json.dumps(payload)
        if mode == "malformed_json":
            return (
                "Sure! Here is the answer:\n```json\n"
                + text[:-1].replace('"confidence"', "'confidence'")
                + ",\n```"
            )
        return text

    def _rewrite(self, user: str) -> str:
        question = extract_tag(user, "query")
        terms = [w for w in words(question) if w not in STOPWORDS and len(w) > 2]
        variants = [" ".join(terms)] if terms else []
        stripped = re.sub(
            r"^(what|which|how|why|when|where|who)\b\s*(is|are|was|were|did|does|do)?\s*",
            "",
            question,
            flags=re.I,
        )
        if stripped and stripped != question:
            variants.append(stripped.rstrip("?"))
        return json.dumps({"queries": [v for v in variants if v]})

    def _condense(self, user: str) -> str:
        history = extract_tag(user, "history")
        followup = extract_tag(user, "followup").strip()
        user_turns = [
            line.split(":", 1)[1].strip() for line in history.splitlines() if line.startswith("user:")
        ]
        if not user_turns or (not _PRONOUN_RE.search(followup) and len(followup.split()) > 6):
            return json.dumps({"standalone_question": followup})
        previous = user_turns[-1]
        follow_terms = set(content_terms(followup))
        context = [w for w in words(previous) if w not in STOPWORDS and len(w) > 2 and w not in follow_terms]
        standalone = followup.rstrip("?") + " (regarding " + " ".join(dict.fromkeys(context)) + ")?"
        return json.dumps({"standalone_question": standalone})

    def _verify(self, user: str) -> str:
        result = lexical_judge(extract_tag(user, "claim"), extract_tag(user, "evidence"))
        return json.dumps({"verdict": result.verdict, "rationale": result.rationale})

    def _rerank(self, user: str) -> str:
        q_terms = set(content_terms(extract_tag(user, "query")))
        scores = [
            {"id": n, "score": round(10 * coverage(q_terms, set(content_terms(t))), 2)}
            for n, t in parse_passages(user)
        ]
        return json.dumps({"scores": scores})

    def _judge(self, user: str) -> str:
        gold = extract_tag(user, "gold")
        candidate = extract_tag(user, "candidate")
        gold_terms = set(content_terms(gold))
        cand_terms = set(content_terms(candidate))
        if not candidate.strip():
            score = 0.0
        else:
            overlap = len(gold_terms & cand_terms)
            precision = overlap / len(cand_terms) if cand_terms else 0.0
            recall = overlap / len(gold_terms) if gold_terms else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            gold_nums = numbers(gold)
            num_hit = len(gold_nums & numbers(candidate)) / len(gold_nums) if gold_nums else 1.0
            score = round(0.5 * min(1.0, recall * 1.25) + 0.25 * f1 + 0.25 * num_hit, 3)
        return json.dumps({"score": score, "rationale": "mock lexical rubric (term recall, F1, numbers)"})


def _asks_number(question: str) -> bool:
    return bool(
        re.search(
            r"\b(how (many|much|long|large)|what (percent|fraction|value|rate|number)|percentage)\b",
            question,
            re.I,
        )
    )


def _perturb_number(text: str, rng: random.Random) -> str:
    """Change one number in ``text`` (simulated numeric hallucination); append one if none."""
    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return text.rstrip(".") + f", an improvement of {rng.randint(11, 97)} percent."
    match = matches[rng.randrange(len(matches))]
    raw = match.group(1)
    new = f"{float(raw) * 1.37 + 1:.2f}".rstrip("0").rstrip(".") if "." in raw else str(int(raw) * 2 + 7)
    return text[: match.start(1)] + new + text[match.end(1) :]
