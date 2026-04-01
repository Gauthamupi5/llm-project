"""Simple trainable bigram language model.

This model is intentionally lightweight and CPU-friendly. It is not a
transformer, but it provides a true training loop and probabilistic
next-token generation path for local development.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BigramLanguageModel:
    """Stores transition probabilities for token->next-token generation."""

    transitions: dict[int, tuple[np.ndarray, np.ndarray]]
    fallback_next_ids: np.ndarray
    fallback_probs: np.ndarray
    vocab_size: int

    @classmethod
    def fit(
        cls,
        token_ids: list[int],
        vocab_size: int,
        smoothing: float = 1.0,
    ) -> "BigramLanguageModel":
        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if len(token_ids) < 2:
            raise ValueError("Need at least 2 tokens to train bigram model")

        transition_counts: dict[int, dict[int, float]] = {}
        unigram_next_counts: dict[int, float] = {}

        for i in range(len(token_ids) - 1):
            prev_tok = int(token_ids[i])
            next_tok = int(token_ids[i + 1])
            if 0 <= prev_tok < vocab_size and 0 <= next_tok < vocab_size:
                row = transition_counts.setdefault(prev_tok, {})
                row[next_tok] = row.get(next_tok, 0.0) + 1.0
                unigram_next_counts[next_tok] = unigram_next_counts.get(next_tok, 0.0) + 1.0

        transitions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for prev_tok, row in transition_counts.items():
            next_ids = np.array(list(row.keys()), dtype=np.int32)
            counts = np.array(list(row.values()), dtype=np.float64)
            probs = (counts + smoothing) / (counts.sum() + (smoothing * len(counts)))
            transitions[prev_tok] = (next_ids, probs)

        fallback_next_ids = np.array(list(unigram_next_counts.keys()), dtype=np.int32)
        fallback_counts = np.array(list(unigram_next_counts.values()), dtype=np.float64)
        fallback_probs = (fallback_counts + smoothing) / (
            fallback_counts.sum() + (smoothing * len(fallback_counts))
        )

        return cls(
            transitions=transitions,
            fallback_next_ids=fallback_next_ids,
            fallback_probs=fallback_probs,
            vocab_size=vocab_size,
        )

    def save(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        prev_ids = np.array(sorted(self.transitions.keys()), dtype=np.int32)
        offsets = [0]
        flat_next: list[int] = []
        flat_probs: list[float] = []

        for prev in prev_ids:
            next_ids, probs = self.transitions[int(prev)]
            flat_next.extend(next_ids.tolist())
            flat_probs.extend(probs.tolist())
            offsets.append(len(flat_next))

        np.savez_compressed(
            path,
            prev_ids=prev_ids,
            offsets=np.array(offsets, dtype=np.int64),
            flat_next=np.array(flat_next, dtype=np.int32),
            flat_probs=np.array(flat_probs, dtype=np.float32),
            fallback_next_ids=self.fallback_next_ids.astype(np.int32),
            fallback_probs=self.fallback_probs.astype(np.float32),
            vocab_size=np.array([self.vocab_size], dtype=np.int64),
        )

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> "BigramLanguageModel":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        data = np.load(path, allow_pickle=False)

        prev_ids = np.asarray(data["prev_ids"], dtype=np.int32)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        flat_next = np.asarray(data["flat_next"], dtype=np.int32)
        flat_probs = np.asarray(data["flat_probs"], dtype=np.float64)

        transitions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, prev in enumerate(prev_ids):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            transitions[int(prev)] = (flat_next[start:end], flat_probs[start:end])

        fallback_next_ids = np.asarray(data["fallback_next_ids"], dtype=np.int32)
        fallback_probs = np.asarray(data["fallback_probs"], dtype=np.float64)
        vocab_size = int(np.asarray(data["vocab_size"])[0])
        return cls(
            transitions=transitions,
            fallback_next_ids=fallback_next_ids,
            fallback_probs=fallback_probs,
            vocab_size=vocab_size,
        )

    def sample_next_token(
        self,
        prev_token: int,
        rng: np.random.Generator,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> int:
        if not 0 <= prev_token < self.vocab_size:
            prev_token = 0

        next_ids, base_probs = self.transitions.get(
            prev_token,
            (self.fallback_next_ids, self.fallback_probs),
        )
        probs = base_probs.copy()

        if temperature <= 0:
            temperature = 1e-6
        logits = np.log(np.clip(probs, 1e-12, None)) / temperature
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()

        if top_k is not None and top_k > 0 and top_k < len(probs):
            top_idx = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_idx] = probs[top_idx]
            probs = mask / mask.sum()

        if 0 < top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumsum, top_p, side="left") + 1
            keep = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask / mask.sum()

        sampled_idx = int(rng.choice(len(next_ids), p=probs))
        return int(next_ids[sampled_idx])
