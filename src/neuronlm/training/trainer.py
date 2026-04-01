"""Training orchestration for the lightweight bigram backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neuronlm.core.tokenizer import TokenizerService
from neuronlm.training.bigram_lm import BigramLanguageModel


@dataclass
class BigramTrainingConfig:
    """Configuration for bigram model training."""

    checkpoint_path: str
    smoothing: float = 1.0
    min_tokens: int = 32


class BigramTrainer:
    """Trains and saves a bigram language model from text corpora."""

    def __init__(self, tokenizer: TokenizerService | None = None) -> None:
        self._tokenizer = tokenizer or TokenizerService()

    def train_from_texts(
        self,
        texts: list[str],
        config: BigramTrainingConfig,
    ) -> str:
        if not texts:
            raise ValueError("texts must not be empty")

        combined_text = "\n".join(t for t in texts if t.strip())
        token_ids = self._tokenizer.encode(combined_text)

        if len(token_ids) < config.min_tokens:
            raise ValueError(
                f"Not enough training tokens: got {len(token_ids)}, "
                f"need at least {config.min_tokens}"
            )

        vocab_size = self._tokenizer.vocab_size
        model = BigramLanguageModel.fit(
            token_ids=token_ids,
            vocab_size=vocab_size,
            smoothing=config.smoothing,
        )
        model.save(config.checkpoint_path)
        return str(Path(config.checkpoint_path).resolve())
