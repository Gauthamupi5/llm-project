"""CLI for training the lightweight bigram backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuronlm.training.trainer import BigramTrainer, BigramTrainingConfig


def _read_corpus(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    text = file_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Corpus file is empty")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NeuronLM lightweight bigram backend")
    parser.add_argument("--corpus", required=True, help="Path to text corpus file")
    parser.add_argument("--checkpoint", required=True, help="Output .npz checkpoint path")
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--min-tokens", type=int, default=32)
    args = parser.parse_args()

    texts = _read_corpus(args.corpus)
    trainer = BigramTrainer()
    config = BigramTrainingConfig(
        checkpoint_path=args.checkpoint,
        smoothing=args.smoothing,
        min_tokens=args.min_tokens,
    )
    checkpoint = trainer.train_from_texts(texts=texts, config=config)
    print(f"Trained checkpoint saved to: {checkpoint}")


if __name__ == "__main__":
    main()
