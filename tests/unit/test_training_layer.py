"""Unit tests for the lightweight training layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuronlm.models.schemas import ChatMessage, MessageRole
from neuronlm.services.inference_engine import InferenceEngine
from neuronlm.training.bigram_lm import BigramLanguageModel
from neuronlm.training.trainer import BigramTrainer, BigramTrainingConfig


class TestBigramTraining:
    """Training and checkpoint tests."""

    def test_train_and_save_checkpoint(self, tmp_path: Path):
        trainer = BigramTrainer()
        checkpoint = tmp_path / "bigram_model.npz"
        config = BigramTrainingConfig(
            checkpoint_path=str(checkpoint),
            min_tokens=4,
        )
        out_path = trainer.train_from_texts(
            texts=["hello world hello ai", "ai helps world"],
            config=config,
        )

        assert Path(out_path).exists()

        model = BigramLanguageModel.load(out_path)
        assert model.vocab_size > 0
        assert len(model.transitions) > 0

    def test_training_requires_enough_tokens(self, tmp_path: Path):
        trainer = BigramTrainer()
        config = BigramTrainingConfig(
            checkpoint_path=str(tmp_path / "tiny.npz"),
            min_tokens=1000,
        )
        with pytest.raises(ValueError):
            trainer.train_from_texts(texts=["tiny corpus"], config=config)


class TestInferenceWithTrainedCheckpoint:
    """Inference should use trained backend when checkpoint is loaded."""

    @pytest.mark.asyncio
    async def test_complete_works_after_loading_bigram_checkpoint(self, tmp_path: Path):
        trainer = BigramTrainer()
        checkpoint = tmp_path / "trained_bigram.npz"
        config = BigramTrainingConfig(checkpoint_path=str(checkpoint), min_tokens=4)
        trainer.train_from_texts(
            texts=[
                "machine learning is useful",
                "learning systems need data",
                "data drives machine learning",
            ],
            config=config,
        )

        engine = InferenceEngine()
        engine.load_bigram_checkpoint(str(checkpoint))

        response = await engine.complete(
            model_id="neuronlm-7b",
            messages=[ChatMessage(role=MessageRole.USER, content="Tell me about learning")],
            max_tokens=20,
            seed=123,
        )

        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert len(text.strip()) > 0
