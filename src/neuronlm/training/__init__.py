"""Training utilities for NeuronLM.

This package provides a lightweight trainable language model backend
used for local development and integration testing.
"""

from neuronlm.training.bigram_lm import BigramLanguageModel
from neuronlm.training.trainer import BigramTrainingConfig, BigramTrainer

__all__ = ["BigramLanguageModel", "BigramTrainingConfig", "BigramTrainer"]
