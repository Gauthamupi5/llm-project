"""HuggingFace Transformers inference backend.

Wraps an AutoModelForCausalLM (default: distilgpt2) and exposes a
generate() method compatible with InferenceEngine's dispatch table.

On first call, the model is downloaded from the HuggingFace Hub and
cached in ~/.cache/huggingface/. Subsequent starts reuse the cache.

Supported models (any causal-LM on the Hub):
  distilgpt2   (~300 MB, fast on CPU, good for development)
  gpt2         (~500 MB)
  gpt2-medium  (~1.5 GB)
  Any other AutoModelForCausalLM model name.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TransformersBackend:
    """CPU-based text generation backend using HuggingFace Transformers.

    Model and tokenizer are loaded lazily on the first generate() call so
    that importing this module never triggers a download.
    """

    def __init__(self, model_name: str = "distilgpt2") -> None:
        self._model_name = model_name
        self._model = None        # AutoModelForCausalLM, loaded lazily
        self._hf_tokenizer = None  # HF AutoTokenizer (separate from tiktoken)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lazy_load(self) -> None:
        """Download and cache the model on first call."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "torch and transformers must be installed to use the "
                "transformers backend.  Run:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install transformers"
            ) from exc

        logger.info("Loading model '%s' — first run downloads from HuggingFace Hub …", self._model_name)

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            dtype=torch.float32,  # CPU — float32 is fastest
        )
        model.eval()

        # GPT-2 family has no pad token; use eos as pad so batching doesn't warn
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._hf_tokenizer = tokenizer
        self._model = model
        logger.info("Model '%s' loaded and ready.", self._model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate text that continues *prompt*.

        Returns only the newly generated tokens (not the prompt itself).

        Args:
            prompt: The full formatted prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. 0 → greedy decoding.
            top_p: Nucleus sampling probability mass.
            seed: RNG seed for reproducibility.
            stop: Stop sequences; generation is truncated at the first match.

        Returns:
            Generated text (prompt excluded).
        """
        import torch

        self._lazy_load()

        if seed is not None:
            torch.manual_seed(seed)

        # Truncate prompt to leave room for new tokens (model max is 1024 for GPT-2)
        model_max = getattr(self._hf_tokenizer, "model_max_length", 1024)
        max_prompt_tokens = max(1, model_max - max_new_tokens)

        inputs = self._hf_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        input_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0.0
        gen_temperature = max(temperature, 1e-4)  # avoid division-by-zero in logit scaling

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=gen_temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self._hf_tokenizer.eos_token_id,
                eos_token_id=self._hf_tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_token_ids = output_ids[0][input_len:]
        generated_text = self._hf_tokenizer.decode(
            new_token_ids, skip_special_tokens=True
        )

        # Apply stop sequences
        if stop:
            for seq in stop:
                if seq in generated_text:
                    generated_text = generated_text[: generated_text.index(seq)]
                    break

        return generated_text.strip() or "(no output)"

    @property
    def model_name(self) -> str:
        """Return the model name / identifier."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """True after the first generate() call has completed model loading."""
        return self._model is not None
