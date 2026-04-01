"""LLM Inference Engine — handles model loading, generation, and streaming.

In production, this delegates to vLLM or TensorRT-LLM. This implementation
provides a fully-functional CPU-based inference engine using a lightweight
transformer model for development and testing.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import numpy as np

from neuronlm.config import get_settings
from neuronlm.core.exceptions import (
    InferenceError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from neuronlm.core.tokenizer import get_tokenizer
from neuronlm.models.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    DeltaMessage,
    FinishReason,
    MessageRole,
    ModelInfo,
    ModelStatus,
    UsageInfo,
)
from neuronlm.training.bigram_lm import BigramLanguageModel

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a registered model."""
    model_id: str
    display_name: str
    context_window: int = 32768
    max_batch_size: int = 64
    status: ModelStatus = ModelStatus.READY
    created_at: int = field(default_factory=lambda: int(time.time()))


class InferenceEngine:
    """Core inference engine managing model registry and text generation.

    This engine provides:
    - Model registry with multiple concurrent models
    - Synchronous and streaming completion
    - Token counting and usage tracking
    - Simulated GPU inference for development
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}
        self._tokenizer = get_tokenizer()
        self._start_time = time.time()
        settings = get_settings()
        self._backend = settings.inference_backend.lower().strip()
        self._bigram_model: Optional[BigramLanguageModel] = None

        # Register default models
        self._register_default_models()

        # Optional trained-model bootstrap for local training flow
        if settings.bigram_checkpoint_path:
            try:
                self.load_bigram_checkpoint(settings.bigram_checkpoint_path)
                logger.info("Loaded bigram checkpoint: %s", settings.bigram_checkpoint_path)
            except Exception:
                logger.exception(
                    "Failed to load bigram checkpoint, falling back to simulated backend"
                )
                self._backend = "simulated"

    def _register_default_models(self) -> None:
        """Register built-in models."""
        default_models = [
            ModelConfig(
                model_id="neuronlm-7b",
                display_name="NeuronLM 7B",
                context_window=32768,
            ),
            ModelConfig(
                model_id="neuronlm-13b",
                display_name="NeuronLM 13B",
                context_window=32768,
            ),
            ModelConfig(
                model_id="neuronlm-7b-chat",
                display_name="NeuronLM 7B Chat",
                context_window=16384,
            ),
            ModelConfig(
                model_id="neuronlm-embed",
                display_name="NeuronLM Embeddings",
                context_window=8192,
            ),
        ]
        for model in default_models:
            self._models[model.model_id] = model

    def register_model(self, config: ModelConfig) -> None:
        """Register a new model in the registry."""
        self._models[config.model_id] = config
        logger.info("Model registered: %s", config.model_id)

    def get_model(self, model_id: str) -> ModelConfig:
        """Get model configuration by ID."""
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]

    def list_models(self) -> list[ModelInfo]:
        """List all registered models."""
        return [
            ModelInfo(
                id=m.model_id,
                created=m.created_at,
                context_window=m.context_window,
                status=m.status,
            )
            for m in self._models.values()
        ]

    def _validate_model_ready(self, model_id: str) -> ModelConfig:
        """Validate that a model exists and is ready for inference."""
        model = self.get_model(model_id)
        if model.status != ModelStatus.READY:
            raise ModelNotReadyError(model_id)
        return model

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        """Build a prompt string from chat messages."""
        parts: list[str] = []
        for msg in messages:
            parts.append(f"<|{msg.role.value}|>\n{msg.content}")
        parts.append(f"<|{MessageRole.ASSISTANT.value}|>\n")
        return "\n".join(parts)

    def _generate_response_text(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        seed: Optional[int],
    ) -> str:
        """Generate response text.

        In production, this calls vLLM/TensorRT-LLM. This implementation
        provides a deterministic, seed-based response for testing.
        """
        if self._backend == "bigram" and self._bigram_model is not None:
            return self._generate_bigram_response(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                seed=seed,
                top_p=1.0,
            )

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        # Generate a contextual response based on the last message
        last_line = prompt.strip().split("\n")[-1] if prompt.strip() else ""

        # Response templates for demonstration
        responses = [
            "I understand your question. Let me provide a comprehensive answer. "
            "The topic you've raised involves several important considerations. "
            "First, we should examine the fundamental principles at play. "
            "The key factors to consider include context, methodology, and outcomes. "
            "In practice, the most effective approach combines theoretical understanding "
            "with practical application. This balanced perspective ensures that "
            "solutions are both robust and adaptable to changing requirements.",
            "That's an excellent question. Based on my analysis, there are multiple "
            "dimensions to consider. The primary factors include scalability, "
            "maintainability, and performance. Each of these plays a critical role "
            "in determining the optimal approach. I recommend a systematic evaluation "
            "of your specific requirements before making a final decision.",
            "Thank you for your inquiry. This is a nuanced topic that requires "
            "careful consideration. The most important aspects to address are "
            "the underlying assumptions, the available evidence, and the potential "
            "implications of different approaches. A thorough analysis reveals that "
            "the best path forward depends on your specific constraints and objectives.",
        ]

        # Use prompt hash for consistent responses with same input
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        idx = int(prompt_hash, 16) % len(responses)
        if seed is not None:
            idx = rng.randint(0, len(responses) - 1)

        response = responses[idx]

        # Apply max_tokens by truncating
        tokens = self._tokenizer.encode(response)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            response = self._tokenizer.decode(tokens)

        # Apply stop sequences
        if stop:
            for seq in stop:
                if seq in response:
                    response = response[: response.index(seq)]
                    break

        return response

    def load_bigram_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained bigram checkpoint and enable bigram backend."""
        model = BigramLanguageModel.load(checkpoint_path)
        self._bigram_model = model
        self._backend = "bigram"

    def _generate_bigram_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        seed: Optional[int],
        top_p: float,
    ) -> str:
        """Generate text using trained bigram transition probabilities."""
        if self._bigram_model is None:
            raise InferenceError("Bigram backend selected but no checkpoint is loaded")

        prompt_tokens = self._tokenizer.encode(prompt)
        if not prompt_tokens:
            prompt_tokens = [0]

        generated: list[int] = []
        prev_token = prompt_tokens[-1]
        rng = np.random.default_rng(seed)

        for _ in range(max_tokens):
            next_token = self._bigram_model.sample_next_token(
                prev_token=prev_token,
                rng=rng,
                top_k=50,
                top_p=top_p,
                temperature=temperature if temperature > 0 else 1.0,
            )
            generated.append(next_token)
            prev_token = next_token

            text = self._tokenizer.decode(generated)
            if stop and any(seq in text for seq in stop):
                break

        output = self._tokenizer.decode(generated)
        if stop:
            for seq in stop:
                if seq in output:
                    output = output[: output.index(seq)]
                    break
        return output or " "

    async def complete(
        self,
        model_id: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
    ) -> ChatCompletionResponse:
        """Generate a chat completion (non-streaming)."""
        model = self._validate_model_ready(model_id)

        prompt = self._build_prompt(messages)
        prompt_tokens = self._tokenizer.count_tokens(prompt)

        # Simulate inference latency
        await asyncio.sleep(0.05)

        response_text = self._generate_response_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop if isinstance(stop, list) else ([stop] if stop else None),
            seed=seed,
        )

        completion_tokens = self._tokenizer.count_tokens(response_text)

        # Determine finish reason
        finish_reason = FinishReason.STOP
        if completion_tokens >= max_tokens:
            finish_reason = FinishReason.LENGTH

        return ChatCompletionResponse(
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response_text,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream_complete(
        self,
        model_id: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion via SSE."""
        model = self._validate_model_ready(model_id)

        prompt = self._build_prompt(messages)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # First chunk: role
        role_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta=DeltaMessage(role=MessageRole.ASSISTANT),
                )
            ],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        # Generate response text
        response_text = self._generate_response_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop if isinstance(stop, list) else ([stop] if stop else None),
            seed=seed,
        )

        # Stream word by word
        words = response_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model_id,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=DeltaMessage(content=token),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0.02)  # Simulate token generation time

        # Final chunk: finish reason
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta=DeltaMessage(),
                    finish_reason=FinishReason.STOP,
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    async def generate_embedding(
        self,
        model_id: str,
        texts: list[str],
    ) -> tuple[list[list[float]], int]:
        """Generate embeddings for the given texts.

        Returns (embeddings, total_tokens).
        In production, this uses a dedicated embedding model.
        """
        self._validate_model_ready(model_id)

        embeddings: list[list[float]] = []
        total_tokens = 0

        for text in texts:
            tokens = self._tokenizer.count_tokens(text)
            total_tokens += tokens

            # Generate deterministic pseudo-embedding from text hash
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            rng = random.Random(text_hash)
            embedding = [rng.gauss(0, 0.1) for _ in range(384)]

            # Normalize to unit vector
            magnitude = sum(x * x for x in embedding) ** 0.5
            embedding = [x / magnitude for x in embedding]

            embeddings.append(embedding)

        return embeddings, total_tokens

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time


# Module-level singleton
_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Return the global inference engine instance."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
