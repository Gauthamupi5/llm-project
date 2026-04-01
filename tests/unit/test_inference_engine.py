"""Unit tests for the inference engine."""

import pytest

from neuronlm.core.exceptions import ModelNotFoundError, ModelNotReadyError
from neuronlm.models.schemas import ChatMessage, MessageRole, ModelStatus
from neuronlm.services.inference_engine import InferenceEngine, ModelConfig


class TestModelRegistry:
    """Model registry tests."""

    def test_default_models_registered(self, inference_engine: InferenceEngine):
        """Default models are registered on initialization."""
        models = inference_engine.list_models()
        model_ids = [m.id for m in models]
        assert "neuronlm-7b" in model_ids
        assert "neuronlm-13b" in model_ids
        assert "neuronlm-embed" in model_ids

    def test_get_existing_model(self, inference_engine: InferenceEngine):
        """UT-VAL-012 (inverse): Existing model is returned."""
        model = inference_engine.get_model("neuronlm-7b")
        assert model.model_id == "neuronlm-7b"

    def test_get_nonexistent_model(self, inference_engine: InferenceEngine):
        """UT-VAL-012: Non-existent model raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            inference_engine.get_model("nonexistent-model")

    def test_register_custom_model(self, inference_engine: InferenceEngine):
        """Custom models can be registered."""
        config = ModelConfig(model_id="custom-model", display_name="Custom")
        inference_engine.register_model(config)
        model = inference_engine.get_model("custom-model")
        assert model.model_id == "custom-model"

    def test_not_ready_model_raises(self, inference_engine: InferenceEngine):
        """Model with non-ready status raises ModelNotReadyError."""
        config = ModelConfig(
            model_id="loading-model",
            display_name="Loading",
            status=ModelStatus.LOADING,
        )
        inference_engine.register_model(config)
        with pytest.raises(ModelNotReadyError):
            inference_engine._validate_model_ready("loading-model")


class TestInference:
    """Inference generation tests."""

    @pytest.mark.asyncio
    async def test_complete_returns_valid_response(self, inference_engine: InferenceEngine):
        """Non-streaming completion returns valid response."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hello, how are you?")]
        response = await inference_engine.complete(
            model_id="neuronlm-7b",
            messages=messages,
            max_tokens=100,
        )

        assert response.model == "neuronlm-7b"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == MessageRole.ASSISTANT
        assert len(response.choices[0].message.content) > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    @pytest.mark.asyncio
    async def test_complete_with_seed_is_deterministic(self, inference_engine: InferenceEngine):
        """CT-CHAT-014: Same seed produces same output."""
        messages = [ChatMessage(role=MessageRole.USER, content="Tell me about AI")]

        response1 = await inference_engine.complete(
            model_id="neuronlm-7b", messages=messages, seed=42, max_tokens=50
        )
        response2 = await inference_engine.complete(
            model_id="neuronlm-7b", messages=messages, seed=42, max_tokens=50
        )

        assert response1.choices[0].message.content == response2.choices[0].message.content

    @pytest.mark.asyncio
    async def test_complete_with_stop_sequence(self, inference_engine: InferenceEngine):
        """CT-CHAT-013: Stop sequence stops generation."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        response = await inference_engine.complete(
            model_id="neuronlm-7b",
            messages=messages,
            stop=["."],
            max_tokens=200,
        )
        content = response.choices[0].message.content
        # Content should not contain the stop sequence (except at the very end)
        assert "." not in content or content.endswith(".")

    @pytest.mark.asyncio
    async def test_complete_model_not_found(self, inference_engine: InferenceEngine):
        """Inference with unknown model raises error."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        with pytest.raises(ModelNotFoundError):
            await inference_engine.complete(model_id="bad-model", messages=messages)

    @pytest.mark.asyncio
    async def test_stream_complete_yields_chunks(self, inference_engine: InferenceEngine):
        """Streaming produces multiple chunks ending with [DONE]."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        chunks = []
        async for chunk in inference_engine.stream_complete(
            model_id="neuronlm-7b",
            messages=messages,
            max_tokens=50,
        ):
            chunks.append(chunk)

        assert len(chunks) > 2  # At least: role chunk, content chunks, final chunk, [DONE]
        assert chunks[-1].strip() == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_stream_first_chunk_has_role(self, inference_engine: InferenceEngine):
        """First streaming chunk contains the assistant role."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        chunks = []
        async for chunk in inference_engine.stream_complete(
            model_id="neuronlm-7b",
            messages=messages,
            max_tokens=50,
        ):
            chunks.append(chunk)

        assert '"role":"assistant"' in chunks[0] or '"role": "assistant"' in chunks[0]


class TestEmbeddings:
    """Embedding generation tests."""

    @pytest.mark.asyncio
    async def test_single_embedding(self, inference_engine: InferenceEngine):
        """CT-EMB-001: Single text embedding returns correct dimensions."""
        embeddings, tokens = await inference_engine.generate_embedding(
            model_id="neuronlm-embed",
            texts=["Hello world"],
        )
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_batch_embeddings(self, inference_engine: InferenceEngine):
        """CT-EMB-002: Batch embeddings return correct number."""
        embeddings, tokens = await inference_engine.generate_embedding(
            model_id="neuronlm-embed",
            texts=["Hello", "World", "Test"],
        )
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    @pytest.mark.asyncio
    async def test_embeddings_are_normalized(self, inference_engine: InferenceEngine):
        """Embeddings should be approximately unit vectors."""
        embeddings, _ = await inference_engine.generate_embedding(
            model_id="neuronlm-embed",
            texts=["Test normalization"],
        )
        magnitude = sum(x * x for x in embeddings[0]) ** 0.5
        assert abs(magnitude - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self, inference_engine: InferenceEngine):
        """Same text produces same embedding."""
        emb1, _ = await inference_engine.generate_embedding("neuronlm-embed", ["Hello"])
        emb2, _ = await inference_engine.generate_embedding("neuronlm-embed", ["Hello"])
        assert emb1[0] == emb2[0]

    @pytest.mark.asyncio
    async def test_embedding_model_not_found(self, inference_engine: InferenceEngine):
        """Unknown model raises error."""
        with pytest.raises(ModelNotFoundError):
            await inference_engine.generate_embedding("bad-model", ["Hello"])

    def test_uptime(self, inference_engine: InferenceEngine):
        """Engine tracks uptime."""
        assert inference_engine.uptime_seconds >= 0
