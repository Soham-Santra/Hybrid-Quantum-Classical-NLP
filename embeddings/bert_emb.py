"""Transformer-based text embeddings (BERT / mBERT / IndicBERT).

Loads a HuggingFace ``AutoModel`` and ``AutoTokenizer``, runs batched
inference with ``torch.no_grad()``, and returns dense numpy arrays.
Supports CLS-token pooling and mean-token pooling.
"""

import logging
from typing import List, Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TransformerEmbedder:
    """Encode texts with a pre-trained transformer model.

    The model and tokenizer are loaded lazily on the first call to
    :meth:`transform` so that construction is cheap and multiple
    embedders can be instantiated without downloading weights.

    Parameters
    ----------
    model_name : str
        HuggingFace Hub model identifier, e.g. ``"bert-base-uncased"``.
    pooling : ``"cls"`` | ``"mean"``
        How to aggregate token-level hidden states into a single vector.
    max_length : int
        Maximum number of tokens per input.
    batch_size : int
        Batch size for inference.
    device : str
        ``"cpu"`` or ``"cuda"``.  If ``"cuda"`` is requested but
        unavailable, the embedder silently falls back to CPU.
    """

    def __init__(
        self,
        model_name: str,
        pooling: Literal["cls", "mean"] = "cls",
        max_length: int = 128,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size

        # Resolve device — fall back to CPU when CUDA is not available.
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available — falling back to CPU."
            )
            device = "cpu"
        self.device = torch.device(device)

        # Will be populated on first transform().
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._hidden_dim: int | None = None

        logger.info(
            "TransformerEmbedder created — model=%s, pooling=%s, "
            "max_length=%d, batch_size=%d, device=%s",
            model_name,
            pooling,
            max_length,
            batch_size,
            self.device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Download / cache the tokenizer and model weights."""
        logger.info("Loading tokenizer and model: %s …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        # Determine hidden dimension from model config.
        self._hidden_dim = self._model.config.hidden_size
        logger.info(
            "Model loaded — hidden_dim=%d, device=%s",
            self._hidden_dim,
            self.device,
        )

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token-level representations into a single vector per sample.

        Parameters
        ----------
        last_hidden_state : Tensor
            Shape ``(batch, seq_len, hidden_dim)``.
        attention_mask : Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Shape ``(batch, hidden_dim)``.
        """
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]

        # Mean pooling — mask out padding tokens.
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
        sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode *texts* into dense embeddings.

        Parameters
        ----------
        texts : list[str]
            Input documents.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), hidden_dim)``.
        """
        if self._model is None:
            self._load_model()

        assert self._tokenizer is not None
        assert self._model is not None

        all_embeddings: List[np.ndarray] = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(
            "Encoding %d texts in %d batches (batch_size=%d) …",
            len(texts),
            n_batches,
            self.batch_size,
        )

        for start in tqdm(
            range(0, len(texts), self.batch_size),
            total=n_batches,
            desc=f"Embedding ({self.model_name})",
        ):
            batch_texts = texts[start : start + self.batch_size]

            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)

            pooled = self._pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            all_embeddings.append(pooled.cpu().numpy())

        result = np.concatenate(all_embeddings, axis=0)
        logger.info("Embedding complete — output shape: %s", result.shape)
        return result

    def get_dim(self) -> int:
        """Return the embedding dimensionality.

        If the model has not been loaded yet, it will be loaded now so
        that the hidden size can be determined.

        Returns
        -------
        int
            The transformer's hidden dimension.
        """
        if self._hidden_dim is None:
            self._load_model()
        assert self._hidden_dim is not None
        return self._hidden_dim
