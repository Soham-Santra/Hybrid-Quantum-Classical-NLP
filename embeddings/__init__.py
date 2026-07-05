"""Text representation and embedding modules.

Public API
----------
- :class:`TfidfEmbedder`        — TF-IDF bag-of-ngrams embeddings.
- :class:`TransformerEmbedder`  — BERT / mBERT / IndicBERT embeddings.
- :func:`get_embedder`          — Factory that returns the right embedder.
- :func:`embed_dataset`         — Embed all splits of an EmotionDataset.
"""

from embeddings.tfidf import TfidfEmbedder
from embeddings.bert_emb import TransformerEmbedder
from embeddings.factory import get_embedder, embed_dataset

__all__ = [
    "TfidfEmbedder",
    "TransformerEmbedder",
    "get_embedder",
    "embed_dataset",
]
