"""Embedder factory and dataset embedding utilities.

Provides :func:`get_embedder` which inspects ``config.embeddings.method``
and returns the matching embedder instance, and :func:`embed_dataset` which
runs the embedder over every split of an ``EmotionDataset``.
"""

import logging
from typing import Any, Dict, Union

import numpy as np

from embeddings.tfidf import TfidfEmbedder
from embeddings.bert_emb import TransformerEmbedder

logger = logging.getLogger(__name__)

# Type alias for any embedder object this factory can return.
Embedder = Union[TfidfEmbedder, TransformerEmbedder]


def get_embedder(config: Any, language: str = "english") -> Embedder:
    """Return an embedder instance based on the framework configuration.

    Parameters
    ----------
    config : Config
        Framework configuration with a ``config.embeddings`` sub-tree.
    language : str
        Target language — currently informational; individual transformer
        model names are selected via the config.

    Returns
    -------
    TfidfEmbedder | TransformerEmbedder
        Ready-to-use embedder.

    Raises
    ------
    NotImplementedError
        For ``fasttext`` or ``word2vec`` methods (not yet implemented).
    ValueError
        For unknown embedding methods.
    """
    emb_cfg = getattr(config, "embeddings", None)
    method: str = getattr(emb_cfg, "method", "tfidf")

    logger.info("Creating embedder — method=%s, language=%s", method, language)

    if method == "tfidf":
        return TfidfEmbedder(config)

    if method in ("bert", "mbert", "indicbert"):
        # Pull the method-specific sub-config (e.g. config.embeddings.bert).
        method_cfg = getattr(emb_cfg, method, None)

        model_name: str = getattr(method_cfg, "model_name", "bert-base-uncased")
        pooling: str = getattr(method_cfg, "pooling", "cls")
        max_length: int = getattr(method_cfg, "max_length", 128)
        batch_size: int = getattr(method_cfg, "batch_size", 32)
        device: str = getattr(config, "device", "cpu")

        return TransformerEmbedder(
            model_name=model_name,
            pooling=pooling,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )

    if method == "fasttext":
        raise NotImplementedError(
            "FastText embeddings are not yet implemented. "
            "Please use 'tfidf', 'bert', 'mbert', or 'indicbert' instead, "
            "or contribute a FastTextEmbedder to embeddings/fasttext.py."
        )

    if method == "word2vec":
        raise NotImplementedError(
            "Word2Vec embeddings are not yet implemented. "
            "Please use 'tfidf', 'bert', 'mbert', or 'indicbert' instead, "
            "or contribute a Word2VecEmbedder to embeddings/word2vec.py."
        )

    raise ValueError(
        f"Unknown embedding method: '{method}'. "
        f"Supported methods: tfidf, bert, mbert, indicbert, "
        f"fasttext (planned), word2vec (planned)."
    )


def embed_dataset(
    embedder: Embedder,
    dataset: Any,
) -> Dict[str, np.ndarray]:
    """Embed every split of an ``EmotionDataset``.

    For :class:`TfidfEmbedder` the vectorizer is **fit** on the training
    split and then applied (transform only) to validation and test.  For
    :class:`TransformerEmbedder` every split is simply transformed (no
    fitting is needed).

    Parameters
    ----------
    embedder : TfidfEmbedder | TransformerEmbedder
        The embedder to use.
    dataset : EmotionDataset
        Dataset with ``.train``, ``.val``, and ``.test`` dicts, each
        containing a ``"texts"`` key.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"train": np.ndarray, "val": np.ndarray, "test": np.ndarray}``
        where each array has shape ``(n_samples, embedding_dim)``.
    """
    result: Dict[str, np.ndarray] = {}

    train_texts = dataset.train.get("texts", [])
    val_texts = dataset.val.get("texts", [])
    test_texts = dataset.test.get("texts", [])

    if isinstance(embedder, TfidfEmbedder):
        logger.info("Embedding dataset with TF-IDF (fit on train) …")
        result["train"] = embedder.fit_transform(train_texts)
        result["val"] = embedder.transform(val_texts) if val_texts else np.empty((0, embedder.get_dim()))
        result["test"] = embedder.transform(test_texts) if test_texts else np.empty((0, embedder.get_dim()))
    else:
        # Transformer — no fitting required.
        logger.info("Embedding dataset with transformer …")
        for split_name, texts in [
            ("train", train_texts),
            ("val", val_texts),
            ("test", test_texts),
        ]:
            if texts:
                logger.info("  Embedding %s split (%d texts) …", split_name, len(texts))
                result[split_name] = embedder.transform(texts)
            else:
                logger.warning("  %s split is empty — returning zero-row array.", split_name)
                result[split_name] = np.empty((0, embedder.get_dim()))

    for split_name, arr in result.items():
        logger.info("  %s embeddings shape: %s", split_name, arr.shape)

    return result
