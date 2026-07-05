"""TF-IDF text embedding using scikit-learn's TfidfVectorizer.

Wraps ``TfidfVectorizer`` behind a simple fit / transform API that is
consistent with the other embedders in this package.  All outputs are
**dense** numpy arrays so downstream code never has to worry about sparse
matrices.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class TfidfEmbedder:
    """Bag-of-ngrams TF-IDF embedder.

    Parameters are read from the ``config.embeddings.tfidf`` sub-tree using
    ``getattr`` with safe defaults, so missing keys never crash.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        The underlying scikit-learn vectorizer (available after ``fit``).
    """

    def __init__(self, config) -> None:
        tfidf_cfg = getattr(getattr(config, "embeddings", None), "tfidf", None)

        max_features: int = getattr(tfidf_cfg, "max_features", 5000)

        # ngram_range may arrive as a Config object, a list, or a tuple.
        raw_ngram = getattr(tfidf_cfg, "ngram_range", [1, 2])
        if hasattr(raw_ngram, "__iter__") and not isinstance(raw_ngram, str):
            ngram_range: Tuple[int, int] = tuple(int(v) for v in raw_ngram)  # type: ignore[arg-type]
        else:
            ngram_range = (1, 2)

        sublinear_tf: bool = getattr(tfidf_cfg, "sublinear_tf", True)

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )
        self._is_fitted: bool = False

        logger.info(
            "TfidfEmbedder initialised — max_features=%d, ngram_range=%s, "
            "sublinear_tf=%s",
            max_features,
            ngram_range,
            sublinear_tf,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> "TfidfEmbedder":
        """Fit the vectorizer on *texts*.

        Parameters
        ----------
        texts : list[str]
            Training documents.

        Returns
        -------
        TfidfEmbedder
            ``self``, for method chaining.
        """
        logger.info("Fitting TF-IDF on %d documents …", len(texts))
        self.vectorizer.fit(texts)
        self._is_fitted = True
        logger.info(
            "TF-IDF fitted — vocabulary size: %d", len(self.vectorizer.vocabulary_)
        )
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform *texts* into a dense TF-IDF matrix.

        Parameters
        ----------
        texts : list[str]
            Documents to embed.

        Returns
        -------
        np.ndarray
            Dense array of shape ``(len(texts), n_features)``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TfidfEmbedder.transform() called before fit(). "
                "Call fit() or fit_transform() first."
            )
        sparse_matrix = self.vectorizer.transform(texts)
        dense = sparse_matrix.toarray()
        logger.debug("TF-IDF transform → shape %s", dense.shape)
        return dense

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit on *texts* and immediately return the dense TF-IDF matrix.

        Parameters
        ----------
        texts : list[str]
            Training documents.

        Returns
        -------
        np.ndarray
            Dense array of shape ``(len(texts), n_features)``.
        """
        logger.info("Fit-transforming TF-IDF on %d documents …", len(texts))
        sparse_matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        dense = sparse_matrix.toarray()
        logger.info(
            "TF-IDF fit_transform complete — vocabulary size: %d, output shape: %s",
            len(self.vectorizer.vocabulary_),
            dense.shape,
        )
        return dense

    def get_dim(self) -> int:
        """Return the feature dimensionality.

        Returns
        -------
        int
            Number of features (columns) the vectorizer produces.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TfidfEmbedder.get_dim() called before fit(). "
                "Call fit() or fit_transform() first."
            )
        return len(self.vectorizer.vocabulary_)
