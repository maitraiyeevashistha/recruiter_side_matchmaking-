from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

        self.use_e5_prefix = "e5" in model_name.lower()

    def _prepare_documents(self, texts):
        if not self.use_e5_prefix:
            return texts

        return [f"passage: {text}" for text in texts]

    def _prepare_query(self, text):
        if not self.use_e5_prefix:
            return text

        return f"query: {text}"

    def embed_documents(self, texts):
        vectors = self.model.encode(
            self._prepare_documents(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(vectors).tolist()

    def embed_query(self, text):
        vector = self.model.encode(
            [self._prepare_query(text)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return np.asarray(vector).tolist()


@lru_cache(maxsize=1)
def get_embedding_service(model_name):
    return EmbeddingService(model_name)