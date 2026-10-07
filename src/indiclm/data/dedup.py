"""Deduplication: exact (hash) and near-duplicate (MinHash/LSH).

Semantic (embedding-based) deduplication is designed for but not
implemented in this milestone — see `SemanticDeduplicator` docstring for
why (no embedding model is bundled here; wiring one in is a drop-in
extension of the same `Deduplicator` interface).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from datasketch import MinHash, MinHashLSH

from indiclm.data.schema import Document
from indiclm.data.text_utils import split_words


def _normalized_hash(text: str) -> str:
    normalized = " ".join(split_words(text.lower()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 3) -> set[str]:
    words = split_words(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class Deduplicator(Protocol):
    def process(self, docs: list[Document]) -> list[Document]: ...


@dataclass
class ExactDeduplicator:
    """Marks exact duplicates via a hash of whitespace/case-normalized text."""

    def process(self, docs: list[Document]) -> list[Document]:
        seen: dict[str, int] = {}
        cluster_id = 0
        for doc in docs:
            h = _normalized_hash(doc.text)
            if h in seen:
                doc.is_duplicate = True
                doc.dedup_cluster = seen[h]
            else:
                seen[h] = cluster_id
                doc.dedup_cluster = cluster_id
                cluster_id += 1
        return docs


@dataclass
class MinHashNearDeduplicator:
    """LSH-based near-duplicate detection over word-shingle MinHash sketches."""

    num_perm: int = 64
    threshold: float = 0.8
    shingle_size: int = 3

    def process(self, docs: list[Document]) -> list[Document]:
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        sketches: dict[str, MinHash] = {}

        for doc in docs:
            if doc.is_duplicate:
                continue  # already an exact duplicate; skip near-dup pass
            shingles = _shingles(doc.text, self.shingle_size)
            m = MinHash(num_perm=self.num_perm)
            for sh in shingles:
                m.update(sh.encode("utf-8"))
            matches: list[str] = lsh.query(m)
            if matches:
                doc.is_near_duplicate = True
                doc.dedup_cluster = matches[0]
            else:
                lsh.insert(doc.document_id, m)
                sketches[doc.document_id] = m
        return docs


@dataclass
class SemanticDeduplicator:
    """TF-IDF cosine-similarity near-duplicate detection.

    Computes a TF-IDF matrix over all non-duplicate documents, then marks
    any document whose cosine similarity to a previously-seen document
    exceeds `threshold` as a near-duplicate (first-seen wins).

    Uses scikit-learn's TfidfVectorizer — no external embedding model needed.
    The interface mirrors `MinHashNearDeduplicator` so it can be swapped in
    without changing pipeline wiring.
    """

    threshold: float = 0.85
    max_features: int = 50_000

    def process(self, docs: list[Document]) -> list[Document]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = [doc for doc in docs if not doc.is_duplicate and not doc.is_near_duplicate]
        if len(candidates) < 2:
            return docs

        texts = [doc.text for doc in candidates]
        vectorizer = TfidfVectorizer(max_features=self.max_features, sublinear_tf=True)
        tfidf = vectorizer.fit_transform(texts)

        seen_indices: list[int] = []
        for i in range(len(candidates)):
            if candidates[i].is_near_duplicate:
                continue
            if seen_indices:
                sims = cosine_similarity(tfidf[i], tfidf[seen_indices]).flatten()
                if float(sims.max()) >= self.threshold:
                    best = seen_indices[int(sims.argmax())]
                    candidates[i].is_near_duplicate = True
                    candidates[i].dedup_cluster = candidates[best].document_id
                    continue
            seen_indices.append(i)

        return docs


@dataclass
class EmbeddingDeduplicator:
    """GPU-accelerated semantic deduplication using multilingual-e5-small.

    Encodes all candidate documents, then greedily removes any document whose
    cosine similarity to a previously-kept document exceeds `threshold`.
    Runs at ~7,000 texts/s on a V100 (len 64); adjust batch_size for memory.

    Requires: pip install sentence-transformers
    """

    model_name: str = "intfloat/multilingual-e5-small"
    threshold: float = 0.90
    batch_size: int = 512
    device: str = "cuda"

    def process(self, docs: list[Document]) -> list[Document]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers") from e

        import torch

        candidates = [d for d in docs if not d.is_duplicate and not d.is_near_duplicate]
        if len(candidates) < 2:
            return docs

        model = SentenceTransformer(self.model_name, device=self.device)
        # e5 models are trained with "passage: " prefix for asymmetric retrieval
        texts = [f"passage: {d.text[:512]}" for d in candidates]

        embeddings: torch.Tensor = model.encode(  # type: ignore[assignment]
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=self.device,
        )  # [N, D], unit-normed so cosine sim == dot product

        N, D = embeddings.shape
        # Pre-allocate keeper buffer to avoid growing a Python list of tensors
        keeper_buf = torch.empty(N, D, device=embeddings.device, dtype=embeddings.dtype)
        keeper_idx: list[int] = []  # maps keeper_buf row → candidates index
        kept_count = 0

        for i in range(N):
            if candidates[i].is_near_duplicate:
                continue
            if kept_count > 0:
                # [kept_count] dot products — fully on GPU
                sims = embeddings[i] @ keeper_buf[:kept_count].T  # [kept_count]
                max_sim = float(sims.max())
                if max_sim >= self.threshold:
                    best = int(sims.argmax())
                    candidates[i].is_near_duplicate = True
                    candidates[i].dedup_cluster = candidates[keeper_idx[best]].document_id
                    continue
            keeper_buf[kept_count] = embeddings[i]
            keeper_idx.append(i)
            kept_count += 1

        return docs
