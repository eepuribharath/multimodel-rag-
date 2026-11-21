"""Qdrant hybrid retrieval: dense + BM25 sparse, fused with RRF.

Runs against embedded local mode by default (no server, no Docker): points
persist under QDRANT_URL=local (a folder) or live in-memory for tests.
An http(s):// QDRANT_URL switches to a real server/cloud with no code change.

Fusion is computed client-side rather than with Qdrant's Fusion.RRF because
(a) the spec fixes k=60 and Qdrant's built-in fusion does not expose k, and
(b) RetrievedItem carries per-leg dense/sparse scores, which a fused
server-side result no longer has.
"""

import uuid

from qdrant_client import QdrantClient, models

from app.config import PROJECT_ROOT, Settings, get_settings
from app.retrieval.sparse import Bm25SparseEncoder
from app.schemas import Chunk, RetrievedItem, Vector, chunk_match_text

DENSE = "dense"
SPARSE = "sparse"


def _point_id(chunk_id: str) -> str:
    # Qdrant point IDs must be uint or UUID; chunk IDs are arbitrary strings.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dim: int,
        encoder: Bm25SparseEncoder | None = None,
        rrf_k: int = 60,
    ):
        self.client = client
        self.collection = collection
        self.dim = dim
        self.encoder = encoder or Bm25SparseEncoder()
        self.rrf_k = rrf_k
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors[DENSE].size
            if existing != self.dim:
                raise RuntimeError(
                    f"collection '{self.collection}' was created with dim "
                    f"{existing}, config now says {self.dim} — changing "
                    "EMBED_DIM requires a full re-index into a new collection"
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        points = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dim:
                raise ValueError(
                    f"vector for chunk {chunk.id} has dim {len(vector)}, "
                    f"collection requires {self.dim}"
                )
            points.append(
                models.PointStruct(
                    id=_point_id(chunk.id),
                    vector={
                        DENSE: vector,
                        SPARSE: self.encoder.encode_document(chunk_match_text(chunk)),
                    },
                    payload=chunk.model_dump(),
                )
            )
        self.client.upsert(self.collection, points=points)

    def hybrid_search(
        self,
        query_vector: Vector,
        query_text: str,
        top_k: int = 100,
        filters: dict | None = None,
    ) -> list[RetrievedItem]:
        query_filter = self._filter(filters)
        dense_hits = self.client.query_points(
            self.collection,
            query=query_vector,
            using=DENSE,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        sparse_query = self.encoder.encode_query(query_text)
        sparse_hits = (
            self.client.query_points(
                self.collection,
                query=sparse_query,
                using=SPARSE,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            ).points
            if sparse_query.indices
            else []
        )
        return self._fuse(dense_hits, sparse_hits, top_k)

    def _fuse(self, dense_hits, sparse_hits, top_k: int) -> list[RetrievedItem]:
        """rrf_score(d) = Σ_legs 1 / (rrf_k + rank_leg(d)), ranks 1-based."""
        items: dict[str, dict] = {}
        for leg, hits in ((DENSE, dense_hits), (SPARSE, sparse_hits)):
            for rank, hit in enumerate(hits, start=1):
                entry = items.setdefault(
                    str(hit.id),
                    {"payload": hit.payload, DENSE: None, SPARSE: None, "rrf": 0.0},
                )
                entry[leg] = hit.score
                entry["rrf"] += 1.0 / (self.rrf_k + rank)
        ranked = sorted(items.values(), key=lambda e: e["rrf"], reverse=True)
        return [
            RetrievedItem(
                chunk=Chunk(**entry["payload"]),
                dense_score=entry[DENSE],
                sparse_score=entry[SPARSE],
                rrf_score=entry["rrf"],
            )
            for entry in ranked[:top_k]
        ]

    @staticmethod
    def _filter(filters: dict | None) -> models.Filter | None:
        if not filters:
            return None
        conditions = [
            models.FieldCondition(
                key=key,
                match=models.MatchAny(any=value)
                if isinstance(value, list)
                else models.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
        return models.Filter(must=conditions)


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    url = settings.qdrant_url
    if url.startswith("http"):
        return QdrantClient(url=url)
    if url == ":memory:":
        return QdrantClient(":memory:")
    # anything else is embedded local mode persisting to a folder
    path = PROJECT_ROOT / "qdrant_data" if url == "local" else url
    return QdrantClient(path=str(path))


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    return VectorStore(
        client=get_qdrant_client(settings),
        collection=settings.qdrant_collection,
        dim=settings.embed_dim,
        rrf_k=settings.rrf_k,
    )
