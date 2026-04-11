"""
Ретривер: поиск и постобработка чанков.
"""
from typing import List, Optional
from models import Chunk
from embedder import Embedder

def retrieve(
    query: str,
    table,
    embedder: Embedder,
    top_k: int = 3,
    year_filter: Optional[int] = None,
    max_total_chars: int = 6000,
    distance_threshold: float = 0.7,
) -> List[Chunk]:
    query_emb = embedder.embed(query)
    results = table.search(query_emb).metric("cosine").limit(top_k * 3).to_list()

    filtered = []
    for r in results:
        if r["_distance"] > distance_threshold:
            continue
        if year_filter is not None and r["year"] != year_filter:
            continue
        filtered.append(r)

    filtered.sort(key=lambda x: x["_distance"])

    selected = []
    total_len = 0
    for r in filtered[:top_k]:
        chunk_len = len(r["text"])
        if total_len + chunk_len > max_total_chars and selected:
            break
        selected.append(r)
        total_len += chunk_len

    return [Chunk(text=r["text"], chunk_id=r["chunk_id"], year=r["year"], distance=r["_distance"]) for r in selected]