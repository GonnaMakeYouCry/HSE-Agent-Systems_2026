from pydantic import BaseModel
from typing import List, Optional

class Chunk(BaseModel):
    text: str
    chunk_id: int
    year: int
    distance: float

class RAGAgentAnswer(BaseModel):
    dataset_row_id: Optional[str] = None
    answer: str
    retrieved_chunks: Optional[List[Chunk]] = None