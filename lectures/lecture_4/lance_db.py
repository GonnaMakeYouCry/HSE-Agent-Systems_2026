# ---------- lance_db.py ----------
"""
Работа с векторной БД LanceDB.
"""
import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional


class LanceDBClient:
    def __init__(self, uri: str):
        self.db = lancedb.connect(uri)

    def create_table(self, table_name: str, mode: str = "create"):
        """Создаёт таблицу с заданной схемой, если её нет."""
        if table_name in self.db.table_names():
            if mode == "overwrite":
                self.db.drop_table(table_name)
            else:
                return
        schema = pa.schema([
            pa.field("text", pa.string()),
            pa.field("chunk_id", pa.int32()),
            pa.field("year", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=1536)),
            pa.field("metadata", pa.string()),  # JSON строка
        ])
        self.db.create_table(table_name, schema=schema)

    def open_table(self, table_name: str):
        return self.db.open_table(table_name)

    def add_records(self, records: List[Dict[str, Any]]):
        """Добавляет записи в таблицу (таблица должна быть открыта)."""
        import json
        for r in records:
            r["metadata"] = json.dumps(r["metadata"])
        self.table.add(records)

    def search(self, query_embedding: List[float], top_k: int = 5, year_filter: Optional[int] = None):
        """Поиск ближайших чанков с опциональной фильтрацией по году."""
        q = self.table.search(query_embedding).metric("cosine").limit(top_k)
        if year_filter is not None:
            q = q.where(f"year = {year_filter}")
        return q.to_list()

    def count(self) -> int:
        return self.table.count_rows()

    def read_all(self, limit: int = 100) -> List[Dict]:
        """Чтение всех записей (для проверки)."""
        return self.table.to_pandas().to_dict("records")