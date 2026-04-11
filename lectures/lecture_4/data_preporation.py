"""
Подготовка данных: чтение .txt файлов, чанкинг, эмбеддинг, сохранение в LanceDB.
"""
from pathlib import Path
import lancedb
import pyarrow as pa
from chunking import adaptive_chunking
from embedder import Embedder
from tqdm import tqdm
import sys

# Добавляем корень проекта в sys.path для импорта src.config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings

def read_txt_file(file_path: Path) -> str:
    """Читает текстовый файл и возвращает его содержимое."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def process_txt_files(data_dir: str, db_uri: str, table_name: str, embedder: Embedder):
    """
    Обрабатывает все .txt файлы в директории, создаёт чанки, вычисляет эмбеддинги
    и сохраняет в LanceDB.
    """
    db = lancedb.connect(db_uri)
    # Создаём таблицу (перезаписываем, если есть)
    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("chunk_id", pa.int32()),
        pa.field("year", pa.int32()),
        pa.field("embedding", pa.list_(pa.float32(), list_size=1536)),
        pa.field("metadata", pa.string()),
    ])
    db.create_table(table_name, schema=schema, mode="overwrite")
    table = db.open_table(table_name)

    txt_files = list(Path(data_dir).glob("*.txt"))
    if not txt_files:
        print(f"В папке {data_dir} нет .txt файлов!")
        return

    for txt_path in tqdm(txt_files, desc="Processing TXT files"):
        # Определяем год из имени файла (например, report_2019.txt или 2019.txt)
        try:
            year = int(''.join(filter(str.isdigit, txt_path.stem)))
        except ValueError:
            print(f"Не удалось извлечь год из имени файла {txt_path.name}, пропускаем.")
            continue

        text = read_txt_file(txt_path)

        chunks = adaptive_chunking(
            text,
            min_chunk_size=800,
            max_chunk_size=1500,
            min_chunk_overlap=200,
            max_chunk_overlap=400,
            metadata={"source": txt_path.name, "year": year},
        )

        # Подготовка записей для вставки
        records = []
        for chunk in chunks:
            embedding = embedder.embed(chunk.page_content)
            records.append({
                "text": chunk.page_content,
                "chunk_id": chunk.metadata["chunk_id"],
                "year": year,
                "embedding": embedding,
                "metadata": str(chunk.metadata),
            })

        if records:
            table.add(records)

    print(f"Готово! Таблица '{table_name}' содержит {table.count_rows()} строк.")

if __name__ == "__main__":
    # Используем настройки из settings
    API_KEY = settings.polza_ai_api_key
    BASE_URL = "https://api.polza.ai/api/v1"
    MODEL = "text-embedding-3-small"

    embedder = Embedder(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    script_dir = Path(__file__).parent
    process_txt_files(
        data_dir=  str(script_dir / "dataset"),
        db_uri="lectures/lecture_4/lance_db/vectorstore",
        table_name="chunks",
        embedder=embedder,
    )