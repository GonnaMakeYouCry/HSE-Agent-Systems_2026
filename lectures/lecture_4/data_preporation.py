"""
Подготовка данных: чтение PDF, очистка, чанкинг (1500/400), эмбеддинг, сохранение в LanceDB.
"""
from pathlib import Path
import re
import sys

import lancedb
import pyarrow as pa
from pypdf import PdfReader
from chunking import semantic_chunking
from embedder import Embedder
from tqdm import tqdm

# Добавляем корень проекта для импорта src.config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


def extract_year_from_filename(filename: str) -> int:
    match = re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", filename)
    if not match:
        raise ValueError(f"Не удалось извлечь год из имени файла: {filename}")
    return int(match.group(0))


def clean_text(text: str) -> str:
    """Базовая очистка текста PDF."""
    text = text.replace("\xa0", " ").replace("\u200b", " ").replace("\r", "\n")
    text = re.sub(r"-\n(?=\w)", "", text)  # убираем переносы слов
    # Заменяем два и более переносов строки на плейсхолдер, одиночные – на пробел
    placeholder = "<<<PARA>>>"
    text = re.sub(r"\n{2,}", placeholder, text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace(placeholder, "\n\n")
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())
    return clean_text("\n\n".join(pages))


def process_pdfs(pdf_dir: str, db_uri: str, table_name: str, embedder: Embedder):
    db = lancedb.connect(db_uri)
    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("chunk_id", pa.int32()),
        pa.field("year", pa.int32()),
        pa.field("embedding", pa.list_(pa.float32(), list_size=1536)),
        pa.field("metadata", pa.string()),
    ])
    db.create_table(table_name, schema=schema, mode="overwrite")
    table = db.open_table(table_name)

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"В папке {pdf_dir} нет PDF-файлов!")
        return

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        year = extract_year_from_filename(pdf_path.name)
        text = extract_text_from_pdf(pdf_path)

        chunks = semantic_chunking(
            text,
            chunk_size=1500,
            chunk_overlap=400,
            metadata={"source": pdf_path.name, "year": year},
        )

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
    # Пути относительно расположения скрипта
    script_dir = Path(__file__).parent
    pdf_dir = script_dir / "dataset"
    db_uri = script_dir / "lance_db" / "vectorstore"

    embedder = Embedder(
        api_key=settings.polza_ai_api_key,
        base_url="https://api.polza.ai/api/v1",
        model="text-embedding-3-small"
    )

    process_pdfs(
        pdf_dir=str(pdf_dir),
        db_uri=str(db_uri),
        table_name="chunks",
        embedder=embedder,
    )