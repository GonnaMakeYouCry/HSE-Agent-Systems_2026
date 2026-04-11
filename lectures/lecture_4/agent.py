"""
RAG-агент: простой retrieval-augmented генератор.
"""
import re
from typing import Optional
from openai import OpenAI
from models import Chunk, RAGAgentAnswer
from retriever import retrieve
from embedder import Embedder


class RAGAgent:
    def __init__(self, client: OpenAI, model: str, table, embedder: Embedder):
        self.client = client
        self.model = model
        self.table = table
        self.embedder = embedder

    def run(self, question: str, dataset_row_id: Optional[str] = None, verbose: bool = False) -> RAGAgentAnswer:
        year_match = re.search(r'\b(2019|2025)\b', question)
        year = int(year_match.group(1)) if year_match else None

        chunks = retrieve(
            query=question,
            table=self.table,
            embedder=self.embedder,
            year_filter=year,
            max_total_chars=6000,
            top_k=2,
            distance_threshold=0.6
        )

        if not chunks:
            answer = "Информация не найдена в доступных документах."
        else:
            context_parts = []
            for i, c in enumerate(chunks, 1):
                context_parts.append(f"[{i}] (год {c.year}) {c.text}")
            context = "\n\n".join(context_parts)

            system_prompt = (
                "Ты — эксперт по проверке фактов. Отвечай на вопрос пользователя строго на основе предоставленного контекста.\n"
                "Выполни следующие шаги:\n"
                "1. Найди в контексте все фрагменты, которые имеют отношение к вопросу.\n"
                "2. Выпиши из них точные цифры, даты, названия или утверждения.\n"
                "3. Сформулируй ответ, используя только эти выписанные факты.\n"
                "4. Каждое фактическое утверждение в ответе должно заканчиваться ссылкой на номер источника в квадратных скобках, например [1].\n"
                "5. Если ни один фрагмент не содержит ответа, напиши ровно 'Информация не найдена'.\n"
                "Не добавляй никакой информации от себя, даже если она кажется очевидной."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            answer = response.choices[0].message.content

        return RAGAgentAnswer(
            dataset_row_id=dataset_row_id,
            answer=answer,
            retrieved_chunks=chunks,
        )
