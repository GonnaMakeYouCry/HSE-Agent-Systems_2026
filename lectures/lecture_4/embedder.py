"""
Эмбеддер на основе OpenAI API.
"""
from openai import OpenAI

class Embedder:
    def __init__(self, api_key: str, base_url: str = None, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = 1536

    def embed(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in response.data]