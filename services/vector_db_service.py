from pinecone import Pinecone, Index, PodSpec
from typing import List, Dict, Any
from config import settings
import asyncio

class VectorDBService:
    def __init__(self):
        self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT)
        self.index_name = settings.PINECONE_INDEX_NAME
        self._index: Index = None

    async def _get_or_create_index(self) -> Index:
        if self._index:
            return self._index
        
        # Pinecone list_indexes is synchronous
        existing_indexes = self.pinecone.list_indexes()
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=1536,  # Dimension for OpenAI's text-embedding-ada-002
                metric='cosine',
                spec=PodSpec(environment=settings.PINECONE_ENVIRONMENT) # Use PodSpec for serverless or specific environment
            )
            # Give it a moment to become active
            await asyncio.sleep(5)
        
        self._index = self.pinecone.Index(self.index_name)
        return self._index

    async def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """
        Upserts vectors into the Pinecone index.
        Each vector dict should have 'id', 'values', and 'metadata'.
        """
        index = await self._get_or_create_index()
        try:
            # Pinecone upsert is synchronous
            asyncio.get_event_loop().run_in_executor(
                None, lambda: index.upsert(vectors=vectors, batch_size=100)
            )
            print(f"Upserted {len(vectors)} vectors to Pinecone.")
        except Exception as e:
            print(f"Error upserting vectors to Pinecone: {e}")
            raise

    async def query_vectors(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the Pinecone index with a given embedding.
        """
        index = await self._get_or_create_index()
        try:
            # Pinecone query is synchronous
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            )
            return response.matches
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            raise