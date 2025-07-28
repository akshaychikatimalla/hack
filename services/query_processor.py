import json
from typing import List, Set
from pydantic import HttpUrl

from models import QueryAnswer, MappedClause
from services.document_parser import DocumentParser
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
from services.llm_service import LLMService

class QueryProcessor:
    def __init__(self):
        self.doc_parser = DocumentParser()
        self.embedding_service = EmbeddingService()
        self.vector_db_service = VectorDBService()
        self.llm_service = LLMService()
        self.ingested_docs: Set[HttpUrl] = set()

    # This is the method that is missing from your file.
    async def ingest_documents_if_needed(self, document_urls: List[HttpUrl]):
        """
        Processes and embeds documents, skipping any that have already been ingested.
        """
        for doc_url in document_urls:
            if doc_url in self.ingested_docs:
                print(f"Skipping already ingested document: {doc_url}")
                continue

            processed_chunks = await self.doc_parser.parse_document(str(doc_url))
            if not processed_chunks:
                continue

            texts_to_embed = [chunk["text_content"] for chunk in processed_chunks]
            embeddings = await self.embedding_service.get_embeddings(texts_to_embed)

            vectors_to_upsert = []
            for i, chunk in enumerate(processed_chunks):
                # CRITICAL FIX: Add the text content to the metadata for retrieval.
                metadata = chunk['metadata']
                metadata['original_text'] = chunk['text_content']
                
                vectors_to_upsert.append({
                    "id": chunk['id'],
                    "values": embeddings[i],
                    "metadata": metadata
                })

            await self.vector_db_service.upsert_vectors(vectors_to_upsert)
            self.ingested_docs.add(doc_url)

    async def process_query(self, document_urls: List[HttpUrl], query: str) -> QueryAnswer:
        """
        Processes a single query against documents that are assumed to be ingested.
        """
        query_embedding = (await self.embedding_service.get_embeddings([query]))[0]
        
        retrieved_matches = await self.vector_db_service.query_vectors(query_embedding, top_k=5)
        
        context_clauses_for_llm = []
        retrieved_mapped_clauses = []
        if retrieved_matches:
            for match in retrieved_matches:
                meta = match.metadata
                clause_text = meta.get("original_text", "")
                doc_url_str = meta.get("document_url", "")
                page_num = meta.get("page_number")
                
                if clause_text and doc_url_str:
                    context_clauses_for_llm.append(f"Clause from {meta.get('filename', 'doc')}: {clause_text}")
                    retrieved_mapped_clauses.append(MappedClause(text=clause_text, document_url=HttpUrl(doc_url_str), page_number=page_num))

        if not context_clauses_for_llm:
            return QueryAnswer(query=query, decision="N/A", justification="No relevant clauses could be retrieved from the documents for this query.", mapped_clauses=[])

        system_prompt = "You are an intelligent policy analysis system. Analyze the user's query and the provided clauses. Your output MUST be a JSON object with keys: 'decision', 'status', 'payout_amount', 'justification', 'mapped_clauses_texts'."
        clauses_str = '\n---\n'.join(context_clauses_for_llm)
        user_content = f"User Query: {query}\n\nRelevant Document Clauses:\n{clauses_str}"
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        
        llm_response_str = await self.llm_service.get_chat_completion(messages, json_mode=True)
        llm_output = json.loads(llm_response_str)

        cited_texts = set(llm_output.get("mapped_clauses_texts", []))
        final_mapped_clauses = [clause for clause in retrieved_mapped_clauses if clause.text in cited_texts]
        
        return QueryAnswer(
            query=query,
            decision=llm_output.get("decision", "Error"),
            status=llm_output.get("status"),
            payout_amount=llm_output.get("payout_amount"),
            justification=llm_output.get("justification", "No justification provided."),
            mapped_clauses=final_mapped_clauses or retrieved_mapped_clauses
        )
