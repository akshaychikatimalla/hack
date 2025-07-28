import httpx  # Use httpx for async requests
from typing import List, Dict, Any
from io import BytesIO
from urllib.parse import urlparse
import uuid

# You'll need libraries for specific file types:
import PyPDF2
from docx import Document as DocxDocument

# Correctly import the token-based chunker
from utils.chunking import get_tokenizer, chunk_text_by_tokens

class DocumentParser:
    def __init__(self):
        # Initialize the tokenizer once
        self.tokenizer = get_tokenizer()

    async def parse_document(self, url: str) -> List[Dict[str, Any]]:
        """
        Asynchronously fetches a document from a URL, parses its content based on
        file type, and splits it into token-based chunks.
        """
        parsed_url = urlparse(url)
        filename = parsed_url.path.split('/')[-1]
        file_extension = filename.split('.')[-1].lower() if '.' in filename else ''

        try:
            # Use an async client to fetch the document without blocking
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()  # Raise an exception for HTTP errors
                content = response.content

            text_content = ""
            if file_extension == "pdf":
                text_content = self._parse_pdf(content)
            elif file_extension in ["doc", "docx"]:
                text_content = self._parse_docx(content)
            elif file_extension in ["txt", "md", "html", "json"]:  # Basic text handling
                text_content = content.decode('utf-8', errors='ignore')
            else:
                print(f"Warning: Unsupported file type for {url}: {file_extension}")
                return []
            
            if not text_content:
                print(f"Warning: No text content extracted from {url}")
                return []

            # Correctly call the token-based chunker
            chunks = chunk_text_by_tokens(text_content, self.tokenizer)

            processed_chunks = []
            for i, chunk_str in enumerate(chunks):
                processed_chunks.append({
                    "id": str(uuid.uuid4()),  # Generate a unique ID for each chunk
                    "text_content": chunk_str,
                    "metadata": {
                        "document_url": url,
                        "filename": filename,
                        "file_type": file_extension,
                        "chunk_index": i
                    }
                })
            return processed_chunks

        except httpx.RequestError as e:
            print(f"Error fetching document {url}: {e}")
            return []
        except Exception as e:
            print(f"Error parsing document {url}: {e}")
            return []

    def _parse_pdf(self, content: bytes) -> str:
        text = ""
        try:
            reader = PyPDF2.PdfReader(BytesIO(content))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error parsing PDF content: {e}")
        return text

    def _parse_docx(self, content: bytes) -> str:
        text = ""
        try:
            doc = DocxDocument(BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"Error parsing DOCX content: {e}")
        return text
