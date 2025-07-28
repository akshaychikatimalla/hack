from typing import List
import tiktoken

def get_tokenizer(model_name: str = "gpt-3.5-turbo"):
    """
    Returns a tiktoken tokenizer for a given model.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: Model '{model_name}' not found. Using 'cl100k_base' encoding.")
        return tiktoken.get_encoding("cl100k_base")

def chunk_text_by_tokens(
    text: str,
    encoding: tiktoken.Encoding,
    max_tokens: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Splits text into chunks based on actual token count with an optional token overlap.
    """
    if not text:
        return []

    # Encode the entire text into tokens
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        # Determine the end of the chunk
        end = start + max_tokens
        
        # Slice the tokens for the current chunk and decode back to text
        chunk_text = encoding.decode(tokens[start:end])
        chunks.append(chunk_text)
        
        # Move the start position for the next chunk, accounting for overlap
        start += max_tokens - overlap
        
    return chunks