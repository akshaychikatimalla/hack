import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenAI API Key
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key_here")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "your_pinecone_environment_here")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "policy-retrieval-index")
    EMBEDDING_MODEL: str = "text-embedding-ada-002"

    # API Authentication Token (for /hackrx/run)
    API_AUTH_TOKEN: str = os.getenv("API_AUTH_TOKEN", "73332fdc9c30b48a918eadc5e9a8c379e902dd1126f2bfb9024c15c6daeaff29") # From image

    # LLM Model
    LLM_MODEL: str = "gpt-4-turbo" # Or "gpt-4", "gpt-4o"

settings = Settings()