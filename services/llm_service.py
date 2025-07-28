from openai import OpenAI
from typing import List, Dict, Any
from config import settings

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL

    async def get_chat_completion(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """
        Gets a chat completion from the LLM.
        """
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=0.0 # For consistent, factual responses
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting LLM completion: {e}")
            raise