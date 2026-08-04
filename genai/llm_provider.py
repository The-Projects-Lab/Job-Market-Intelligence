# llm_provider.py
import os
from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass

class GroqProvider(BaseLLMProvider):
    def __init__(self, model_name: Optional[str] = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content
