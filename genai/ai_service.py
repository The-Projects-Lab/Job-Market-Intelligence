# genai/ai_service.py
from typing import Optional
from genai.llm_provider import GroqProvider, BaseLLMProvider
from genai.guardrails import PromptGuardService

class AIService:
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or GroqProvider()
        self.guardrail = PromptGuardService()

    def process_request(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        # 1. Guardrail Validation (nomic-embed-text-v1.5)
        guard_check = self.guardrail.validate_prompt(user_prompt)
        
        if not guard_check.is_safe:
            return f"⚠️ **Security Notice**: Request rejected. {guard_check.reason}"

        # 2. LLM Execution (Groq)
        return self.provider.generate(prompt=user_prompt, system_prompt=system_prompt)
