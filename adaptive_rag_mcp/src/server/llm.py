"""LLM Client for Google Generative AI."""

import google.generativeai as genai
from typing import Optional

from src.config import get_settings
from src.server.logging import get_logger

logger = get_logger(__name__)

class LLMClient:
    """Wrapper for Google Generative AI client."""

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.google_api_key
        
        if not self.api_key:
            logger.warning("no_google_api_key_found")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("llm_client_initialized", provider="gemini")
            except Exception as e:
                logger.error("llm_client_init_failed", error=str(e))
                self.model = None

    def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt."""
        if not self.model:
            return "Error: LLM not configured (missing API Key)."
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            return f"Error generating text: {str(e)}"

# Global instance
_CLIENT = None

def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
