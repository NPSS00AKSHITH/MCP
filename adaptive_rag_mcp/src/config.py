"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # Authentication
    adaptive_rag_api_key: str = "dev-secret-key-change-in-production"
    
    # Future phases (placeholders)
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "./data/chroma"
    llm_provider: str = "gemini"
    google_api_key: str = "" # Set in .env


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
