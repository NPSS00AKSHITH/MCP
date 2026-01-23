"""Configuration management using Pydantic Settings.

Enhanced with security settings per MCP best practices.
"""

from functools import lru_cache
from pydantic import Field
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
    host: str = Field(
        default="127.0.0.1",
        description="Server bind address. Use 127.0.0.1 for local, 0.0.0.0 for Docker/remote",
    )
    port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Environment mode
    environment: str = Field(
        default="development",
        description="Environment: 'development', 'staging', or 'production'",
    )
    
    # Authentication
    adaptive_rag_api_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="API key for authentication. MUST be changed in production!",
    )
    
    # Security settings (MCP Best Practices)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins. Use '*' only in development.",
    )
    cors_allow_all: bool = Field(
        default=False,
        description="Allow all CORS origins (development only!)",
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Maximum requests per rate limit window",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate limit window in seconds",
    )
    
    # RAG settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    vector_db_path: str = Field(
        default="./data/chroma",
        description="Path to vector database storage",
    )
    llm_provider: str = Field(default="gemini", description="LLM provider")
    google_api_key: str = Field(default="", description="Google API key for Gemini")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"
    
    @property
    def effective_cors_origins(self) -> list[str]:
        """Get effective CORS origins based on environment."""
        if self.cors_allow_all and not self.is_production:
            return ["*"]
        return self.cors_origins


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

