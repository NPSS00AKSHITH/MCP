"""Pytest configuration and shared fixtures for Adaptive RAG MCP tests.

This file is automatically loaded by pytest and provides common fixtures
that can be used across all test files.
"""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Get the data directory."""
    return project_root / "data"


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_api_key() -> str:
    """Get the test API key."""
    return os.environ.get("ADAPTIVE_RAG_API_KEY", "test-api-key")


@pytest.fixture
def test_config() -> dict:
    """Get test configuration."""
    return {
        "host": "127.0.0.1",
        "port": 8000,
        "environment": "test",
    }


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_documents() -> list[dict]:
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "Retrieval Augmented Generation (RAG) combines retrieval with generation.",
            "metadata": {"source": "test", "type": "definition"},
        },
        {
            "id": "doc2", 
            "content": "Vector embeddings represent text as dense numerical vectors.",
            "metadata": {"source": "test", "type": "technical"},
        },
        {
            "id": "doc3",
            "content": "BM25 is a sparse retrieval algorithm based on term frequency.",
            "metadata": {"source": "test", "type": "technical"},
        },
    ]


@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "What is RAG and how does it work?"


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Sample chunks for reranking tests."""
    return [
        {"id": "chunk1", "content": "RAG improves accuracy by grounding responses in retrieved documents."},
        {"id": "chunk2", "content": "The weather today is sunny with a high of 75 degrees."},
        {"id": "chunk3", "content": "Retrieval augmented generation reduces hallucinations in LLMs."},
    ]


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture
def temp_test_db(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary database path for tests."""
    db_path = tmp_path / "test_chunks.db"
    yield db_path
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def temp_index_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for vector indexes."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    yield index_dir
    # Cleanup happens automatically with tmp_path
