"""
Runtime Configuration Management for Corrective RAG with Personalized PageRank.
Provides dynamic, thread-safe configuration updates at runtime via API and Streamlit UI.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
import threading
import os


class RuntimeConfig(BaseModel):
    """Configurable system parameters editable at runtime."""
    # Retrieval configuration
    retrieval_algorithm: Literal["ppr", "vector_hop", "hybrid"] = Field(
        default="ppr",
        description="Retrieval strategy: 'ppr' (Personalized PageRank), 'vector_hop' (sequential dense search), or 'hybrid' (RRF fusion)"
    )
    top_k_passages: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top context passages to retrieve for answer generation"
    )
    ppr_alpha: float = Field(
        default=0.85,
        ge=0.01,
        le=0.99,
        description="Damping factor for Personalized PageRank (probability of following graph edges vs teleports)"
    )

    # CRAG Grading thresholds
    grader_threshold_high: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Upper threshold: context above this score is graded CORRECT and used directly"
    )
    grader_threshold_low: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Lower threshold: context between low and high is AMBIGUOUS (triggers rewrite); below is INCORRECT (triggers fallback)"
    )
    sufficiency_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of relevant retrieved chunks required to proceed to generation without query rewriting"
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum query reformulation loops before forcing best-effort generation"
    )

    # Models and LLM options
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-Transformers model name for dense retrieval and grading"
    )
    llm_provider: Literal["local_extractive", "openai", "gemini"] = Field(
        default="local_extractive",
        description="Generator model provider: 'local_extractive' (offline, fast), 'openai', or 'gemini'"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier when provider is openai"
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Google Gemini model identifier when provider is gemini"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for external LLM provider"
    )

    # Dataset paths
    dataset_path: str = Field(
        default="data/2wikimultihopqa_dev.parquet",
        description="Path to the primary 2WikiMultiHopQA dataset (Parquet format)"
    )
    sample_dataset_path: str = Field(
        default="data/2wikimultihopqa_sample.json",
        description="Path to the lightweight cached sample dataset (JSON format)"
    )


class ConfigManager:
    """Thread-safe singleton holding the active runtime configuration."""
    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._config = RuntimeConfig()
                    # Check environment variables for initial keys
                    openai_key = os.environ.get("OPENAI_API_KEY")
                    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                    if openai_key:
                        cls._instance._config.api_key = openai_key
                        cls._instance._config.llm_provider = "openai"
                    elif gemini_key:
                        cls._instance._config.api_key = gemini_key
                        cls._instance._config.llm_provider = "gemini"
        return cls._instance

    def get_config(self) -> RuntimeConfig:
        """Get the current configuration snapshot."""
        with self._lock:
            return self._config.model_copy()

    def update_config(self, updates: dict) -> RuntimeConfig:
        """Safely update configuration fields at runtime."""
        with self._lock:
            current_dict = self._config.model_dump()
            current_dict.update(updates)
            self._config = RuntimeConfig(**current_dict)
            return self._config.model_copy()


# Global singleton access
config_manager = ConfigManager()


def get_current_config() -> RuntimeConfig:
    return config_manager.get_config()


def update_runtime_config(updates: dict) -> RuntimeConfig:
    return config_manager.update_config(updates)
