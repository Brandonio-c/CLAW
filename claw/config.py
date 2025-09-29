"""Configuration management using pydantic-settings."""

from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field


class LLMConfig(BaseSettings):
    """Configuration for LLM components."""
    model_path: str = Field(..., description="Path to the local model")
    model_type: str = Field("huggingface", description="Type of model (huggingface, llama_cpp)")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, description="Repetition penalty")
    device: str = Field("auto", description="Device to run on (auto, cpu, cuda)")
    use_cache: bool = Field(True, description="Use model caching")
    
    class Config:
        env_prefix = "CLAW_LLM_"


class MetaConfig(BaseSettings):
    """Configuration for meta-cognitive controller."""
    max_iters_per_turn: int = Field(5, description="Maximum iterations per turn")
    max_seconds_per_turn: float = Field(30.0, description="Maximum seconds per turn")
    max_samples: int = Field(3, description="Maximum samples per iteration")
    unpredictability: float = Field(0.3, description="Unpredictability dial (0.0-1.0)")
    confidence_threshold: float = Field(0.6, description="Minimum confidence to accept plan")
    improvement_threshold: float = Field(0.05, description="Minimum improvement to continue")
    time_budget_warning: float = Field(0.8, description="Warning threshold for time budget")
    
    class Config:
        env_prefix = "CLAW_META_"


class SymbolicConfig(BaseSettings):
    """Configuration for symbolic planner."""
    opponent_model: str = Field("random", description="Opponent model type")
    simulation_samples: int = Field(10, description="Number of simulation samples")
    repair_attempts: int = Field(3, description="Maximum repair attempts")
    search_depth: int = Field(2, description="Search depth for symbolic planning")
    
    class Config:
        env_prefix = "CLAW_SYMBOLIC_"


class LoggingConfig(BaseSettings):
    """Configuration for logging."""
    level: str = Field("INFO", description="Logging level")
    format: str = Field("json", description="Log format (json, text)")
    trace_file: Optional[str] = Field(None, description="Trace file path")
    enable_rich: bool = Field(True, description="Enable rich console output")
    
    class Config:
        env_prefix = "CLAW_LOG_"


class Config(BaseSettings):
    """Main configuration class."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    meta: MetaConfig = Field(default_factory=MetaConfig)
    symbolic: SymbolicConfig = Field(default_factory=SymbolicConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    seed: Optional[int] = Field(None, description="Random seed")
    debug: bool = Field(False, description="Debug mode")
    output_dir: str = Field("runs", description="Output directory for runs")
    
    class Config:
        env_prefix = "CLAW_"
        case_sensitive = False


# Global config instance
config = Config()
