"""Core datatypes for the CLAW framework."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class PlanStatus(Enum):
    """Status of a plan in the planning process."""
    PENDING = "pending"
    LEGAL = "legal"
    ILLEGAL = "illegal"
    REPAIRED = "repaired"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class Plan:
    """Represents a plan with orders, rationale, and metadata."""
    orders: Dict[str, Any]  # Domain-specific orders (normalized)
    rationale: str  # Human-readable explanation
    confidence: float  # From LLM or heuristic (0.0-1.0)
    status: PlanStatus = PlanStatus.PENDING
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate plan data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class Order:
    """Represents a single order in a plan."""
    unit: str
    action: str
    target: Optional[str] = None
    support: Optional[str] = None
    convoy: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Score:
    """Represents a score with breakdown and metadata."""
    total: float
    breakdown: Dict[str, float]  # Component scores
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate score data after initialization."""
        if not isinstance(self.total, (int, float)):
            raise ValueError(f"Total score must be numeric, got {type(self.total)}")


@dataclass
class Feedback:
    """Feedback from simulation or evaluation."""
    is_legal: bool
    score: Score
    diagnostics: Dict[str, Any]  # Illegal orders, bounced moves, etc.
    suggestions: Optional[List[str]] = None  # Repair suggestions
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TraceEvent:
    """A single event in the planning trace."""
    timestamp: float
    event_type: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
