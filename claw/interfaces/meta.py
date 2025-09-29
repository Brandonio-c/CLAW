"""Meta-policy interface."""

from typing import Protocol, Dict, Any
from claw.types import Plan


class MetaPolicy(Protocol):
    """Interface for meta-cognitive policies."""
    
    def choose(
        self, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Choose control signals based on context.
        
        Args:
            context: Current planning context including:
                - state: Current game state
                - goals: Player goals
                - history: Previous plans and results
                - iteration: Current iteration number
                - time_remaining: Time remaining in budget
                - best_score: Best score so far
                - last_improvement: Improvement in last iteration
                - variance: Variance across samples
                - unpredictability: User unpredictability setting
                
        Returns:
            Control signals dictionary with keys:
                - use: 'llm' | 'search' | 'repair' | 'stop'
                - temperature: float (0.0-2.0)
                - max_samples: int
                - opponent_model: str
                - search_depth: int
                - stop: bool
        """
        ...
    
    def should_continue(
        self,
        context: Dict[str, Any]
    ) -> bool:
        """Determine if planning should continue.
        
        Args:
            context: Current planning context
            
        Returns:
            True if planning should continue, False otherwise
        """
        ...
    
    def adjust_parameters(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust planning parameters based on context.
        
        Args:
            context: Current planning context
            
        Returns:
            Adjusted parameters dictionary
        """
        ...
