"""Neural strategy generator interface."""

from typing import Protocol, Dict, Any, List
from claw.types import Plan


class NeuralStrategyGenerator(Protocol):
    """Interface for neural strategy generators (LLMs)."""
    
    def generate_plan(
        self, 
        state: Dict[str, Any], 
        goals: Dict[str, Any], 
        *,
        temperature: float = 0.7,
        max_samples: int = 3
    ) -> List[Plan]:
        """Generate candidate plans using neural methods.
        
        Args:
            state: Current game state
            goals: Player goals and constraints
            temperature: Sampling temperature for generation
            max_samples: Maximum number of plans to generate
            
        Returns:
            List of candidate plans with rationale and confidence
        """
        ...
    
    def repair_plan(
        self,
        state: Dict[str, Any],
        plan: Plan,
        feedback: Dict[str, Any],
        *,
        temperature: float = 0.5
    ) -> Plan:
        """Repair a plan based on feedback.
        
        Args:
            state: Current game state
            plan: Plan to repair
            feedback: Feedback from simulation/evaluation
            temperature: Sampling temperature for repair
            
        Returns:
            Repaired plan
        """
        ...
    
    def validate_plan(
        self,
        state: Dict[str, Any],
        plan: Plan
    ) -> bool:
        """Validate a plan for basic correctness.
        
        Args:
            state: Current game state
            plan: Plan to validate
            
        Returns:
            True if plan is valid, False otherwise
        """
        ...
