"""Symbolic planner interface."""

from typing import Protocol, Dict, Any, List, Optional
from dataclasses import dataclass
from claw.types import Plan


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    new_state: Dict[str, Any]
    is_legal: bool
    score: float  # Higher is better
    diagnostics: Dict[str, Any]  # Illegal orders, bounced moves, etc.
    metadata: Optional[Dict[str, Any]] = None


class SymbolicPlannerInterface(Protocol):
    """Interface for symbolic planners and simulators."""
    
    def enumerate_legal_orders(
        self, 
        state: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Enumerate all legal orders for each unit.
        
        Args:
            state: Current game state
            
        Returns:
            Dictionary mapping unit names to lists of legal orders
        """
        ...
    
    def simulate(
        self, 
        state: Dict[str, Any], 
        plan: Plan,
        opponent_model: str = "random", 
        samples: int = 1
    ) -> List[SimulationResult]:
        """Simulate a plan against opponent models.
        
        Args:
            state: Current game state
            plan: Plan to simulate
            opponent_model: Type of opponent model ("random", "best", "heuristic")
            samples: Number of simulation samples
            
        Returns:
            List of simulation results
        """
        ...
    
    def repair(
        self, 
        state: Dict[str, Any], 
        plan: Plan
    ) -> Optional[Plan]:
        """Attempt to repair an illegal plan.
        
        Args:
            state: Current game state
            plan: Plan to repair
            
        Returns:
            Repaired plan if successful, None otherwise
        """
        ...
    
    def score_plan(
        self,
        state: Dict[str, Any],
        plan: Plan,
        goals: Dict[str, Any]
    ) -> float:
        """Score a plan based on goals and heuristics.
        
        Args:
            state: Current game state
            plan: Plan to score
            goals: Player goals
            
        Returns:
            Plan score (higher is better)
        """
        ...
    
    def search_plans(
        self,
        state: Dict[str, Any],
        goals: Dict[str, Any],
        max_depth: int = 2,
        max_plans: int = 10
    ) -> List[Plan]:
        """Search for plans using symbolic methods.
        
        Args:
            state: Current game state
            goals: Player goals
            max_depth: Maximum search depth
            max_plans: Maximum number of plans to return
            
        Returns:
            List of discovered plans
        """
        ...
