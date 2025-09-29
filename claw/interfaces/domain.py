"""Domain adapter interface."""

from typing import Protocol, Dict, Any
from claw.types import Plan


class DomainAdapter(Protocol):
    """Interface for domain-specific adapters."""
    
    def normalize_state(self, raw_state: Any) -> Dict[str, Any]:
        """Normalize raw state to framework format.
        
        Args:
            raw_state: Raw state from domain engine
            
        Returns:
            Normalized state dictionary
        """
        ...
    
    def serialize_for_prompt(self, state: Dict[str, Any]) -> str:
        """Serialize state for LLM prompting.
        
        Args:
            state: Normalized state
            
        Returns:
            String representation for prompts
        """
        ...
    
    def parse_orders_from_text(self, text: str) -> Dict[str, Any]:
        """Parse orders from text (e.g., LLM output).
        
        Args:
            text: Text containing orders
            
        Returns:
            Parsed orders dictionary
        """
        ...
    
    def plan_to_engine_orders(self, plan: Plan) -> Dict[str, Any]:
        """Convert plan to engine-specific orders.
        
        Args:
            plan: Framework plan object
            
        Returns:
            Engine-specific orders
        """
        ...
    
    def engine_orders_to_plan(
        self, 
        orders: Dict[str, Any],
        rationale: str = "",
        confidence: float = 1.0
    ) -> Plan:
        """Convert engine orders to framework plan.
        
        Args:
            orders: Engine-specific orders
            rationale: Plan rationale
            confidence: Plan confidence
            
        Returns:
            Framework plan object
        """
        ...
    
    def goal_schema(self) -> Dict[str, Any]:
        """Get goal schema for this domain.
        
        Returns:
            Dictionary describing valid goal keys and types
        """
        ...
    
    def validate_goals(self, goals: Dict[str, Any]) -> bool:
        """Validate goals against domain schema.
        
        Args:
            goals: Goals to validate
            
        Returns:
            True if goals are valid, False otherwise
        """
        ...
    
    def get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for this domain.
        
        Returns:
            Initial state dictionary
        """
        ...
