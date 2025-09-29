"""Pydantic schemas for structured LLM outputs."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from claw.types import Plan, Order, PlanStatus


class OrderSchema(BaseModel):
    """Schema for a single order."""
    unit: str = Field(..., description="Unit identifier (e.g., 'A LVP')")
    action: str = Field(..., description="Action type (e.g., 'MTO', 'SUP', 'HLD')")
    target: Optional[str] = Field(None, description="Target location or unit")
    support: Optional[str] = Field(None, description="Unit being supported")
    convoy: Optional[str] = Field(None, description="Unit being convoyed")
    
    class Config:
        extra = "forbid"


class PlanSchema(BaseModel):
    """Schema for a complete plan."""
    orders: List[OrderSchema] = Field(..., description="List of orders")
    rationale: str = Field(..., description="Explanation of the plan")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    
    @validator('orders')
    def validate_orders(cls, v):
        """Validate that orders list is not empty."""
        if not v:
            raise ValueError("Plan must contain at least one order")
        return v
    
    def to_plan(self, status: PlanStatus = PlanStatus.PENDING) -> Plan:
        """Convert to Plan object."""
        orders_dict = {
            f"order_{i}": order.dict() for i, order in enumerate(self.orders)
        }
        return Plan(
            orders=orders_dict,
            rationale=self.rationale,
            confidence=self.confidence,
            status=status
        )


class RepairRequestSchema(BaseModel):
    """Schema for plan repair requests."""
    original_plan: PlanSchema = Field(..., description="Original plan to repair")
    feedback: Dict[str, Any] = Field(..., description="Feedback from simulation")
    focus_areas: List[str] = Field(default_factory=list, description="Areas to focus on")


class ValidationError(Exception):
    """Raised when LLM output validation fails."""
    
    def __init__(self, message: str, raw_output: str, errors: List[str]):
        super().__init__(message)
        self.raw_output = raw_output
        self.errors = errors


def parse_plan_from_json(
    json_str: str, 
    status: PlanStatus = PlanStatus.PENDING
) -> Plan:
    """Parse a plan from JSON string with error handling.
    
    Args:
        json_str: JSON string containing plan data
        status: Initial status for the plan
        
    Returns:
        Parsed Plan object
        
    Raises:
        ValidationError: If parsing or validation fails
    """
    try:
        import json
        data = json.loads(json_str)
        plan_schema = PlanSchema(**data)
        return plan_schema.to_plan(status)
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON: {e}",
            json_str,
            [str(e)]
        )
    except Exception as e:
        raise ValidationError(
            f"Validation failed: {e}",
            json_str,
            [str(e)]
        )


def validate_plan_structure(plan: Plan) -> List[str]:
    """Validate plan structure and return any errors.
    
    Args:
        plan: Plan to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check confidence range
    if not 0.0 <= plan.confidence <= 1.0:
        errors.append(f"Confidence {plan.confidence} not in range [0.0, 1.0]")
    
    # Check orders structure
    if not plan.orders:
        errors.append("Plan has no orders")
    else:
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                errors.append(f"Order {order_key} is not a dictionary")
            else:
                required_fields = ['unit', 'action']
                for field in required_fields:
                    if field not in order_data:
                        errors.append(f"Order {order_key} missing required field: {field}")
    
    return errors
