"""Tests for LLM schema validation."""

import pytest
import json
from claw.neural.schema import (
    OrderSchema, 
    PlanSchema, 
    ValidationError, 
    parse_plan_from_json,
    validate_plan_structure
)
from claw.types import Plan, PlanStatus


def test_order_schema():
    """Test order schema validation."""
    # Valid order
    valid_order = OrderSchema(
        unit="A LVP",
        action="MTO",
        target="EDI"
    )
    assert valid_order.unit == "A LVP"
    assert valid_order.action == "MTO"
    assert valid_order.target == "EDI"
    
    # Order with support
    support_order = OrderSchema(
        unit="F LON",
        action="SUP",
        support="A LVP",
        target="EDI"
    )
    assert support_order.support == "A LVP"
    
    # Order with convoy
    convoy_order = OrderSchema(
        unit="F ENG",
        action="CVY",
        convoy="A LVP",
        target="EDI"
    )
    assert convoy_order.convoy == "A LVP"


def test_plan_schema():
    """Test plan schema validation."""
    # Valid plan
    valid_plan = PlanSchema(
        orders=[
            OrderSchema(unit="A LVP", action="HLD"),
            OrderSchema(unit="F LON", action="MTO", target="ENG")
        ],
        rationale="Test plan",
        confidence=0.8
    )
    assert len(valid_plan.orders) == 2
    assert valid_plan.rationale == "Test plan"
    assert valid_plan.confidence == 0.8
    
    # Test confidence bounds
    with pytest.raises(ValueError):
        PlanSchema(
            orders=[OrderSchema(unit="A LVP", action="HLD")],
            rationale="Invalid confidence",
            confidence=1.5  # Too high
        )
    
    with pytest.raises(ValueError):
        PlanSchema(
            orders=[OrderSchema(unit="A LVP", action="HLD")],
            rationale="Invalid confidence",
            confidence=-0.1  # Too low
        )
    
    # Test empty orders
    with pytest.raises(ValueError):
        PlanSchema(
            orders=[],
            rationale="No orders",
            confidence=0.5
        )


def test_plan_schema_to_plan():
    """Test conversion from schema to Plan object."""
    schema = PlanSchema(
        orders=[
            OrderSchema(unit="A LVP", action="HLD"),
            OrderSchema(unit="F LON", action="MTO", target="ENG")
        ],
        rationale="Test plan",
        confidence=0.8
    )
    
    plan = schema.to_plan(PlanStatus.LEGAL)
    assert isinstance(plan, Plan)
    assert plan.rationale == "Test plan"
    assert plan.confidence == 0.8
    assert plan.status == PlanStatus.LEGAL
    assert len(plan.orders) == 2


def test_parse_plan_from_json():
    """Test parsing plan from JSON string."""
    # Valid JSON
    valid_json = json.dumps({
        "orders": [
            {"unit": "A LVP", "action": "HLD"},
            {"unit": "F LON", "action": "MTO", "target": "ENG"}
        ],
        "rationale": "Test plan",
        "confidence": 0.8
    })
    
    plan = parse_plan_from_json(valid_json)
    assert isinstance(plan, Plan)
    assert plan.rationale == "Test plan"
    assert plan.confidence == 0.8
    
    # Invalid JSON
    with pytest.raises(ValidationError):
        parse_plan_from_json("invalid json")
    
    # Valid JSON but invalid schema
    invalid_json = json.dumps({
        "orders": [],
        "rationale": "No orders",
        "confidence": 0.5
    })
    
    with pytest.raises(ValidationError):
        parse_plan_from_json(invalid_json)


def test_validate_plan_structure():
    """Test plan structure validation."""
    # Valid plan
    valid_plan = Plan(
        orders={
            "order_0": {"unit": "A LVP", "action": "HLD"},
            "order_1": {"unit": "F LON", "action": "MTO", "target": "ENG"}
        },
        rationale="Valid plan",
        confidence=0.8
    )
    
    errors = validate_plan_structure(valid_plan)
    assert len(errors) == 0
    
    # Invalid confidence
    invalid_plan = Plan(
        orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
        rationale="Invalid confidence",
        confidence=1.5
    )
    
    errors = validate_plan_structure(invalid_plan)
    assert len(errors) > 0
    assert any("Confidence" in error for error in errors)
    
    # Empty orders
    empty_plan = Plan(
        orders={},
        rationale="Empty orders",
        confidence=0.5
    )
    
    errors = validate_plan_structure(empty_plan)
    assert len(errors) > 0
    assert any("no orders" in error.lower() for error in errors)
    
    # Invalid order structure
    invalid_order_plan = Plan(
        orders={"order_0": "not a dict"},
        rationale="Invalid order structure",
        confidence=0.5
    )
    
    errors = validate_plan_structure(invalid_order_plan)
    assert len(errors) > 0
    assert any("not a dictionary" in error for error in errors)
    
    # Missing required fields
    incomplete_plan = Plan(
        orders={"order_0": {"unit": "A LVP"}},  # Missing action
        rationale="Incomplete order",
        confidence=0.5
    )
    
    errors = validate_plan_structure(incomplete_plan)
    assert len(errors) > 0
    assert any("missing required field" in error for error in errors)


def test_validation_error():
    """Test ValidationError exception."""
    error = ValidationError(
        message="Test error",
        raw_output="raw text",
        errors=["error1", "error2"]
    )
    
    assert str(error) == "Test error"
    assert error.raw_output == "raw text"
    assert error.errors == ["error1", "error2"]


def test_complex_plan_parsing():
    """Test parsing complex plans with various order types."""
    complex_json = json.dumps({
        "orders": [
            {"unit": "A LVP", "action": "HLD"},
            {"unit": "F LON", "action": "MTO", "target": "ENG"},
            {"unit": "A PAR", "action": "SUP", "support": "A LVP", "target": "EDI"},
            {"unit": "F ENG", "action": "CVY", "convoy": "A LVP", "target": "EDI"},
            {"unit": "A BUR", "action": "RTO", "target": "PAR"},
            {"unit": "F NTH", "action": "DSB"}
        ],
        "rationale": "Complex plan with all order types",
        "confidence": 0.9
    })
    
    plan = parse_plan_from_json(complex_json)
    assert isinstance(plan, Plan)
    assert len(plan.orders) == 6
    assert plan.confidence == 0.9
    
    # Check that all order types are preserved
    order_types = set()
    for order_data in plan.orders.values():
        if isinstance(order_data, dict):
            order_types.add(order_data.get('action', ''))
    
    expected_actions = {'HLD', 'MTO', 'SUP', 'CVY', 'RTO', 'DSB'}
    assert order_types == expected_actions
