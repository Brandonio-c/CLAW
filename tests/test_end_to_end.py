"""End-to-end tests for CLAW framework."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from claw.meta.controller import MetaCognitiveController
from claw.neural.hf_llm import HuggingFaceLLM
from claw.symbolic.diplomacy_adapter.adapter import DiplomacyAdapter
from claw.types import Plan, PlanStatus


class MockHuggingFaceLLM:
    """Mock HuggingFace LLM for testing."""
    
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
    
    def generate_plan(self, state, goals, *, temperature=0.7, max_samples=3):
        plans = []
        for i in range(max_samples):
            plan = Plan(
                orders={
                    f"order_{i}": {
                        "unit": "A LVP",
                        "action": "HLD" if i == 0 else "MTO",
                        "target": "EDI" if i > 0 else None
                    }
                },
                rationale=f"Mock LLM plan {i+1}",
                confidence=0.6 + i * 0.1
            )
            plans.append(plan)
        return plans
    
    def repair_plan(self, state, plan, feedback, *, temperature=0.5):
        repaired = Plan(
            orders=plan.orders.copy(),
            rationale=f"Repaired: {plan.rationale}",
            confidence=min(1.0, plan.confidence + 0.1),
            status=PlanStatus.REPAIRED
        )
        return repaired
    
    def validate_plan(self, state, plan):
        return plan.confidence > 0.3


@patch('claw.symbolic.diplomacy_adapter.adapter.DiplomacyEngine')
def test_end_to_end_plan_generation(mock_engine_class):
    """Test complete end-to-end plan generation."""
    # Mock engine instance
    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine
    mock_engine.get_state.return_value = {
        'units': {'ENGLAND': ['A LVP', 'F LON']},
        'centers': {'ENGLAND': ['LVP', 'LON']},
        'phase': 'SPRING 1901',
        'powers': ['ENGLAND', 'FRANCE', 'GERMANY'],
        'current_power': 'ENGLAND',
        'map': {
            'provinces': ['LVP', 'LON', 'EDI', 'PAR'],
            'supply_centers': ['LVP', 'LON', 'EDI', 'PAR'],
            'coasts': []
        }
    }
    mock_engine.get_legal_orders.return_value = [
        'A LVP HLD', 'A LVP MTO EDI', 'F LON HLD', 'F LON MTO ENG'
    ]
    mock_engine.get_units.return_value = ['A LVP', 'F LON']
    mock_engine.get_centers.return_value = ['LVP', 'LON']
    mock_engine.is_legal_order.return_value = True
    
    # Initialize components
    llm = MockHuggingFaceLLM(model_path="/mock/model")
    adapter = DiplomacyAdapter()
    
    # Create temporary trace file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        trace_file = f.name
    
    try:
        controller = MetaCognitiveController(
            neural_generator=llm,
            symbolic_planner=adapter,
            domain_adapter=adapter,
            trace_file=trace_file
        )
        
        # Test plan generation
        state = adapter.get_initial_state()
        goals = {
            'defend': ['LVP'],
            'attack': ['EDI']
        }
        
        plan = controller.generate_plan(
            state=state,
            goals=goals,
            unpredictability=0.3,
            max_iters=3,
            time_budget=10.0
        )
        
        # Verify plan
        assert isinstance(plan, Plan)
        assert plan.rationale is not None
        assert 0.0 <= plan.confidence <= 1.0
        assert plan.orders is not None
        
        # Verify trace file was created
        assert Path(trace_file).exists()
        
        # Verify trace file contains events
        with open(trace_file, 'r') as f:
            events = [json.loads(line) for line in f if line.strip()]
        
        assert len(events) > 0
        event_types = {event['event_type'] for event in events}
        assert 'planning_start' in event_types
        assert 'planning_complete' in event_types
        
    finally:
        # Clean up
        Path(trace_file).unlink(missing_ok=True)


@patch('claw.symbolic.diplomacy_adapter.adapter.DiplomacyEngine')
def test_end_to_end_with_repair(mock_engine_class):
    """Test end-to-end with plan repair."""
    # Mock engine instance
    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine
    mock_engine.get_state.return_value = {
        'units': {'ENGLAND': ['A LVP']},
        'centers': {'ENGLAND': ['LVP']},
        'phase': 'SPRING 1901',
        'powers': ['ENGLAND'],
        'current_power': 'ENGLAND',
        'map': {
            'provinces': ['LVP', 'EDI'],
            'supply_centers': ['LVP', 'EDI'],
            'coasts': []
        }
    }
    mock_engine.get_legal_orders.return_value = ['A LVP HLD']
    mock_engine.get_units.return_value = ['A LVP']
    mock_engine.get_centers.return_value = ['LVP']
    mock_engine.is_legal_order.return_value = True
    
    # Initialize components
    llm = MockHuggingFaceLLM(model_path="/mock/model")
    adapter = DiplomacyAdapter()
    
    controller = MetaCognitiveController(
        neural_generator=llm,
        symbolic_planner=adapter,
        domain_adapter=adapter
    )
    
    # Test with goals that might trigger repair
    state = adapter.get_initial_state()
    goals = {
        'defend': ['LVP'],
        'attack': ['EDI']
    }
    
    plan = controller.generate_plan(
        state=state,
        goals=goals,
        unpredictability=0.5,  # Higher unpredictability
        max_iters=5,
        time_budget=15.0
    )
    
    # Verify plan
    assert isinstance(plan, Plan)
    assert plan.rationale is not None
    assert 0.0 <= plan.confidence <= 1.0


@patch('claw.symbolic.diplomacy_adapter.adapter.DiplomacyEngine')
def test_end_to_end_error_handling(mock_engine_class):
    """Test end-to-end error handling."""
    # Mock engine that raises exceptions
    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine
    mock_engine.get_state.side_effect = Exception("Engine error")
    
    # Initialize components
    llm = MockHuggingFaceLLM(model_path="/mock/model")
    adapter = DiplomacyAdapter()
    
    controller = MetaCognitiveController(
        neural_generator=llm,
        symbolic_planner=adapter,
        domain_adapter=adapter
    )
    
    # Test that errors are handled gracefully
    state = {"units": {"ENGLAND": ["A LVP"]}}
    goals = {"defend": ["LVP"]}
    
    # Should not raise exception, should return fallback plan
    plan = controller.generate_plan(
        state=state,
        goals=goals,
        unpredictability=0.3,
        max_iters=2,
        time_budget=5.0
    )
    
    # Should get fallback plan
    assert isinstance(plan, Plan)
    assert "fallback" in plan.rationale.lower()


def test_end_to_end_interface_compatibility():
    """Test that all interfaces work together correctly."""
    # This test verifies that the interfaces are properly implemented
    # and can be used together without type errors
    
    from claw.interfaces.neural import NeuralStrategyGenerator
    from claw.interfaces.symbolic import SymbolicPlannerInterface
    from claw.interfaces.domain import DomainAdapter
    from claw.interfaces.meta import MetaPolicy
    
    # Test that our implementations satisfy the interfaces
    assert issubclass(MockHuggingFaceLLM, NeuralStrategyGenerator)
    
    # Test that we can create instances
    llm = MockHuggingFaceLLM(model_path="/mock/model")
    assert hasattr(llm, 'generate_plan')
    assert hasattr(llm, 'repair_plan')
    assert hasattr(llm, 'validate_plan')
    
    # Test method signatures
    state = {"units": {"ENGLAND": ["A LVP"]}}
    goals = {"defend": ["LVP"]}
    
    plans = llm.generate_plan(state, goals, temperature=0.7, max_samples=2)
    assert isinstance(plans, list)
    assert all(isinstance(plan, Plan) for plan in plans)
    
    plan = plans[0]
    repaired = llm.repair_plan(state, plan, {"error": "test"})
    assert isinstance(repaired, Plan)
    
    is_valid = llm.validate_plan(state, plan)
    assert isinstance(is_valid, bool)


@patch('claw.symbolic.diplomacy_adapter.adapter.DiplomacyEngine')
def test_end_to_end_trace_logging(mock_engine_class):
    """Test end-to-end trace logging."""
    # Mock engine instance
    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine
    mock_engine.get_state.return_value = {
        'units': {'ENGLAND': ['A LVP']},
        'centers': {'ENGLAND': ['LVP']},
        'phase': 'SPRING 1901',
        'powers': ['ENGLAND'],
        'current_power': 'ENGLAND',
        'map': {
            'provinces': ['LVP'],
            'supply_centers': ['LVP'],
            'coasts': []
        }
    }
    mock_engine.get_legal_orders.return_value = ['A LVP HLD']
    mock_engine.get_units.return_value = ['A LVP']
    mock_engine.get_centers.return_value = ['LVP']
    mock_engine.is_legal_order.return_value = True
    
    # Create temporary trace file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        trace_file = f.name
    
    try:
        # Initialize components
        llm = MockHuggingFaceLLM(model_path="/mock/model")
        adapter = DiplomacyAdapter()
        
        controller = MetaCognitiveController(
            neural_generator=llm,
            symbolic_planner=adapter,
            domain_adapter=adapter,
            trace_file=trace_file
        )
        
        # Generate plan
        state = adapter.get_initial_state()
        goals = {"defend": ["LVP"]}
        
        plan = controller.generate_plan(
            state=state,
            goals=goals,
            unpredictability=0.3,
            max_iters=2,
            time_budget=5.0
        )
        
        # Verify trace file
        assert Path(trace_file).exists()
        
        with open(trace_file, 'r') as f:
            events = [json.loads(line) for line in f if line.strip()]
        
        # Should have multiple events
        assert len(events) >= 2
        
        # Check for expected event types
        event_types = {event['event_type'] for event in events}
        assert 'planning_start' in event_types
        assert 'planning_complete' in event_types
        
        # Check event structure
        for event in events:
            assert 'timestamp' in event
            assert 'event_type' in event
            assert 'data' in event
            assert isinstance(event['timestamp'], (int, float))
            assert isinstance(event['event_type'], str)
            assert isinstance(event['data'], dict)
        
    finally:
        # Clean up
        Path(trace_file).unlink(missing_ok=True)


def test_end_to_end_plan_validation():
    """Test end-to-end plan validation."""
    from claw.neural.schema import parse_plan_from_json, ValidationError
    
    # Test valid plan JSON
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
    assert len(plan.orders) == 2
    
    # Test invalid plan JSON
    invalid_json = json.dumps({
        "orders": [],  # Empty orders should fail
        "rationale": "Invalid plan",
        "confidence": 0.5
    })
    
    with pytest.raises(ValidationError):
        parse_plan_from_json(invalid_json)
    
    # Test malformed JSON
    with pytest.raises(ValidationError):
        parse_plan_from_json("invalid json")


def test_end_to_end_configuration():
    """Test end-to-end configuration loading."""
    from claw.config import Config, LLMConfig, MetaConfig
    
    # Test default configuration
    config = Config()
    assert isinstance(config.llm, LLMConfig)
    assert isinstance(config.meta, MetaConfig)
    assert config.llm.temperature == 0.7
    assert config.meta.max_iters_per_turn == 5
    
    # Test configuration validation
    assert 0.0 <= config.llm.temperature <= 2.0
    assert config.meta.max_iters_per_turn > 0
    assert config.meta.time_budget_warning > 0.0
    assert config.meta.time_budget_warning < 1.0
