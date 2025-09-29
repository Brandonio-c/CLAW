"""Tests for Diplomacy adapter components."""

import pytest
from unittest.mock import Mock, patch
from claw.symbolic.diplomacy_adapter.orders import OrderParser, OrderValidator
from claw.symbolic.diplomacy_adapter.score import DiplomacyScorer, ScoringWeights
from claw.symbolic.diplomacy_adapter.adapter import DiplomacyAdapter
from claw.types import Plan, PlanStatus


def test_order_parser():
    """Test order parsing functionality."""
    parser = OrderParser()
    
    # Test valid orders
    valid_orders = [
        "A LVP HLD",
        "F LON MTO ENG",
        "A PAR SUP A LVP EDI",
        "F ENG CVY A LVP EDI",
        "A BUR RTO PAR",
        "F NTH DSB"
    ]
    
    for order_str in valid_orders:
        parsed = parser.parse_order(order_str)
        assert parsed is not None
        assert parsed.original == order_str.upper()
    
    # Test invalid orders
    invalid_orders = [
        "INVALID ORDER",
        "A LVP",
        "F LON MTO",
        "A PAR SUP",
        ""
    ]
    
    for order_str in invalid_orders:
        parsed = parser.parse_order(order_str)
        assert parsed is None
    
    # Test order formatting
    parsed = parser.parse_order("A LVP HLD")
    formatted = parser.format_order(parsed)
    assert formatted == "A LVP HLD"
    
    parsed = parser.parse_order("F LON MTO ENG")
    formatted = parser.format_order(parsed)
    assert formatted == "F LON MTO ENG"


def test_order_parser_complex():
    """Test parsing complex orders."""
    parser = OrderParser()
    
    # Test support order
    support_order = "A PAR SUP A LVP EDI"
    parsed = parser.parse_order(support_order)
    assert parsed is not None
    assert parsed.unit == "A PAR"
    assert parsed.action == "SUP"
    assert parsed.support == "A LVP"
    assert parsed.target == "EDI"
    
    # Test convoy order
    convoy_order = "F ENG CVY A LVP EDI"
    parsed = parser.parse_order(convoy_order)
    assert parsed is not None
    assert parsed.unit == "F ENG"
    assert parsed.action == "CVY"
    assert parsed.convoy == "A LVP"
    assert parsed.target == "EDI"
    
    # Test case insensitive parsing
    mixed_case = "a lvp hld"
    parsed = parser.parse_order(mixed_case)
    assert parsed is not None
    assert parsed.unit == "A LVP"
    assert parsed.action == "HLD"


def test_order_validator():
    """Test order validation functionality."""
    # Mock engine
    mock_engine = Mock()
    mock_engine.is_legal_order.return_value = True
    mock_engine.get_units.return_value = ["A LVP", "F LON"]
    
    validator = OrderValidator(mock_engine)
    
    # Test valid order
    is_valid, errors = validator.validate_order("ENGLAND", "A LVP HLD")
    assert is_valid == True
    assert len(errors) == 0
    
    # Test invalid order (not legal)
    mock_engine.is_legal_order.return_value = False
    is_valid, errors = validator.validate_order("ENGLAND", "A LVP MTO INVALID")
    assert is_valid == False
    assert len(errors) > 0
    
    # Test order for non-existent unit
    mock_engine.get_units.return_value = ["A PAR"]
    is_valid, errors = validator.validate_order("ENGLAND", "A LVP HLD")
    assert is_valid == False
    assert any("does not exist" in error for error in errors)
    
    # Test multiple orders
    orders = ["A LVP HLD", "F LON MTO ENG", "INVALID ORDER"]
    result = validator.validate_orders("ENGLAND", orders)
    assert result['valid'] == False
    assert len(result['legal_orders']) == 2
    assert len(result['illegal_orders']) == 1


def test_diplomacy_scorer():
    """Test Diplomacy scoring functionality."""
    scorer = DiplomacyScorer()
    
    # Test basic scoring
    state = {
        'units': {'ENGLAND': ['A LVP', 'F LON']},
        'centers': {'ENGLAND': ['LVP', 'LON']},
        'map': {
            'supply_centers': ['LVP', 'LON', 'EDI', 'PAR']
        }
    }
    
    plan = Plan(
        orders={
            'order_0': {'unit': 'A LVP', 'action': 'MTO', 'target': 'EDI'},
            'order_1': {'unit': 'F LON', 'action': 'HLD'}
        },
        rationale="Test plan",
        confidence=0.8
    )
    
    goals = {
        'defend': ['LVP'],
        'attack': ['EDI']
    }
    
    score = scorer.score_plan(state, plan, goals)
    assert isinstance(score.total, float)
    assert 'center_gain' in score.breakdown
    assert 'positional_advantage' in score.breakdown
    assert 'risk' in score.breakdown
    assert 'goal_fit' in score.breakdown
    assert 'confidence' in score.breakdown
    
    # Test with custom weights
    custom_weights = ScoringWeights(
        center_gain=2.0,
        positional_advantage=1.0,
        risk_penalty=-0.5,
        goal_fit=1.5,
        confidence_bonus=0.3
    )
    
    custom_scorer = DiplomacyScorer(custom_weights)
    custom_score = custom_scorer.score_plan(state, plan, goals)
    assert custom_score.total != score.total


def test_diplomacy_scorer_goals():
    """Test scoring with different goal types."""
    scorer = DiplomacyScorer()
    
    state = {
        'units': {'ENGLAND': ['A LVP', 'F LON']},
        'centers': {'ENGLAND': ['LVP', 'LON']},
        'map': {
            'supply_centers': ['LVP', 'LON', 'EDI', 'PAR']
        }
    }
    
    # Test defend goals
    defend_plan = Plan(
        orders={
            'order_0': {'unit': 'A LVP', 'action': 'SUP', 'support': 'F LON', 'target': 'LVP'}
        },
        rationale="Defend LVP",
        confidence=0.7
    )
    
    defend_goals = {'defend': ['LVP']}
    defend_score = scorer.score_plan(state, defend_plan, defend_goals)
    assert defend_score.total > 0
    
    # Test attack goals
    attack_plan = Plan(
        orders={
            'order_0': {'unit': 'A LVP', 'action': 'MTO', 'target': 'EDI'}
        },
        rationale="Attack EDI",
        confidence=0.8
    )
    
    attack_goals = {'attack': ['EDI']}
    attack_score = scorer.score_plan(state, attack_plan, attack_goals)
    assert attack_score.total > 0


@patch('claw.symbolic.diplomacy_adapter.adapter.DiplomacyEngine')
def test_diplomacy_adapter(mock_engine_class):
    """Test Diplomacy adapter functionality."""
    # Mock engine instance
    mock_engine = Mock()
    mock_engine_class.return_value = mock_engine
    mock_engine.get_state.return_value = {
        'units': {'ENGLAND': ['A LVP']},
        'centers': {'ENGLAND': ['LVP']},
        'phase': 'SPRING 1901',
        'powers': ['ENGLAND', 'FRANCE'],
        'current_power': 'ENGLAND',
        'map': {
            'provinces': ['LVP', 'LON', 'EDI'],
            'supply_centers': ['LVP', 'LON', 'EDI'],
            'coasts': []
        }
    }
    mock_engine.get_legal_orders.return_value = ['A LVP HLD', 'A LVP MTO EDI']
    mock_engine.get_units.return_value = ['A LVP']
    mock_engine.get_centers.return_value = ['LVP']
    mock_engine.is_legal_order.return_value = True
    
    adapter = DiplomacyAdapter()
    
    # Test state normalization
    raw_state = {'test': 'data'}
    normalized = adapter.normalize_state(raw_state)
    assert isinstance(normalized, dict)
    
    # Test state serialization
    state = adapter.get_initial_state()
    serialized = adapter.serialize_for_prompt(state)
    assert isinstance(serialized, str)
    assert 'Phase:' in serialized
    assert 'Units:' in serialized
    
    # Test order parsing
    orders = adapter.parse_orders_from_text("A LVP HLD\nF LON MTO ENG")
    assert 'orders' in orders
    assert len(orders['orders']) == 2
    
    # Test plan conversion
    plan = Plan(
        orders={'order_0': {'unit': 'A LVP', 'action': 'HLD'}},
        rationale="Test plan",
        confidence=0.8
    )
    
    engine_orders = adapter.plan_to_engine_orders(plan)
    assert isinstance(engine_orders, dict)
    
    converted_plan = adapter.engine_orders_to_plan(engine_orders, "Test", 0.8)
    assert isinstance(converted_plan, Plan)
    
    # Test goal schema
    schema = adapter.goal_schema()
    assert 'defend' in schema
    assert 'attack' in schema
    assert 'ally' in schema
    assert 'avoid' in schema
    
    # Test goal validation
    valid_goals = {'defend': ['LVP'], 'attack': ['EDI']}
    assert adapter.validate_goals(valid_goals) == True
    
    invalid_goals = {'defend': 'LVP'}  # Should be list
    assert adapter.validate_goals(invalid_goals) == False
    
    # Test initial state
    initial_state = adapter.get_initial_state()
    assert isinstance(initial_state, dict)
    assert 'units' in initial_state


def test_diplomacy_adapter_simulation():
    """Test Diplomacy adapter simulation functionality."""
    with patch('claw.symbolic.diplomacy_adapter.adapter.DiplomacyEngine'):
        adapter = DiplomacyAdapter()
        
        state = {
            'units': {'ENGLAND': ['A LVP']},
            'centers': {'ENGLAND': ['LVP']},
            'current_power': 'ENGLAND'
        }
        
        plan = Plan(
            orders={'order_0': {'unit': 'A LVP', 'action': 'HLD'}},
            rationale="Test plan",
            confidence=0.8
        )
        
        # Test simulation
        results = adapter.simulate(state, plan, opponent_model="random", samples=2)
        assert len(results) == 2
        assert all(hasattr(result, 'is_legal') for result in results)
        assert all(hasattr(result, 'score') for result in results)
        
        # Test repair
        low_confidence_plan = Plan(
            orders={'order_0': {'unit': 'A LVP', 'action': 'HLD'}},
            rationale="Low confidence",
            confidence=0.3
        )
        
        repaired = adapter.repair(state, low_confidence_plan)
        # Repair might succeed or fail depending on implementation
        if repaired:
            assert isinstance(repaired, Plan)
        
        # Test scoring
        goals = {'defend': ['LVP']}
        score = adapter.score_plan(state, plan, goals)
        assert isinstance(score, float)
        
        # Test search
        search_plans = adapter.search_plans(state, goals, max_depth=2, max_plans=5)
        assert len(search_plans) <= 5
        assert all(isinstance(p, Plan) for p in search_plans)


def test_diplomacy_adapter_legal_orders():
    """Test legal orders enumeration."""
    with patch('claw.symbolic.diplomacy_adapter.adapter.DiplomacyEngine'):
        adapter = DiplomacyAdapter()
        
        state = {
            'units': {'ENGLAND': ['A LVP']},
            'current_power': 'ENGLAND'
        }
        
        legal_orders = adapter.enumerate_legal_orders(state)
        assert isinstance(legal_orders, dict)
        # Should have orders for A LVP
        assert 'A LVP' in legal_orders
