"""Tests for interface implementations."""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock

from claw.interfaces.neural import NeuralStrategyGenerator
from claw.interfaces.symbolic import SymbolicPlannerInterface, SimulationResult
from claw.interfaces.domain import DomainAdapter
from claw.interfaces.meta import MetaPolicy
from claw.types import Plan, PlanStatus


class MockNeuralStrategyGenerator(NeuralStrategyGenerator):
    """Mock implementation of NeuralStrategyGenerator."""
    
    def generate_plan(
        self, 
        state: Dict[str, Any], 
        goals: Dict[str, Any], 
        *,
        temperature: float = 0.7,
        max_samples: int = 3
    ) -> List[Plan]:
        """Generate mock plans."""
        plans = []
        for i in range(max_samples):
            plan = Plan(
                orders={f"order_{i}": {"unit": f"A LVP", "action": "HLD"}},
                rationale=f"Mock plan {i}",
                confidence=0.5 + i * 0.1
            )
            plans.append(plan)
        return plans
    
    def repair_plan(
        self,
        state: Dict[str, Any],
        plan: Plan,
        feedback: Dict[str, Any],
        *,
        temperature: float = 0.5
    ) -> Plan:
        """Repair mock plan."""
        repaired = Plan(
            orders=plan.orders.copy(),
            rationale=f"Repaired: {plan.rationale}",
            confidence=plan.confidence * 0.8,
            status=PlanStatus.REPAIRED
        )
        return repaired
    
    def validate_plan(
        self,
        state: Dict[str, Any],
        plan: Plan
    ) -> bool:
        """Validate mock plan."""
        return plan.confidence > 0.3


class MockSymbolicPlannerInterface(SymbolicPlannerInterface):
    """Mock implementation of SymbolicPlannerInterface."""
    
    def enumerate_legal_orders(
        self, 
        state: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Enumerate mock legal orders."""
        return {
            "A LVP": [
                {"unit": "A LVP", "action": "HLD"},
                {"unit": "A LVP", "action": "MTO", "target": "EDI"}
            ]
        }
    
    def simulate(
        self, 
        state: Dict[str, Any], 
        plan: Plan,
        opponent_model: str = "random", 
        samples: int = 1
    ) -> List[SimulationResult]:
        """Simulate mock plan."""
        results = []
        for i in range(samples):
            result = SimulationResult(
                new_state=state.copy(),
                is_legal=True,
                score=0.5 + i * 0.1,
                diagnostics={"sample": i}
            )
            results.append(result)
        return results
    
    def repair(
        self, 
        state: Dict[str, Any], 
        plan: Plan
    ) -> Plan | None:
        """Repair mock plan."""
        if plan.confidence < 0.5:
            repaired = Plan(
                orders=plan.orders.copy(),
                rationale=f"Repaired: {plan.rationale}",
                confidence=0.6,
                status=PlanStatus.REPAIRED
            )
            return repaired
        return None
    
    def score_plan(
        self,
        state: Dict[str, Any],
        plan: Plan,
        goals: Dict[str, Any]
    ) -> float:
        """Score mock plan."""
        return plan.confidence * 10.0
    
    def search_plans(
        self,
        state: Dict[str, Any],
        goals: Dict[str, Any],
        max_depth: int = 2,
        max_plans: int = 10
    ) -> List[Plan]:
        """Search mock plans."""
        plans = []
        for i in range(min(max_plans, 3)):
            plan = Plan(
                orders={f"search_order_{i}": {"unit": f"A LVP", "action": "HLD"}},
                rationale=f"Search plan {i}",
                confidence=0.4 + i * 0.1
            )
            plans.append(plan)
        return plans


class MockDomainAdapter(DomainAdapter):
    """Mock implementation of DomainAdapter."""
    
    def normalize_state(self, raw_state: Any) -> Dict[str, Any]:
        """Normalize mock state."""
        return {"units": {"ENGLAND": ["A LVP"]}, "phase": "SPRING 1901"}
    
    def serialize_for_prompt(self, state: Dict[str, Any]) -> str:
        """Serialize mock state."""
        return f"Mock state: {state}"
    
    def parse_orders_from_text(self, text: str) -> Dict[str, Any]:
        """Parse mock orders."""
        return {"orders": [{"unit": "A LVP", "action": "HLD"}]}
    
    def plan_to_engine_orders(self, plan: Plan) -> Dict[str, Any]:
        """Convert mock plan to orders."""
        return {"ENGLAND": ["A LVP HLD"]}
    
    def engine_orders_to_plan(
        self, 
        orders: Dict[str, Any],
        rationale: str = "",
        confidence: float = 1.0
    ) -> Plan:
        """Convert mock orders to plan."""
        return Plan(
            orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
            rationale=rationale,
            confidence=confidence
        )
    
    def goal_schema(self) -> Dict[str, Any]:
        """Get mock goal schema."""
        return {
            "defend": {"type": "list", "items": {"type": "string"}},
            "attack": {"type": "list", "items": {"type": "string"}}
        }
    
    def validate_goals(self, goals: Dict[str, Any]) -> bool:
        """Validate mock goals."""
        return isinstance(goals, dict)
    
    def get_initial_state(self) -> Dict[str, Any]:
        """Get mock initial state."""
        return {"units": {"ENGLAND": ["A LVP"]}, "phase": "SPRING 1901"}


class MockMetaPolicy(MetaPolicy):
    """Mock implementation of MetaPolicy."""
    
    def choose(
        self, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Choose mock control signals."""
        return {
            "use": "llm",
            "temperature": 0.7,
            "max_samples": 3,
            "opponent_model": "random",
            "search_depth": 2,
            "stop": False
        }
    
    def should_continue(
        self,
        context: Dict[str, Any]
    ) -> bool:
        """Mock continue decision."""
        return context.get("iteration", 0) < 5
    
    def adjust_parameters(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust mock parameters."""
        return {
            "temperature": 0.7,
            "max_samples": 3,
            "opponent_model": "random",
            "search_depth": 2
        }


def test_neural_strategy_generator():
    """Test neural strategy generator interface."""
    generator = MockNeuralStrategyGenerator()
    
    state = {"units": {"ENGLAND": ["A LVP"]}}
    goals = {"defend": ["LVP"]}
    
    # Test plan generation
    plans = generator.generate_plan(state, goals, temperature=0.7, max_samples=2)
    assert len(plans) == 2
    assert all(isinstance(plan, Plan) for plan in plans)
    
    # Test plan repair
    original_plan = plans[0]
    feedback = {"error": "test"}
    repaired_plan = generator.repair_plan(state, original_plan, feedback)
    assert isinstance(repaired_plan, Plan)
    assert repaired_plan.status == PlanStatus.REPAIRED
    
    # Test plan validation
    assert generator.validate_plan(state, original_plan) == True


def test_symbolic_planner_interface():
    """Test symbolic planner interface."""
    planner = MockSymbolicPlannerInterface()
    
    state = {"units": {"ENGLAND": ["A LVP"]}}
    plan = Plan(
        orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
        rationale="Test plan",
        confidence=0.5
    )
    goals = {"defend": ["LVP"]}
    
    # Test legal orders enumeration
    legal_orders = planner.enumerate_legal_orders(state)
    assert isinstance(legal_orders, dict)
    assert "A LVP" in legal_orders
    
    # Test simulation
    results = planner.simulate(state, plan, opponent_model="random", samples=2)
    assert len(results) == 2
    assert all(isinstance(result, SimulationResult) for result in results)
    
    # Test repair
    low_confidence_plan = Plan(
        orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
        rationale="Low confidence plan",
        confidence=0.3
    )
    repaired = planner.repair(state, low_confidence_plan)
    assert repaired is not None
    assert repaired.status == PlanStatus.REPAIRED
    
    # Test scoring
    score = planner.score_plan(state, plan, goals)
    assert isinstance(score, float)
    assert score > 0
    
    # Test search
    search_plans = planner.search_plans(state, goals, max_depth=2, max_plans=5)
    assert len(search_plans) <= 5
    assert all(isinstance(plan, Plan) for plan in search_plans)


def test_domain_adapter():
    """Test domain adapter interface."""
    adapter = MockDomainAdapter()
    
    raw_state = {"test": "data"}
    goals = {"defend": ["LVP"], "attack": ["EDI"]}
    
    # Test state normalization
    normalized = adapter.normalize_state(raw_state)
    assert isinstance(normalized, dict)
    
    # Test state serialization
    serialized = adapter.serialize_for_prompt(normalized)
    assert isinstance(serialized, str)
    
    # Test order parsing
    orders = adapter.parse_orders_from_text("A LVP HLD")
    assert isinstance(orders, dict)
    assert "orders" in orders
    
    # Test plan conversion
    plan = Plan(
        orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
        rationale="Test plan",
        confidence=0.5
    )
    
    engine_orders = adapter.plan_to_engine_orders(plan)
    assert isinstance(engine_orders, dict)
    
    converted_plan = adapter.engine_orders_to_plan(engine_orders, "Test", 0.8)
    assert isinstance(converted_plan, Plan)
    
    # Test goal schema
    schema = adapter.goal_schema()
    assert isinstance(schema, dict)
    
    # Test goal validation
    assert adapter.validate_goals(goals) == True
    
    # Test initial state
    initial_state = adapter.get_initial_state()
    assert isinstance(initial_state, dict)


def test_meta_policy():
    """Test meta-policy interface."""
    policy = MockMetaPolicy()
    
    context = {
        "iteration": 1,
        "time_remaining": 30.0,
        "best_score": 0.5,
        "last_improvement": 0.1,
        "variance": 0.2,
        "unpredictability": 0.3,
        "consecutive_failures": 0,
        "last_plan_legal": True
    }
    
    # Test choice
    decision = policy.choose(context)
    assert isinstance(decision, dict)
    assert "use" in decision
    assert "temperature" in decision
    
    # Test continue decision
    assert policy.should_continue(context) == True
    
    # Test parameter adjustment
    params = policy.adjust_parameters(context)
    assert isinstance(params, dict)
    assert "temperature" in params


def test_interface_compatibility():
    """Test that all interfaces work together."""
    # Create all mock implementations
    neural = MockNeuralStrategyGenerator()
    symbolic = MockSymbolicPlannerInterface()
    domain = MockDomainAdapter()
    meta = MockMetaPolicy()
    
    # Test that they can be used together
    state = domain.get_initial_state()
    goals = {"defend": ["LVP"]}
    
    # Generate plans
    plans = neural.generate_plan(state, goals)
    assert len(plans) > 0
    
    # Simulate plans
    for plan in plans:
        results = symbolic.simulate(state, plan)
        assert len(results) > 0
    
    # Test meta-policy
    context = {
        "iteration": 0,
        "time_remaining": 30.0,
        "best_score": 0.0,
        "last_improvement": 0.0,
        "variance": 0.0,
        "unpredictability": 0.3,
        "consecutive_failures": 0,
        "last_plan_legal": True
    }
    
    decision = meta.choose(context)
    assert decision["use"] in ["llm", "search", "repair", "stop"]
