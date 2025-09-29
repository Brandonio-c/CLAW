"""Tests for meta-cognitive controller."""

import pytest
from unittest.mock import Mock, patch
from claw.meta.controller import MetaCognitiveController, PlanningContext
from claw.meta.policy import HeuristicMetaPolicy
from claw.meta.tracing import TraceLogger
from claw.types import Plan, PlanStatus
from claw.interfaces.symbolic import SimulationResult


class MockNeuralGenerator:
    """Mock neural strategy generator."""
    
    def generate_plan(self, state, goals, *, temperature=0.7, max_samples=3):
        plans = []
        for i in range(max_samples):
            plan = Plan(
                orders={f"order_{i}": {"unit": f"A LVP", "action": "HLD"}},
                rationale=f"Mock plan {i}",
                confidence=0.5 + i * 0.1
            )
            plans.append(plan)
        return plans
    
    def repair_plan(self, state, plan, feedback, *, temperature=0.5):
        repaired = Plan(
            orders=plan.orders.copy(),
            rationale=f"Repaired: {plan.rationale}",
            confidence=plan.confidence * 0.8,
            status=PlanStatus.REPAIRED
        )
        return repaired
    
    def validate_plan(self, state, plan):
        return plan.confidence > 0.3


class MockSymbolicPlanner:
    """Mock symbolic planner."""
    
    def enumerate_legal_orders(self, state):
        return {"A LVP": [{"unit": "A LVP", "action": "HLD"}]}
    
    def simulate(self, state, plan, opponent_model="random", samples=1):
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
    
    def repair(self, state, plan):
        if plan.confidence < 0.5:
            repaired = Plan(
                orders=plan.orders.copy(),
                rationale=f"Repaired: {plan.rationale}",
                confidence=0.6,
                status=PlanStatus.REPAIRED
            )
            return repaired
        return None
    
    def score_plan(self, state, plan, goals):
        return plan.confidence * 10.0
    
    def search_plans(self, state, goals, max_depth=2, max_plans=10):
        plans = []
        for i in range(min(max_plans, 3)):
            plan = Plan(
                orders={f"search_order_{i}": {"unit": f"A LVP", "action": "HLD"}},
                rationale=f"Search plan {i}",
                confidence=0.4 + i * 0.1
            )
            plans.append(plan)
        return plans


class MockDomainAdapter:
    """Mock domain adapter."""
    
    def normalize_state(self, raw_state):
        return {"units": {"ENGLAND": ["A LVP"]}, "phase": "SPRING 1901"}
    
    def serialize_for_prompt(self, state):
        return f"Mock state: {state}"
    
    def parse_orders_from_text(self, text):
        return {"orders": [{"unit": "A LVP", "action": "HLD"}]}
    
    def plan_to_engine_orders(self, plan):
        return {"ENGLAND": ["A LVP HLD"]}
    
    def engine_orders_to_plan(self, orders, rationale="", confidence=1.0):
        return Plan(
            orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
            rationale=rationale,
            confidence=confidence
        )
    
    def goal_schema(self):
        return {
            "defend": {"type": "list", "items": {"type": "string"}},
            "attack": {"type": "list", "items": {"type": "string"}}
        }
    
    def validate_goals(self, goals):
        return isinstance(goals, dict)
    
    def get_initial_state(self):
        return {"units": {"ENGLAND": ["A LVP"]}, "phase": "SPRING 1901"}


def test_planning_context():
    """Test PlanningContext dataclass."""
    context = PlanningContext(
        state={"units": {"ENGLAND": ["A LVP"]}},
        goals={"defend": ["LVP"]},
        iteration=1,
        start_time=1000.0,
        time_remaining=30.0,
        best_plan=None,
        best_score=0.0,
        last_improvement=0.0,
        consecutive_failures=0,
        unpredictability=0.3
    )
    
    assert context.iteration == 1
    assert context.time_remaining == 30.0
    assert context.best_score == 0.0
    assert context.unpredictability == 0.3


def test_meta_cognitive_controller_init():
    """Test controller initialization."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    assert controller.neural_generator == neural
    assert controller.symbolic_planner == symbolic
    assert controller.domain_adapter == domain
    assert controller.meta_policy == policy


def test_meta_cognitive_controller_generate_plan():
    """Test plan generation."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    state = {"units": {"ENGLAND": ["A LVP"]}}
    goals = {"defend": ["LVP"]}
    
    plan = controller.generate_plan(
        state=state,
        goals=goals,
        unpredictability=0.3,
        max_iters=3,
        time_budget=10.0
    )
    
    assert isinstance(plan, Plan)
    assert plan.rationale is not None
    assert 0.0 <= plan.confidence <= 1.0


def test_meta_cognitive_controller_should_continue():
    """Test should_continue logic."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    # Test with valid context
    context = PlanningContext(
        state={"units": {"ENGLAND": ["A LVP"]}},
        goals={"defend": ["LVP"]},
        iteration=1,
        start_time=1000.0,
        time_remaining=30.0,
        best_plan=None,
        best_score=0.0,
        last_improvement=0.0,
        consecutive_failures=0,
        unpredictability=0.3
    )
    
    should_continue = controller._should_continue(context, max_iters=5)
    assert should_continue == True
    
    # Test with max iterations reached
    context.iteration = 5
    should_continue = controller._should_continue(context, max_iters=5)
    assert should_continue == False
    
    # Test with no time remaining
    context.iteration = 1
    context.time_remaining = 0.0
    should_continue = controller._should_continue(context, max_iters=5)
    assert should_continue == False


def test_meta_cognitive_controller_get_meta_decision():
    """Test meta-policy decision making."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    context = PlanningContext(
        state={"units": {"ENGLAND": ["A LVP"]}},
        goals={"defend": ["LVP"]},
        iteration=1,
        start_time=1000.0,
        time_remaining=30.0,
        best_plan=None,
        best_score=0.0,
        last_improvement=0.0,
        consecutive_failures=0,
        unpredictability=0.3
    )
    
    decision = controller._get_meta_decision(context)
    assert isinstance(decision, dict)
    assert 'use' in decision
    assert 'temperature' in decision
    assert 'max_samples' in decision


def test_meta_cognitive_controller_execute_llm_decision():
    """Test LLM decision execution."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    context = PlanningContext(
        state={"units": {"ENGLAND": ["A LVP"]}},
        goals={"defend": ["LVP"]},
        iteration=1,
        start_time=1000.0,
        time_remaining=30.0,
        best_plan=None,
        best_score=0.0,
        last_improvement=0.0,
        consecutive_failures=0,
        unpredictability=0.3
    )
    
    decision = {
        'use': 'llm',
        'temperature': 0.7,
        'max_samples': 3,
        'opponent_model': 'random'
    }
    
    result = controller._execute_llm_decision(context, decision)
    assert result['success'] == True
    assert 'plans' in result
    assert result['method'] == 'llm'


def test_meta_cognitive_controller_execute_search_decision():
    """Test search decision execution."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    context = PlanningContext(
        state={"units": {"ENGLAND": ["A LVP"]}},
        goals={"defend": ["LVP"]},
        iteration=1,
        start_time=1000.0,
        time_remaining=30.0,
        best_plan=None,
        best_score=0.0,
        last_improvement=0.0,
        consecutive_failures=0,
        unpredictability=0.3
    )
    
    decision = {
        'use': 'search',
        'search_depth': 2,
        'max_samples': 5,
        'opponent_model': 'random'
    }
    
    result = controller._execute_search_decision(context, decision)
    assert result['success'] == True
    assert 'plans' in result
    assert result['method'] == 'search'


def test_meta_cognitive_controller_execute_repair_decision():
    """Test repair decision execution."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    # Test with repairable plan
    context = PlanningContext(
        state={"units": {"ENGLAND": ["A LVP"]}},
        goals={"defend": ["LVP"]},
        iteration=1,
        start_time=1000.0,
        time_remaining=30.0,
        best_plan=Plan(
            orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
            rationale="Low confidence plan",
            confidence=0.3
        ),
        best_score=0.0,
        last_improvement=0.0,
        consecutive_failures=0,
        unpredictability=0.3
    )
    
    decision = {'use': 'repair', 'opponent_model': 'random'}
    
    result = controller._execute_repair_decision(context, decision)
    assert result['success'] == True
    assert 'plans' in result
    assert result['method'] == 'repair'
    
    # Test with no plan to repair
    context.best_plan = None
    result = controller._execute_repair_decision(context, decision)
    assert result['success'] == False
    assert 'No plan to repair' in result['reason']


def test_meta_cognitive_controller_evaluate_plans():
    """Test plan evaluation."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    context = PlanningContext(
        state={"units": {"ENGLAND": ["A LVP"]}},
        goals={"defend": ["LVP"]},
        iteration=1,
        start_time=1000.0,
        time_remaining=30.0,
        best_plan=None,
        best_score=0.0,
        last_improvement=0.0,
        consecutive_failures=0,
        unpredictability=0.3
    )
    
    plans = [
        Plan(
            orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
            rationale="Test plan 1",
            confidence=0.8
        ),
        Plan(
            orders={"order_1": {"unit": "A LVP", "action": "MTO", "target": "EDI"}},
            rationale="Test plan 2",
            confidence=0.6
        )
    ]
    
    decision = {'opponent_model': 'random', 'simulation_samples': 2}
    
    evaluated_plans = controller._evaluate_plans(context, plans, decision)
    
    assert len(evaluated_plans) == 2
    for plan in evaluated_plans:
        assert plan.metadata is not None
        assert 'evaluation' in plan.metadata
        assert 'avg_score' in plan.metadata['evaluation']


def test_meta_cognitive_controller_update_context():
    """Test context updating."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    context = PlanningContext(
        state={"units": {"ENGLAND": ["A LVP"]}},
        goals={"defend": ["LVP"]},
        iteration=1,
        start_time=1000.0,
        time_remaining=30.0,
        best_plan=None,
        best_score=0.0,
        last_improvement=0.0,
        consecutive_failures=0,
        unpredictability=0.3
    )
    
    # Test successful result
    result = {
        'success': True,
        'plans': [
            Plan(
                orders={"order_0": {"unit": "A LVP", "action": "HLD"}},
                rationale="Better plan",
                confidence=0.8,
                metadata={'evaluation': {'avg_score': 0.8}}
            )
        ]
    }
    
    updated_context = controller._update_context(context, result)
    assert updated_context.best_score == 0.8
    assert updated_context.best_plan is not None
    assert updated_context.consecutive_failures == 0
    assert updated_context.last_improvement == 0.8
    
    # Test failed result
    failed_result = {'success': False, 'plans': []}
    updated_context = controller._update_context(context, failed_result)
    assert updated_context.consecutive_failures == 1


def test_meta_cognitive_controller_fallback_plan():
    """Test fallback plan generation."""
    neural = MockNeuralGenerator()
    symbolic = MockSymbolicPlanner()
    domain = MockDomainAdapter()
    policy = HeuristicMetaPolicy()
    
    controller = MetaCognitiveController(
        neural_generator=neural,
        symbolic_planner=symbolic,
        domain_adapter=domain,
        meta_policy=policy
    )
    
    state = {"units": {"ENGLAND": ["A LVP"]}}
    fallback_plan = controller._get_fallback_plan(state)
    
    assert isinstance(fallback_plan, Plan)
    assert fallback_plan.rationale == "Fallback plan: hold position"
    assert fallback_plan.confidence == 0.1
