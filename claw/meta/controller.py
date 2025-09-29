"""Meta-cognitive controller implementation."""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from claw.interfaces.neural import NeuralStrategyGenerator
from claw.interfaces.symbolic import SymbolicPlannerInterface, SimulationResult
from claw.interfaces.domain import DomainAdapter
from claw.interfaces.meta import MetaPolicy
from claw.types import Plan, PlanStatus, Feedback
from claw.meta.policy import HeuristicMetaPolicy
from claw.meta.tracing import TraceLogger
from claw.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PlanningContext:
    """Context for planning decisions."""
    state: Dict[str, Any]
    goals: Dict[str, Any]
    iteration: int
    start_time: float
    time_remaining: float
    best_plan: Optional[Plan]
    best_score: float
    last_improvement: float
    consecutive_failures: int
    unpredictability: float


class MetaCognitiveController:
    """Meta-cognitive controller orchestrating the planning loop."""
    
    def __init__(
        self,
        neural_generator: NeuralStrategyGenerator,
        symbolic_planner: SymbolicPlannerInterface,
        domain_adapter: DomainAdapter,
        meta_policy: Optional[MetaPolicy] = None,
        trace_file: Optional[str] = None
    ):
        """Initialize the meta-cognitive controller.
        
        Args:
            neural_generator: Neural strategy generator
            symbolic_planner: Symbolic planner interface
            domain_adapter: Domain adapter
            meta_policy: Meta-policy (uses heuristic if None)
            trace_file: Optional trace file path
        """
        self.neural_generator = neural_generator
        self.symbolic_planner = symbolic_planner
        self.domain_adapter = domain_adapter
        self.meta_policy = meta_policy or HeuristicMetaPolicy()
        self.trace_logger = TraceLogger(trace_file)
        
        logger.info("Initialized meta-cognitive controller")
    
    def generate_plan(
        self,
        state: Dict[str, Any],
        goals: Dict[str, Any],
        *,
        unpredictability: float = 0.3,
        max_iters: int = 5,
        time_budget: float = 30.0
    ) -> Plan:
        """Generate a plan using the meta-cognitive loop.
        
        Args:
            state: Current game state
            goals: Player goals
            unpredictability: Unpredictability setting (0.0-1.0)
            max_iters: Maximum iterations
            time_budget: Time budget in seconds
            
        Returns:
            Best plan found
        """
        logger.info(f"Starting plan generation with unpredictability={unpredictability}")
        
        # Initialize context
        context = PlanningContext(
            state=state,
            goals=goals,
            iteration=0,
            start_time=time.time(),
            time_remaining=time_budget,
            best_plan=None,
            best_score=0.0,
            last_improvement=0.0,
            consecutive_failures=0,
            unpredictability=unpredictability
        )
        
        # Log start
        self.trace_logger.log_event(
            event_type='planning_start',
            data={
                'state': state,
                'goals': goals,
                'unpredictability': unpredictability,
                'max_iters': max_iters,
                'time_budget': time_budget
            }
        )
        
        # Main planning loop
        while self._should_continue(context, max_iters):
            try:
                # Update time remaining
                context.time_remaining = time_budget - (time.time() - context.start_time)
                
                # Get meta-policy decision
                decision = self._get_meta_decision(context)
                
                # Execute decision
                result = self._execute_decision(context, decision)
                
                # Update context
                context = self._update_context(context, result)
                
                # Log iteration
                self.trace_logger.log_iteration(
                    iteration=context.iteration,
                    best_plan=context.best_plan,
                    best_score=context.best_score,
                    time_remaining=context.time_remaining,
                    improvement=context.last_improvement
                )
                
                context.iteration += 1
                
            except Exception as e:
                logger.error(f"Error in planning iteration {context.iteration}: {e}")
                self.trace_logger.log_error(str(e), {'iteration': context.iteration})
                context.consecutive_failures += 1
                context.iteration += 1
        
        # Return best plan or fallback
        final_plan = context.best_plan or self._get_fallback_plan(state)
        
        # Log completion
        self.trace_logger.log_event(
            event_type='planning_complete',
            data={
                'final_plan': {
                    'orders': final_plan.orders,
                    'rationale': final_plan.rationale,
                    'confidence': final_plan.confidence,
                    'status': final_plan.status.value
                },
                'iterations': context.iteration,
                'best_score': context.best_score,
                'total_time': time.time() - context.start_time
            }
        )
        
        logger.info(f"Plan generation complete: {final_plan.rationale[:50]}...")
        return final_plan
    
    def _should_continue(self, context: PlanningContext, max_iters: int) -> bool:
        """Check if planning should continue.
        
        Args:
            context: Current planning context
            max_iters: Maximum iterations
            
        Returns:
            True if should continue, False otherwise
        """
        # Check iteration limit
        if context.iteration >= max_iters:
            return False
        
        # Check time budget
        if context.time_remaining <= 0:
            return False
        
        # Check consecutive failures
        if context.consecutive_failures >= 3:
            return False
        
        # Use meta-policy to decide
        policy_context = {
            'iteration': context.iteration,
            'time_remaining': context.time_remaining,
            'best_score': context.best_score,
            'last_improvement': context.last_improvement,
            'variance': 0.0,  # Would need to calculate from recent results
            'unpredictability': context.unpredictability,
            'consecutive_failures': context.consecutive_failures,
            'last_plan_legal': context.best_plan is not None and context.best_plan.status == PlanStatus.LEGAL
        }
        
        return self.meta_policy.should_continue(policy_context)
    
    def _get_meta_decision(self, context: PlanningContext) -> Dict[str, Any]:
        """Get decision from meta-policy.
        
        Args:
            context: Current planning context
            
        Returns:
            Meta-policy decision
        """
        policy_context = {
            'state': context.state,
            'goals': context.goals,
            'iteration': context.iteration,
            'time_remaining': context.time_remaining,
            'best_score': context.best_score,
            'last_improvement': context.last_improvement,
            'variance': 0.0,  # Would need to calculate from recent results
            'unpredictability': context.unpredictability,
            'consecutive_failures': context.consecutive_failures,
            'last_plan_legal': context.best_plan is not None and context.best_plan.status == PlanStatus.LEGAL
        }
        
        decision = self.meta_policy.choose(policy_context)
        
        # Log decision
        self.trace_logger.log_meta_decision(
            choice=decision.get('use', 'llm'),
            parameters=decision,
            context=policy_context
        )
        
        return decision
    
    def _execute_decision(
        self,
        context: PlanningContext,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a meta-policy decision.
        
        Args:
            context: Current planning context
            decision: Decision to execute
            
        Returns:
            Execution result
        """
        use = decision.get('use', 'llm')
        
        if use == 'llm':
            return self._execute_llm_decision(context, decision)
        elif use == 'search':
            return self._execute_search_decision(context, decision)
        elif use == 'repair':
            return self._execute_repair_decision(context, decision)
        else:
            logger.warning(f"Unknown decision type: {use}")
            return {'success': False, 'plans': [], 'reason': f'Unknown decision: {use}'}
    
    def _execute_llm_decision(
        self,
        context: PlanningContext,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute LLM-based decision.
        
        Args:
            context: Current planning context
            decision: Decision parameters
            
        Returns:
            Execution result
        """
        logger.debug("Executing LLM decision")
        
        try:
            # Generate plans using LLM
            plans = self.neural_generator.generate_plan(
                context.state,
                context.goals,
                temperature=decision.get('temperature', 0.7),
                max_samples=decision.get('max_samples', 3)
            )
            
            # Log plan generation
            self.trace_logger.log_plan_generation(
                plans=plans,
                method='llm',
                parameters=decision
            )
            
            # Evaluate plans
            evaluated_plans = self._evaluate_plans(context, plans, decision)
            
            return {
                'success': True,
                'plans': evaluated_plans,
                'method': 'llm'
            }
            
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return {'success': False, 'plans': [], 'reason': str(e)}
    
    def _execute_search_decision(
        self,
        context: PlanningContext,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute symbolic search decision.
        
        Args:
            context: Current planning context
            decision: Decision parameters
            
        Returns:
            Execution result
        """
        logger.debug("Executing search decision")
        
        try:
            # Search for plans using symbolic methods
            plans = self.symbolic_planner.search_plans(
                context.state,
                context.goals,
                max_depth=decision.get('search_depth', 2),
                max_plans=decision.get('max_samples', 10)
            )
            
            # Log plan generation
            self.trace_logger.log_plan_generation(
                plans=plans,
                method='search',
                parameters=decision
            )
            
            # Evaluate plans
            evaluated_plans = self._evaluate_plans(context, plans, decision)
            
            return {
                'success': True,
                'plans': evaluated_plans,
                'method': 'search'
            }
            
        except Exception as e:
            logger.error(f"Search decision failed: {e}")
            return {'success': False, 'plans': [], 'reason': str(e)}
    
    def _execute_repair_decision(
        self,
        context: PlanningContext,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute plan repair decision.
        
        Args:
            context: Current planning context
            decision: Decision parameters
            
        Returns:
            Execution result
        """
        logger.debug("Executing repair decision")
        
        if not context.best_plan:
            return {'success': False, 'plans': [], 'reason': 'No plan to repair'}
        
        try:
            # Attempt to repair the best plan
            repaired_plan = self.symbolic_planner.repair(
                context.state,
                context.best_plan
            )
            
            if repaired_plan:
                # Log repair
                self.trace_logger.log_repair(
                    original_plan=context.best_plan,
                    repaired_plan=repaired_plan,
                    success=True,
                    feedback={}
                )
                
                # Evaluate repaired plan
                evaluated_plans = self._evaluate_plans(context, [repaired_plan], decision)
                
                return {
                    'success': True,
                    'plans': evaluated_plans,
                    'method': 'repair'
                }
            else:
                return {'success': False, 'plans': [], 'reason': 'Repair failed'}
                
        except Exception as e:
            logger.error(f"Repair decision failed: {e}")
            return {'success': False, 'plans': [], 'reason': str(e)}
    
    def _evaluate_plans(
        self,
        context: PlanningContext,
        plans: List[Plan],
        decision: Dict[str, Any]
    ) -> List[Plan]:
        """Evaluate plans through simulation.
        
        Args:
            context: Current planning context
            plans: Plans to evaluate
            decision: Decision parameters
            
        Returns:
            Evaluated plans with updated status and scores
        """
        evaluated_plans = []
        
        for plan in plans:
            try:
                # Simulate plan
                results = self.symbolic_planner.simulate(
                    context.state,
                    plan,
                    opponent_model=decision.get('opponent_model', 'random'),
                    samples=decision.get('simulation_samples', 5)
                )
                
                # Log simulation
                self.trace_logger.log_simulation(
                    plan=plan,
                    results=[{
                        'is_legal': r.is_legal,
                        'score': r.score,
                        'diagnostics': r.diagnostics
                    } for r in results],
                    opponent_model=decision.get('opponent_model', 'random'),
                    samples=len(results)
                )
                
                # Calculate aggregate score
                legal_results = [r for r in results if r.is_legal]
                if legal_results:
                    avg_score = sum(r.score for r in legal_results) / len(legal_results)
                    plan.status = PlanStatus.LEGAL
                else:
                    avg_score = 0.0
                    plan.status = PlanStatus.ILLEGAL
                
                # Update plan confidence based on results
                if legal_results:
                    plan.confidence = min(1.0, plan.confidence + 0.1)
                else:
                    plan.confidence = max(0.0, plan.confidence - 0.2)
                
                # Store evaluation results
                plan.metadata = plan.metadata or {}
                plan.metadata['evaluation'] = {
                    'avg_score': avg_score,
                    'legal_count': len(legal_results),
                    'total_samples': len(results)
                }
                
                evaluated_plans.append(plan)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate plan: {e}")
                plan.status = PlanStatus.ILLEGAL
                plan.confidence = 0.0
                evaluated_plans.append(plan)
        
        return evaluated_plans
    
    def _update_context(
        self,
        context: PlanningContext,
        result: Dict[str, Any]
    ) -> PlanningContext:
        """Update planning context with result.
        
        Args:
            context: Current context
            result: Execution result
            
        Returns:
            Updated context
        """
        if not result['success']:
            context.consecutive_failures += 1
            return context
        
        plans = result.get('plans', [])
        if not plans:
            context.consecutive_failures += 1
            return context
        
        # Find best plan
        best_plan = max(plans, key=lambda p: p.metadata.get('evaluation', {}).get('avg_score', 0.0) if p.metadata else 0.0)
        best_score = best_plan.metadata.get('evaluation', {}).get('avg_score', 0.0) if best_plan.metadata else 0.0
        
        # Update context
        if best_score > context.best_score:
            context.last_improvement = best_score - context.best_score
            context.best_score = best_score
            context.best_plan = best_plan
            context.consecutive_failures = 0
        else:
            context.last_improvement = 0.0
            context.consecutive_failures += 1
        
        return context
    
    def _get_fallback_plan(self, state: Dict[str, Any]) -> Plan:
        """Get fallback plan when all else fails.
        
        Args:
            state: Current game state
            
        Returns:
            Fallback plan
        """
        logger.warning("Using fallback plan")
        
        # Create a simple hold plan
        fallback_plan = Plan(
            orders={'fallback': {'unit': 'A LVP', 'action': 'HLD'}},
            rationale="Fallback plan: hold position",
            confidence=0.1,
            status=PlanStatus.PENDING
        )
        
        return fallback_plan
