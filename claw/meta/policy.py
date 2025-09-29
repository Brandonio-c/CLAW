"""Heuristic meta-policy implementation."""

import time
from typing import Dict, Any, List
from dataclasses import dataclass
from claw.interfaces.meta import MetaPolicy
from claw.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyState:
    """State for meta-policy decisions."""
    iteration: int
    time_remaining: float
    best_score: float
    last_improvement: float
    variance: float
    unpredictability: float
    consecutive_failures: int
    last_plan_legal: bool


class HeuristicMetaPolicy(MetaPolicy):
    """Heuristic meta-policy using rational metareasoning rules."""
    
    def __init__(
        self,
        max_iters: int = 5,
        time_budget: float = 30.0,
        confidence_threshold: float = 0.6,
        improvement_threshold: float = 0.05
    ):
        """Initialize heuristic meta-policy.
        
        Args:
            max_iters: Maximum iterations per turn
            time_budget: Time budget in seconds
            confidence_threshold: Minimum confidence to accept plan
            improvement_threshold: Minimum improvement to continue
        """
        self.max_iters = max_iters
        self.time_budget = time_budget
        self.confidence_threshold = confidence_threshold
        self.improvement_threshold = improvement_threshold
        
        logger.info("Initialized heuristic meta-policy")
    
    def choose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Choose control signals based on context.
        
        Args:
            context: Current planning context
            
        Returns:
            Control signals dictionary
        """
        # Extract context
        state = PolicyState(
            iteration=context.get('iteration', 0),
            time_remaining=context.get('time_remaining', self.time_budget),
            best_score=context.get('best_score', 0.0),
            last_improvement=context.get('last_improvement', 0.0),
            variance=context.get('variance', 0.0),
            unpredictability=context.get('unpredictability', 0.3),
            consecutive_failures=context.get('consecutive_failures', 0),
            last_plan_legal=context.get('last_plan_legal', True)
        )
        
        # Make decision
        decision = self._make_decision(state)
        
        logger.debug(f"Meta-policy decision: {decision}")
        return decision
    
    def should_continue(self, context: Dict[str, Any]) -> bool:
        """Determine if planning should continue.
        
        Args:
            context: Current planning context
            
        Returns:
            True if planning should continue, False otherwise
        """
        state = PolicyState(
            iteration=context.get('iteration', 0),
            time_remaining=context.get('time_remaining', self.time_budget),
            best_score=context.get('best_score', 0.0),
            last_improvement=context.get('last_improvement', 0.0),
            variance=context.get('variance', 0.0),
            unpredictability=context.get('unpredictability', 0.3),
            consecutive_failures=context.get('consecutive_failures', 0),
            last_plan_legal=context.get('last_plan_legal', True)
        )
        
        # Check stopping conditions
        if state.iteration >= self.max_iters:
            logger.info("Stopping: maximum iterations reached")
            return False
        
        if state.time_remaining <= 0:
            logger.info("Stopping: time budget exhausted")
            return False
        
        if state.consecutive_failures >= 3:
            logger.info("Stopping: too many consecutive failures")
            return False
        
        # Check if we have a good enough plan
        if (state.best_score >= self.confidence_threshold and 
            state.last_plan_legal and 
            state.time_remaining < self.time_budget * 0.2):
            logger.info("Stopping: good plan found with limited time remaining")
            return False
        
        return True
    
    def adjust_parameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust planning parameters based on context.
        
        Args:
            context: Current planning context
            
        Returns:
            Adjusted parameters dictionary
        """
        state = PolicyState(
            iteration=context.get('iteration', 0),
            time_remaining=context.get('time_remaining', self.time_budget),
            best_score=context.get('best_score', 0.0),
            last_improvement=context.get('last_improvement', 0.0),
            variance=context.get('variance', 0.0),
            unpredictability=context.get('unpredictability', 0.3),
            consecutive_failures=context.get('consecutive_failures', 0),
            last_plan_legal=context.get('last_plan_legal', True)
        )
        
        # Adjust parameters based on state
        temperature = self._adjust_temperature(state)
        max_samples = self._adjust_max_samples(state)
        opponent_model = self._adjust_opponent_model(state)
        search_depth = self._adjust_search_depth(state)
        
        return {
            'temperature': temperature,
            'max_samples': max_samples,
            'opponent_model': opponent_model,
            'search_depth': search_depth
        }
    
    def _make_decision(self, state: PolicyState) -> Dict[str, Any]:
        """Make a decision based on policy state.
        
        Args:
            state: Current policy state
            
        Returns:
            Decision dictionary
        """
        # Determine primary action
        if state.consecutive_failures >= 2:
            use = 'repair'
        elif state.variance > 0.5 and state.iteration > 1:
            use = 'search'
        elif state.last_improvement < self.improvement_threshold and state.iteration > 2:
            use = 'llm'  # Try LLM with different parameters
        else:
            use = 'llm'
        
        # Adjust parameters
        params = self.adjust_parameters({
            'iteration': state.iteration,
            'time_remaining': state.time_remaining,
            'best_score': state.best_score,
            'last_improvement': state.last_improvement,
            'variance': state.variance,
            'unpredictability': state.unpredictability,
            'consecutive_failures': state.consecutive_failures,
            'last_plan_legal': state.last_plan_legal
        })
        
        # Determine if we should stop
        stop = not self.should_continue({
            'iteration': state.iteration,
            'time_remaining': state.time_remaining,
            'best_score': state.best_score,
            'last_improvement': state.last_improvement,
            'variance': state.variance,
            'unpredictability': state.unpredictability,
            'consecutive_failures': state.consecutive_failures,
            'last_plan_legal': state.last_plan_legal
        })
        
        return {
            'use': use,
            'stop': stop,
            **params
        }
    
    def _adjust_temperature(self, state: PolicyState) -> float:
        """Adjust temperature based on state.
        
        Args:
            state: Current policy state
            
        Returns:
            Adjusted temperature
        """
        base_temp = 0.7
        
        # Increase temperature if no improvement
        if state.last_improvement < self.improvement_threshold:
            base_temp += 0.2
        
        # Increase temperature for exploration
        if state.unpredictability > 0.5:
            base_temp += 0.3
        
        # Decrease temperature if we have a good plan
        if state.best_score > self.confidence_threshold:
            base_temp -= 0.2
        
        # Increase temperature if many failures
        if state.consecutive_failures > 1:
            base_temp += 0.1
        
        return max(0.1, min(2.0, base_temp))
    
    def _adjust_max_samples(self, state: PolicyState) -> int:
        """Adjust max samples based on state.
        
        Args:
            state: Current policy state
            
        Returns:
            Adjusted max samples
        """
        base_samples = 3
        
        # Increase samples if no improvement
        if state.last_improvement < self.improvement_threshold:
            base_samples += 2
        
        # Increase samples for exploration
        if state.unpredictability > 0.5:
            base_samples += 1
        
        # Decrease samples if time is running out
        if state.time_remaining < self.time_budget * 0.3:
            base_samples = max(1, base_samples - 1)
        
        return max(1, min(10, base_samples))
    
    def _adjust_opponent_model(self, state: PolicyState) -> str:
        """Adjust opponent model based on state.
        
        Args:
            state: Current policy state
            
        Returns:
            Opponent model type
        """
        # Use random for exploration, best for exploitation
        if state.unpredictability > 0.5 or state.iteration < 2:
            return 'random'
        else:
            return 'best'
    
    def _adjust_search_depth(self, state: PolicyState) -> int:
        """Adjust search depth based on state.
        
        Args:
            state: Current policy state
            
        Returns:
            Search depth
        """
        base_depth = 2
        
        # Increase depth if we have time and need better plans
        if (state.time_remaining > self.time_budget * 0.5 and 
            state.best_score < self.confidence_threshold):
            base_depth += 1
        
        # Decrease depth if time is running out
        if state.time_remaining < self.time_budget * 0.3:
            base_depth = max(1, base_depth - 1)
        
        return max(1, min(5, base_depth))
