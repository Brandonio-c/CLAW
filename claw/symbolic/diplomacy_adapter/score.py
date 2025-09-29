"""Scoring heuristics for Diplomacy plans."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from claw.types import Plan, Score
from claw.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScoringWeights:
    """Weights for different scoring components."""
    center_gain: float = 1.0
    positional_advantage: float = 0.5
    risk_penalty: float = -0.3
    goal_fit: float = 0.8
    confidence_bonus: float = 0.2


class DiplomacyScorer:
    """Scorer for Diplomacy plans based on heuristics."""
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        """Initialize scorer with weights.
        
        Args:
            weights: Scoring weights (uses defaults if None)
        """
        self.weights = weights or ScoringWeights()
    
    def score_plan(
        self,
        state: Dict[str, Any],
        plan: Plan,
        goals: Dict[str, Any]
    ) -> Score:
        """Score a plan based on multiple heuristics.
        
        Args:
            state: Current game state
            plan: Plan to score
            goals: Player goals
            
        Returns:
            Score object with breakdown
        """
        breakdown = {}
        
        # Center gain potential
        center_score = self._score_center_gain(state, plan)
        breakdown['center_gain'] = center_score
        
        # Positional advantage
        positional_score = self._score_positional_advantage(state, plan)
        breakdown['positional_advantage'] = positional_score
        
        # Risk assessment
        risk_score = self._score_risk(state, plan)
        breakdown['risk'] = risk_score
        
        # Goal alignment
        goal_score = self._score_goal_alignment(state, plan, goals)
        breakdown['goal_fit'] = goal_score
        
        # Confidence bonus
        confidence_score = plan.confidence * self.weights.confidence_bonus
        breakdown['confidence'] = confidence_score
        
        # Calculate total score
        total = (
            center_score * self.weights.center_gain +
            positional_score * self.weights.positional_advantage +
            risk_score * self.weights.risk_penalty +
            goal_score * self.weights.goal_fit +
            confidence_score
        )
        
        return Score(
            total=total,
            breakdown=breakdown,
            metadata={
                'weights': self.weights.__dict__,
                'plan_rationale': plan.rationale
            }
        )
    
    def _score_center_gain(self, state: Dict[str, Any], plan: Plan) -> float:
        """Score potential for gaining supply centers.
        
        Args:
            state: Current game state
            plan: Plan to score
            
        Returns:
            Center gain score
        """
        score = 0.0
        
        # Analyze move orders for center capture potential
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                continue
            
            action = order_data.get('action', '')
            target = order_data.get('target', '')
            
            if action == 'MTO' and target:
                # Check if target is a supply center
                if self._is_supply_center(target, state):
                    score += 1.0
                # Check if target is adjacent to a supply center
                elif self._is_adjacent_to_center(target, state):
                    score += 0.5
        
        return score
    
    def _score_positional_advantage(self, state: Dict[str, Any], plan: Plan) -> float:
        """Score positional advantages from the plan.
        
        Args:
            state: Current game state
            plan: Plan to score
            
        Returns:
            Positional advantage score
        """
        score = 0.0
        
        # Analyze support orders
        support_count = 0
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                continue
            
            if order_data.get('action') == 'SUP':
                support_count += 1
        
        # Support orders generally improve position
        score += support_count * 0.3
        
        # Analyze convoy orders
        convoy_count = 0
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                continue
            
            if order_data.get('action') == 'CVY':
                convoy_count += 1
        
        # Convoy orders can be strategically valuable
        score += convoy_count * 0.2
        
        return score
    
    def _score_risk(self, state: Dict[str, Any], plan: Plan) -> float:
        """Score risk level of the plan.
        
        Args:
            state: Current game state
            plan: Plan to score
            
        Returns:
            Risk score (higher is more risky)
        """
        risk = 0.0
        
        # Count aggressive moves
        aggressive_moves = 0
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                continue
            
            action = order_data.get('action', '')
            if action in ['MTO', 'SUP']:
                aggressive_moves += 1
        
        # More aggressive moves = higher risk
        risk += aggressive_moves * 0.2
        
        # Check for exposed units
        exposed_units = self._count_exposed_units(state, plan)
        risk += exposed_units * 0.3
        
        return risk
    
    def _score_goal_alignment(self, state: Dict[str, Any], plan: Plan, goals: Dict[str, Any]) -> float:
        """Score how well the plan aligns with goals.
        
        Args:
            state: Current game state
            plan: Plan to score
            goals: Player goals
            
        Returns:
            Goal alignment score
        """
        score = 0.0
        
        # Check defend goals
        defend_targets = goals.get('defend', [])
        if defend_targets:
            defended = self._count_defended_targets(state, plan, defend_targets)
            score += defended * 0.5
        
        # Check attack goals
        attack_targets = goals.get('attack', [])
        if attack_targets:
            attacked = self._count_attacked_targets(state, plan, attack_targets)
            score += attacked * 0.7
        
        # Check ally goals
        ally_targets = goals.get('ally', [])
        if ally_targets:
            allied = self._count_allied_actions(state, plan, ally_targets)
            score += allied * 0.3
        
        return score
    
    def _is_supply_center(self, location: str, state: Dict[str, Any]) -> bool:
        """Check if location is a supply center.
        
        Args:
            location: Location to check
            state: Game state
            
        Returns:
            True if location is a supply center
        """
        supply_centers = state.get('map', {}).get('supply_centers', [])
        return location in supply_centers
    
    def _is_adjacent_to_center(self, location: str, state: Dict[str, Any]) -> bool:
        """Check if location is adjacent to a supply center.
        
        Args:
            location: Location to check
            state: Game state
            
        Returns:
            True if adjacent to supply center
        """
        # Simplified implementation - would need proper adjacency checking
        supply_centers = state.get('map', {}).get('supply_centers', [])
        return any(self._are_adjacent(location, center) for center in supply_centers)
    
    def _are_adjacent(self, loc1: str, loc2: str) -> bool:
        """Check if two locations are adjacent (simplified).
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            True if adjacent
        """
        # This is a simplified implementation
        # In practice, would use proper Diplomacy map data
        return False
    
    def _count_exposed_units(self, state: Dict[str, Any], plan: Plan) -> int:
        """Count units that would be exposed by the plan.
        
        Args:
            state: Current game state
            plan: Plan to analyze
            
        Returns:
            Number of exposed units
        """
        # Simplified implementation
        # Would need to analyze unit positions and support
        return 0
    
    def _count_defended_targets(
        self, 
        state: Dict[str, Any], 
        plan: Plan, 
        targets: List[str]
    ) -> int:
        """Count how many defend targets are covered.
        
        Args:
            state: Current game state
            plan: Plan to analyze
            targets: Targets to defend
            
        Returns:
            Number of defended targets
        """
        defended = 0
        
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                continue
            
            if order_data.get('action') == 'SUP':
                support_target = order_data.get('target', '')
                if support_target in targets:
                    defended += 1
        
        return defended
    
    def _count_attacked_targets(
        self, 
        state: Dict[str, Any], 
        plan: Plan, 
        targets: List[str]
    ) -> int:
        """Count how many attack targets are targeted.
        
        Args:
            state: Current game state
            plan: Plan to analyze
            targets: Targets to attack
            
        Returns:
            Number of attacked targets
        """
        attacked = 0
        
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                continue
            
            if order_data.get('action') == 'MTO':
                target = order_data.get('target', '')
                if target in targets:
                    attacked += 1
        
        return attacked
    
    def _count_allied_actions(
        self, 
        state: Dict[str, Any], 
        plan: Plan, 
        allies: List[str]
    ) -> int:
        """Count actions that support allies.
        
        Args:
            state: Current game state
            plan: Plan to analyze
            allies: Ally powers
            
        Returns:
            Number of allied actions
        """
        # Simplified implementation
        # Would need to track which powers are allies
        return 0
