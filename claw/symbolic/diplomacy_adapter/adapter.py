"""Diplomacy domain adapter implementation."""

from typing import Dict, Any, List, Optional
from claw.interfaces.domain import DomainAdapter
from claw.interfaces.symbolic import SymbolicPlannerInterface, SimulationResult
from claw.types import Plan, PlanStatus
from claw.symbolic.diplomacy_adapter.engine import DiplomacyEngine
from claw.symbolic.diplomacy_adapter.orders import OrderParser, OrderValidator
from claw.symbolic.diplomacy_adapter.simulate import DiplomacySimulator
from claw.symbolic.diplomacy_adapter.score import DiplomacyScorer
from claw.logging import get_logger

logger = get_logger(__name__)


class DiplomacyAdapter(DomainAdapter, SymbolicPlannerInterface):
    """Diplomacy domain adapter implementing both interfaces."""
    
    def __init__(self, game_id: Optional[str] = None):
        """Initialize the Diplomacy adapter.
        
        Args:
            game_id: Optional game ID for loading existing game
        """
        self.engine = DiplomacyEngine(game_id)
        self.parser = OrderParser()
        self.validator = OrderValidator(self.engine)
        self.simulator = DiplomacySimulator(self.engine)
        self.scorer = DiplomacyScorer()
        
        logger.info("Initialized Diplomacy adapter")
    
    # DomainAdapter methods
    
    def normalize_state(self, raw_state: Any) -> Dict[str, Any]:
        """Normalize raw state to framework format.
        
        Args:
            raw_state: Raw state from domain engine
            
        Returns:
            Normalized state dictionary
        """
        if isinstance(raw_state, dict):
            return raw_state
        
        # If raw_state is a Game object, extract state
        if hasattr(raw_state, 'get_units'):
            return self.engine.get_state()
        
        # Default to current engine state
        return self.engine.get_state()
    
    def serialize_for_prompt(self, state: Dict[str, Any]) -> str:
        """Serialize state for LLM prompting.
        
        Args:
            state: Normalized state
            
        Returns:
            String representation for prompts
        """
        lines = []
        
        # Add phase information
        phase = state.get('phase', 'Unknown')
        lines.append(f"Phase: {phase}")
        
        # Add units
        units = state.get('units', {})
        if units:
            lines.append("\nUnits:")
            for power, power_units in units.items():
                if power_units:
                    lines.append(f"  {power}:")
                    for unit in power_units:
                        lines.append(f"    {unit}")
        
        # Add supply centers
        centers = state.get('centers', {})
        if centers:
            lines.append("\nSupply Centers:")
            for power, power_centers in centers.items():
                if power_centers:
                    lines.append(f"  {power}: {', '.join(power_centers)}")
        
        # Add current power
        current_power = state.get('current_power', 'Unknown')
        lines.append(f"\nYour Power: {current_power}")
        
        return "\n".join(lines)
    
    def parse_orders_from_text(self, text: str) -> Dict[str, Any]:
        """Parse orders from text (e.g., LLM output).
        
        Args:
            text: Text containing orders
            
        Returns:
            Parsed orders dictionary
        """
        # Split text into lines and parse each as an order
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        orders = []
        for line in lines:
            parsed = self.parser.parse_order(line)
            if parsed:
                orders.append(parsed.to_dict())
        
        return {
            'orders': orders,
            'raw_text': text
        }
    
    def plan_to_engine_orders(self, plan: Plan) -> Dict[str, Any]:
        """Convert plan to engine-specific orders.
        
        Args:
            plan: Framework plan object
            
        Returns:
            Engine-specific orders
        """
        orders = {}
        
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                continue
            
            # Extract power from unit (simplified)
            unit = order_data.get('unit', '')
            if not unit:
                continue
            
            power = self._detect_power_from_unit(unit)
            if not power:
                continue
            
            if power not in orders:
                orders[power] = []
            
            # Format order
            order_str = self._format_order_string(order_data)
            if order_str:
                orders[power].append(order_str)
        
        return orders
    
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
        plan_orders = {}
        
        for power, power_orders in orders.items():
            for i, order_str in enumerate(power_orders):
                parsed = self.parser.parse_order(order_str)
                if parsed:
                    order_key = f"{power}_order_{i}"
                    plan_orders[order_key] = parsed.to_dict()
        
        return Plan(
            orders=plan_orders,
            rationale=rationale,
            confidence=confidence,
            status=PlanStatus.PENDING
        )
    
    def goal_schema(self) -> Dict[str, Any]:
        """Get goal schema for this domain.
        
        Returns:
            Dictionary describing valid goal keys and types
        """
        return {
            'defend': {
                'type': 'list',
                'description': 'Locations to defend',
                'items': {'type': 'string'}
            },
            'attack': {
                'type': 'list', 
                'description': 'Locations to attack',
                'items': {'type': 'string'}
            },
            'ally': {
                'type': 'list',
                'description': 'Powers to ally with',
                'items': {'type': 'string'}
            },
            'avoid': {
                'type': 'list',
                'description': 'Locations to avoid',
                'items': {'type': 'string'}
            }
        }
    
    def validate_goals(self, goals: Dict[str, Any]) -> bool:
        """Validate goals against domain schema.
        
        Args:
            goals: Goals to validate
            
        Returns:
            True if goals are valid, False otherwise
        """
        schema = self.goal_schema()
        
        for key, value in goals.items():
            if key not in schema:
                logger.warning(f"Unknown goal key: {key}")
                return False
            
            expected_type = schema[key]['type']
            if expected_type == 'list' and not isinstance(value, list):
                logger.warning(f"Goal {key} should be a list")
                return False
        
        return True
    
    def get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for this domain.
        
        Returns:
            Initial state dictionary
        """
        return self.engine.get_state()
    
    # SymbolicPlannerInterface methods
    
    def enumerate_legal_orders(self, state: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Enumerate all legal orders for each unit.
        
        Args:
            state: Current game state
            
        Returns:
            Dictionary mapping unit names to lists of legal orders
        """
        current_power = state.get('current_power', 'ENGLAND')
        
        try:
            legal_orders = self.engine.get_legal_orders(current_power)
            
            # Group by unit
            unit_orders = {}
            for order_str in legal_orders:
                parsed = self.parser.parse_order(order_str)
                if parsed:
                    unit = parsed.unit
                    if unit not in unit_orders:
                        unit_orders[unit] = []
                    unit_orders[unit].append(parsed.to_dict())
            
            return unit_orders
            
        except Exception as e:
            logger.error(f"Failed to enumerate legal orders: {e}")
            return {}
    
    def simulate(
        self, 
        state: Dict[str, Any], 
        plan: Plan,
        opponent_model: str = "random", 
        samples: int = 1
    ) -> List[SimulationResult]:
        """Simulate a plan against opponent models.
        
        Args:
            state: Current game state
            plan: Plan to simulate
            opponent_model: Type of opponent model
            samples: Number of simulation samples
            
        Returns:
            List of simulation results
        """
        return self.simulator.simulate(state, plan, opponent_model, samples)
    
    def repair(
        self, 
        state: Dict[str, Any], 
        plan: Plan
    ) -> Optional[Plan]:
        """Attempt to repair an illegal plan.
        
        Args:
            state: Current game state
            plan: Plan to repair
            
        Returns:
            Repaired plan if successful, None otherwise
        """
        logger.info("Attempting to repair plan")
        
        # Convert plan to orders
        orders = self.plan_to_engine_orders(plan)
        current_power = state.get('current_power', 'ENGLAND')
        
        if current_power not in orders:
            logger.warning("No orders found for current power")
            return None
        
        # Validate orders
        validation = self.validator.validate_orders(current_power, orders[current_power])
        
        if validation['valid']:
            logger.info("Plan is already valid")
            return plan
        
        # Try to repair illegal orders
        repaired_orders = []
        for order in orders[current_power]:
            if order in validation['illegal_orders']:
                # Try to find a legal alternative
                parsed = self.parser.parse_order(order)
                if parsed:
                    unit = parsed.unit
                    legal_alternatives = self.validator.get_legal_orders_for_unit(current_power, unit)
                    if legal_alternatives:
                        # Use first legal alternative
                        repaired_orders.append(legal_alternatives[0])
                        logger.info(f"Repaired {order} -> {legal_alternatives[0]}")
                    else:
                        # Fall back to hold
                        repaired_orders.append(f"{unit} HLD")
                        logger.info(f"Repaired {order} -> {unit} HLD")
                else:
                    # Skip invalid orders
                    continue
            else:
                repaired_orders.append(order)
        
        # Create repaired plan
        if repaired_orders:
            repaired_plan = Plan(
                orders=plan.orders.copy(),
                rationale=f"Repaired plan: {plan.rationale}",
                confidence=plan.confidence * 0.8,  # Reduce confidence for repairs
                status=PlanStatus.REPAIRED
            )
            
            logger.info("Plan repaired successfully")
            return repaired_plan
        
        logger.warning("Failed to repair plan")
        return None
    
    def score_plan(
        self,
        state: Dict[str, Any],
        plan: Plan,
        goals: Dict[str, Any]
    ) -> float:
        """Score a plan based on goals and heuristics.
        
        Args:
            state: Current game state
            plan: Plan to score
            goals: Player goals
            
        Returns:
            Plan score (higher is better)
        """
        score_obj = self.scorer.score_plan(state, plan, goals)
        return score_obj.total
    
    def search_plans(
        self,
        state: Dict[str, Any],
        goals: Dict[str, Any],
        max_depth: int = 2,
        max_plans: int = 10
    ) -> List[Plan]:
        """Search for plans using symbolic methods.
        
        Args:
            state: Current game state
            goals: Player goals
            max_depth: Maximum search depth
            max_plans: Maximum number of plans to return
            
        Returns:
            List of discovered plans
        """
        logger.info(f"Searching for plans with depth {max_depth}, max {max_plans}")
        
        plans = []
        
        # Get legal orders for current power
        current_power = state.get('current_power', 'ENGLAND')
        legal_orders = self.engine.get_legal_orders(current_power)
        
        if not legal_orders:
            logger.warning("No legal orders available")
            return plans
        
        # Generate simple plans by combining legal orders
        # This is a simplified search - in practice would use more sophisticated methods
        
        # Take first few legal orders as a simple plan
        for i in range(min(max_plans, len(legal_orders))):
            order = legal_orders[i]
            parsed = self.parser.parse_order(order)
            if parsed:
                plan_orders = {f"order_{0}": parsed.to_dict()}
                
                plan = Plan(
                    orders=plan_orders,
                    rationale=f"Symbolic search plan {i+1}",
                    confidence=0.6,
                    status=PlanStatus.PENDING
                )
                plans.append(plan)
        
        logger.info(f"Generated {len(plans)} plans from symbolic search")
        return plans
    
    # Helper methods
    
    def _detect_power_from_unit(self, unit: str) -> Optional[str]:
        """Detect power from unit string (simplified).
        
        Args:
            unit: Unit string
            
        Returns:
            Power name or None
        """
        # This is a simplified implementation
        # In practice, would need to track unit ownership properly
        return "ENGLAND"  # Placeholder
    
    def _format_order_string(self, order_data: Dict[str, Any]) -> Optional[str]:
        """Format order data to string.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            Formatted order string or None
        """
        unit = order_data.get('unit', '')
        action = order_data.get('action', '')
        
        if not unit or not action:
            return None
        
        if action == 'HLD':
            return f"{unit} HLD"
        elif action == 'MTO':
            target = order_data.get('target', '')
            return f"{unit} MTO {target}" if target else None
        elif action == 'SUP':
            support = order_data.get('support', '')
            target = order_data.get('target', '')
            return f"{unit} SUP {support} {target}" if support and target else None
        elif action == 'CVY':
            convoy = order_data.get('convoy', '')
            target = order_data.get('target', '')
            return f"{unit} CVY {convoy} {target}" if convoy and target else None
        elif action == 'RTO':
            target = order_data.get('target', '')
            return f"{unit} RTO {target}" if target else None
        elif action == 'DSB':
            return f"{unit} DSB"
        else:
            return None
