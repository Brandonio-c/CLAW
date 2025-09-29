 """Simulation engine for Diplomacy plans."""

import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from claw.interfaces.symbolic import SimulationResult
from claw.types import Plan
from claw.symbolic.diplomacy_adapter.engine import DiplomacyEngine
from claw.symbolic.diplomacy_adapter.orders import OrderParser, OrderValidator
from claw.logging import get_logger

logger = get_logger(__name__)


class DiplomacySimulator:
    """Simulator for Diplomacy plans against opponent models."""
    
    def __init__(self, engine: DiplomacyEngine):
        """Initialize simulator with engine.
        
        Args:
            engine: DiplomacyEngine instance
        """
        self.engine = engine
        self.parser = OrderParser()
        self.validator = OrderValidator(engine)
    
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
        logger.info(f"Simulating plan with {opponent_model} opponents, {samples} samples")
        
        results = []
        for i in range(samples):
            try:
                result = self._simulate_single(state, plan, opponent_model, i)
                results.append(result)
            except Exception as e:
                logger.warning(f"Simulation {i+1} failed: {e}")
                # Create failed result
                results.append(SimulationResult(
                    new_state=state,
                    is_legal=False,
                    score=0.0,
                    diagnostics={'error': str(e)}
                ))
        
        logger.info(f"Completed {len(results)} simulations")
        return results
    
    def _simulate_single(
        self,
        state: Dict[str, Any],
        plan: Plan,
        opponent_model: str,
        sample_id: int
    ) -> SimulationResult:
        """Run a single simulation.
        
        Args:
            state: Current game state
            plan: Plan to simulate
            opponent_model: Opponent model type
            sample_id: Sample identifier
            
        Returns:
            Simulation result
        """
        # Create a copy of the engine for simulation
        sim_engine = DiplomacyEngine()
        
        # Set up the game state (simplified - in practice would need full state restoration)
        # For now, we'll work with the current state
        
        # Convert plan to orders
        our_orders = self._plan_to_orders(plan)
        
        # Generate opponent orders
        opponent_orders = self._generate_opponent_orders(state, opponent_model)
        
        # Combine all orders
        all_orders = {**our_orders, **opponent_orders}
        
        # Validate our orders
        our_power = self._get_our_power(state)
        validation = self.validator.validate_orders(our_power, our_orders.get(our_power, []))
        
        if not validation['valid']:
            return SimulationResult(
                new_state=state,
                is_legal=False,
                score=0.0,
                diagnostics={
                    'illegal_orders': validation['illegal_orders'],
                    'errors': validation['errors']
                }
            )
        
        # Apply orders and get result
        try:
            result = sim_engine.apply_orders(all_orders)
            new_state = sim_engine.get_state()
            
            # Calculate score
            score = self._calculate_score(state, new_state, plan)
            
            return SimulationResult(
                new_state=new_state,
                is_legal=True,
                score=score,
                diagnostics={
                    'applied_orders': all_orders,
                    'result': result
                }
            )
            
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            return SimulationResult(
                new_state=state,
                is_legal=False,
                score=0.0,
                diagnostics={'error': str(e)}
            )
    
    def _plan_to_orders(self, plan: Plan) -> Dict[str, List[str]]:
        """Convert plan to engine orders.
        
        Args:
            plan: Plan to convert
            
        Returns:
            Dictionary of orders by power
        """
        orders = {}
        
        for order_key, order_data in plan.orders.items():
            if not isinstance(order_data, dict):
                continue
            
            # Extract power from unit (simplified)
            unit = order_data.get('unit', '')
            if not unit:
                continue
            
            # This is a simplified mapping - in practice would need proper power detection
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
    
    def _generate_opponent_orders(
        self,
        state: Dict[str, Any],
        opponent_model: str
    ) -> Dict[str, List[str]]:
        """Generate orders for opponent powers.
        
        Args:
            state: Current game state
            opponent_model: Type of opponent model
            
        Returns:
            Dictionary of orders by power
        """
        opponent_orders = {}
        
        # Get all powers except ours
        our_power = self._get_our_power(state)
        all_powers = state.get('powers', [])
        opponent_powers = [p for p in all_powers if p != our_power]
        
        for power in opponent_powers:
            if opponent_model == "random":
                orders = self._generate_random_orders(power, state)
            elif opponent_model == "hold":
                orders = self._generate_hold_orders(power, state)
            else:
                orders = self._generate_random_orders(power, state)
            
            if orders:
                opponent_orders[power] = orders
        
        return opponent_orders
    
    def _generate_random_orders(self, power: str, state: Dict[str, Any]) -> List[str]:
        """Generate random legal orders for a power.
        
        Args:
            power: Power name
            state: Current game state
            
        Returns:
            List of random orders
        """
        try:
            legal_orders = self.engine.get_legal_orders(power)
            if not legal_orders:
                return []
            
            # Select random subset of legal orders
            num_orders = min(len(legal_orders), random.randint(1, len(legal_orders)))
            return random.sample(legal_orders, num_orders)
        except Exception as e:
            logger.warning(f"Failed to generate random orders for {power}: {e}")
            return []
    
    def _generate_hold_orders(self, power: str, state: Dict[str, Any]) -> List[str]:
        """Generate hold orders for all units of a power.
        
        Args:
            power: Power name
            state: Current game state
            
        Returns:
            List of hold orders
        """
        try:
            units = self.engine.get_units(power)
            return [f"{unit} HLD" for unit in units]
        except Exception as e:
            logger.warning(f"Failed to generate hold orders for {power}: {e}")
            return []
    
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
    
    def _get_our_power(self, state: Dict[str, Any]) -> str:
        """Get our power name from state.
        
        Args:
            state: Game state
            
        Returns:
            Our power name
        """
        return state.get('current_power', 'ENGLAND')
    
    def _calculate_score(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        plan: Plan
    ) -> float:
        """Calculate score for the simulation.
        
        Args:
            old_state: State before simulation
            new_state: State after simulation
            plan: Plan that was simulated
            
        Returns:
            Score value
        """
        # Simple scoring based on supply center changes
        our_power = self._get_our_power(old_state)
        
        old_centers = set(old_state.get('centers', {}).get(our_power, []))
        new_centers = set(new_state.get('centers', {}).get(our_power, []))
        
        # Basic score: +1 for each center gained, -1 for each lost
        score = len(new_centers - old_centers) - len(old_centers - new_centers)
        
        # Add confidence bonus
        score += plan.confidence * 0.1
        
        return float(score)