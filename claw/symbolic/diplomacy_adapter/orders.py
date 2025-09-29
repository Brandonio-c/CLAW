"""Order parsing and validation for Diplomacy."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from claw.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedOrder:
    """A parsed Diplomacy order."""
    unit: str
    action: str
    target: Optional[str] = None
    support: Optional[str] = None
    convoy: Optional[str] = None
    original: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'unit': self.unit,
            'action': self.action,
            'original': self.original
        }
        if self.target:
            result['target'] = self.target
        if self.support:
            result['support'] = self.support
        if self.convoy:
            result['convoy'] = self.convoy
        return result


class OrderParser:
    """Parser for Diplomacy orders."""
    
    # Order patterns
    PATTERNS = {
        'HOLD': r'^([AF]\s+[A-Z]{3})\s+HLD$',
        'MOVE': r'^([AF]\s+[A-Z]{3})\s+MTO\s+([A-Z]{3})$',
        'SUPPORT': r'^([AF]\s+[A-Z]{3})\s+SUP\s+([AF]\s+[A-Z]{3})\s+([A-Z]{3})$',
        'CONVOY': r'^([AF]\s+[A-Z]{3})\s+CVY\s+([AF]\s+[A-Z]{3})\s+([A-Z]{3})$',
        'RETREAT': r'^([AF]\s+[A-Z]{3})\s+RTO\s+([A-Z]{3})$',
        'DISBAND': r'^([AF]\s+[A-Z]{3})\s+DSB$',
    }
    
    def __init__(self):
        """Initialize the order parser."""
        self.compiled_patterns = {
            action: re.compile(pattern, re.IGNORECASE)
            for action, pattern in self.PATTERNS.items()
        }
    
    def parse_order(self, order_str: str) -> Optional[ParsedOrder]:
        """Parse a single order string.
        
        Args:
            order_str: Order string to parse
            
        Returns:
            Parsed order or None if parsing fails
        """
        order_str = order_str.strip()
        if not order_str:
            return None
        
        # Try each pattern
        for action, pattern in self.compiled_patterns.items():
            match = pattern.match(order_str)
            if match:
                groups = match.groups()
                unit = groups[0].upper()
                
                if action == 'HOLD':
                    return ParsedOrder(unit=unit, action='HLD', original=order_str)
                elif action == 'MOVE':
                    return ParsedOrder(unit=unit, action='MTO', target=groups[1].upper(), original=order_str)
                elif action == 'SUPPORT':
                    return ParsedOrder(unit=unit, action='SUP', support=groups[1].upper(), target=groups[2].upper(), original=order_str)
                elif action == 'CONVOY':
                    return ParsedOrder(unit=unit, action='CVY', convoy=groups[1].upper(), target=groups[2].upper(), original=order_str)
                elif action == 'RETREAT':
                    return ParsedOrder(unit=unit, action='RTO', target=groups[1].upper(), original=order_str)
                elif action == 'DISBAND':
                    return ParsedOrder(unit=unit, action='DSB', original=order_str)
        
        logger.warning(f"Failed to parse order: {order_str}")
        return None
    
    def parse_orders(self, orders: List[str]) -> List[ParsedOrder]:
        """Parse multiple orders.
        
        Args:
            orders: List of order strings
            
        Returns:
            List of parsed orders (None entries for failed parses)
        """
        parsed = []
        for order in orders:
            parsed_order = self.parse_order(order)
            parsed.append(parsed_order)
        return parsed
    
    def format_order(self, parsed_order: ParsedOrder) -> str:
        """Format a parsed order back to string.
        
        Args:
            parsed_order: Parsed order to format
            
        Returns:
            Formatted order string
        """
        if parsed_order.action == 'HLD':
            return f"{parsed_order.unit} HLD"
        elif parsed_order.action == 'MTO':
            return f"{parsed_order.unit} MTO {parsed_order.target}"
        elif parsed_order.action == 'SUP':
            return f"{parsed_order.unit} SUP {parsed_order.support} {parsed_order.target}"
        elif parsed_order.action == 'CVY':
            return f"{parsed_order.unit} CVY {parsed_order.convoy} {parsed_order.target}"
        elif parsed_order.action == 'RTO':
            return f"{parsed_order.unit} RTO {parsed_order.target}"
        elif parsed_order.action == 'DSB':
            return f"{parsed_order.unit} DSB"
        else:
            return parsed_order.original


class OrderValidator:
    """Validator for Diplomacy orders."""
    
    def __init__(self, engine):
        """Initialize validator with engine reference.
        
        Args:
            engine: DiplomacyEngine instance
        """
        self.engine = engine
        self.parser = OrderParser()
    
    def validate_order(self, power: str, order: str) -> Tuple[bool, List[str]]:
        """Validate a single order.
        
        Args:
            power: Power name
            order: Order string
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Parse the order
        parsed = self.parser.parse_order(order)
        if not parsed:
            errors.append(f"Failed to parse order: {order}")
            return False, errors
        
        # Check if order is legal according to engine
        if not self.engine.is_legal_order(power, order):
            errors.append(f"Order not legal: {order}")
        
        # Additional validation
        if not self._validate_unit_exists(power, parsed.unit):
            errors.append(f"Unit does not exist: {parsed.unit}")
        
        if parsed.action in ['MTO', 'SUP', 'CVY', 'RTO'] and not parsed.target:
            errors.append(f"Order requires target: {order}")
        
        if parsed.action in ['SUP', 'CVY'] and not (parsed.support or parsed.convoy):
            errors.append(f"Order requires support/convoy unit: {order}")
        
        return len(errors) == 0, errors
    
    def validate_orders(self, power: str, orders: List[str]) -> Dict[str, Any]:
        """Validate multiple orders.
        
        Args:
            power: Power name
            orders: List of order strings
            
        Returns:
            Validation result dictionary
        """
        results = {
            'valid': True,
            'legal_orders': [],
            'illegal_orders': [],
            'errors': [],
            'parsed_orders': []
        }
        
        for order in orders:
            is_valid, errors = self.validate_order(power, order)
            parsed = self.parser.parse_order(order)
            
            results['parsed_orders'].append(parsed)
            
            if is_valid:
                results['legal_orders'].append(order)
            else:
                results['illegal_orders'].append(order)
                results['errors'].extend(errors)
                results['valid'] = False
        
        return results
    
    def _validate_unit_exists(self, power: str, unit: str) -> bool:
        """Check if unit exists for power.
        
        Args:
            power: Power name
            unit: Unit string
            
        Returns:
            True if unit exists, False otherwise
        """
        try:
            units = self.engine.get_units(power)
            return unit in units
        except Exception:
            return False
    
    def get_legal_orders_for_unit(self, power: str, unit: str) -> List[str]:
        """Get legal orders for a specific unit.
        
        Args:
            power: Power name
            unit: Unit string
            
        Returns:
            List of legal orders for the unit
        """
        try:
            all_legal = self.engine.get_legal_orders(power)
            unit_orders = [order for order in all_legal if order.startswith(unit)]
            return unit_orders
        except Exception as e:
            logger.error(f"Failed to get legal orders for unit {unit}: {e}")
            return []
    
    def suggest_repairs(self, power: str, illegal_orders: List[str]) -> Dict[str, List[str]]:
        """Suggest repairs for illegal orders.
        
        Args:
            power: Power name
            illegal_orders: List of illegal orders
            
        Returns:
            Dictionary mapping orders to repair suggestions
        """
        suggestions = {}
        
        for order in illegal_orders:
            parsed = self.parser.parse_order(order)
            if not parsed:
                suggestions[order] = ["Order format is invalid"]
                continue
            
            # Get legal orders for the unit
            legal_orders = self.get_legal_orders_for_unit(power, parsed.unit)
            
            if legal_orders:
                suggestions[order] = [f"Try: {legal_orders[0]}"]
            else:
                suggestions[order] = ["Unit has no legal orders - try HLD"]
        
        return suggestions
