"""Diplomacy engine wrapper."""

import random
from typing import Dict, Any, List, Optional
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from diplomacy.utils.imports import from_saved_game_format
from claw.logging import get_logger

logger = get_logger(__name__)


class DiplomacyEngine:
    """Wrapper around the python-diplomacy engine."""
    
    def __init__(self, game_id: Optional[str] = None):
        """Initialize the Diplomacy engine.
        
        Args:
            game_id: Optional game ID for loading existing game
        """
        self.game_id = game_id or f"game_{random.randint(1000, 9999)}"
        self.game = None
        self._initialize_game()
    
    def _initialize_game(self):
        """Initialize a new Diplomacy game."""
        try:
            self.game = Game()
            logger.info(f"Initialized new Diplomacy game: {self.game_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Diplomacy game: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current game state.
        
        Returns:
            Dictionary containing current game state
        """
        if not self.game:
            raise RuntimeError("Game not initialized")
        
        state = {
            'game_id': self.game_id,
            'phase': self.game.get_current_phase(),
            'units': dict(self.game.get_units()),
            'centers': dict(self.game.get_centers()),
            'supply_centers': dict(self.game.get_supply_centers()),
            'powers': list(self.game.get_map().powers),
            'current_power': self.game.get_current_power(),
            'phase_data': self.game.get_phase_data(),
            'map': {
                'provinces': list(self.game.get_map().provinces),
                'supply_centers': list(self.game.get_map().supply_centers),
                'coasts': list(self.game.get_map().coasts)
            }
        }
        
        return state
    
    def apply_orders(self, orders: Dict[str, List[str]]) -> Dict[str, Any]:
        """Apply orders to the game.
        
        Args:
            orders: Dictionary mapping power names to lists of orders
            
        Returns:
            Result of applying orders
        """
        if not self.game:
            raise RuntimeError("Game not initialized")
        
        try:
            # Set orders for each power
            for power, power_orders in orders.items():
                if power_orders:  # Only set if there are orders
                    self.game.set_orders(power, power_orders)
            
            # Process the orders
            result = self.game.process()
            
            logger.info(f"Applied orders for {len(orders)} powers")
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply orders: {e}")
            raise
    
    def get_legal_orders(self, power: str) -> List[str]:
        """Get legal orders for a power.
        
        Args:
            power: Power name
            
        Returns:
            List of legal orders
        """
        if not self.game:
            raise RuntimeError("Game not initialized")
        
        try:
            orders = self.game.get_all_possible_orders()
            power_orders = orders.get(power, [])
            return power_orders
        except Exception as e:
            logger.error(f"Failed to get legal orders for {power}: {e}")
            return []
    
    def get_units(self, power: str) -> List[str]:
        """Get units for a power.
        
        Args:
            power: Power name
            
        Returns:
            List of unit strings
        """
        if not self.game:
            raise RuntimeError("Game not initialized")
        
        try:
            units = self.game.get_units(power)
            return list(units)
        except Exception as e:
            logger.error(f"Failed to get units for {power}: {e}")
            return []
    
    def get_centers(self, power: str) -> List[str]:
        """Get supply centers for a power.
        
        Args:
            power: Power name
            
        Returns:
            List of supply center names
        """
        if not self.game:
            raise RuntimeError("Game not initialized")
        
        try:
            centers = self.game.get_centers(power)
            return list(centers)
        except Exception as e:
            logger.error(f"Failed to get centers for {power}: {e}")
            return []
    
    def is_legal_order(self, power: str, order: str) -> bool:
        """Check if an order is legal.
        
        Args:
            power: Power name
            order: Order string
            
        Returns:
            True if order is legal, False otherwise
        """
        if not self.game:
            raise RuntimeError("Game not initialized")
        
        try:
            legal_orders = self.get_legal_orders(power)
            return order in legal_orders
        except Exception as e:
            logger.error(f"Failed to check order legality: {e}")
            return False
    
    def get_phase_data(self) -> Dict[str, Any]:
        """Get current phase data.
        
        Returns:
            Phase data dictionary
        """
        if not self.game:
            raise RuntimeError("Game not initialized")
        
        try:
            return self.game.get_phase_data()
        except Exception as e:
            logger.error(f"Failed to get phase data: {e}")
            return {}
    
    def save_game(self, filepath: str) -> None:
        """Save game to file.
        
        Args:
            filepath: Path to save file
        """
        if not self.game:
            raise RuntimeError("Game not initialized")
        
        try:
            saved_game = to_saved_game_format(self.game)
            with open(filepath, 'w') as f:
                import json
                json.dump(saved_game, f, indent=2)
            logger.info(f"Game saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save game: {e}")
            raise
    
    def load_game(self, filepath: str) -> None:
        """Load game from file.
        
        Args:
            filepath: Path to load file
        """
        try:
            with open(filepath, 'r') as f:
                import json
                saved_game = json.load(f)
            
            self.game = from_saved_game_format(saved_game)
            logger.info(f"Game loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load game: {e}")
            raise
    
    def reset_game(self) -> None:
        """Reset game to initial state."""
        self._initialize_game()
        logger.info("Game reset to initial state")
