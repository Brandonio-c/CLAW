#!/usr/bin/env python3
"""Example script: Run a full Diplomacy game stub with CLAW."""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from claw.meta.controller import MetaCognitiveController
from claw.neural.hf_llm import HuggingFaceLLM
from claw.symbolic.diplomacy_adapter.adapter import DiplomacyAdapter
from claw.logging import get_logger

logger = get_logger(__name__)


class GameState:
    """Simple game state tracker."""
    
    def __init__(self):
        self.turn = 0
        self.phase = "SPRING 1901"
        self.powers = ["ENGLAND", "FRANCE", "GERMANY", "AUSTRIA", "ITALY", "RUSSIA", "TURKEY"]
        self.current_power = "ENGLAND"
        self.centers = {
            "ENGLAND": ["LVP", "LON", "EDI"],
            "FRANCE": ["PAR", "MAR", "BRE"],
            "GERMANY": ["BER", "MUN", "KIE"],
            "AUSTRIA": ["VIE", "BUD", "TRI"],
            "ITALY": ["ROM", "VEN", "NAP"],
            "RUSSIA": ["MOS", "WAR", "STP", "SEV"],
            "TURKEY": ["CON", "ANK", "SMY"]
        }
        self.units = {
            "ENGLAND": ["A LVP", "F LON", "F EDI"],
            "FRANCE": ["A PAR", "A MAR", "F BRE"],
            "GERMANY": ["A BER", "A MUN", "F KIE"],
            "AUSTRIA": ["A VIE", "A BUD", "F TRI"],
            "ITALY": ["A ROM", "A VEN", "F NAP"],
            "RUSSIA": ["A MOS", "A WAR", "F STP", "A SEV"],
            "TURKEY": ["A CON", "A ANK", "F SMY"]
        }
        self.game_log = []
    
    def get_state_for_power(self, power: str) -> Dict[str, Any]:
        """Get state for a specific power."""
        return {
            "turn": self.turn,
            "phase": self.phase,
            "powers": self.powers,
            "current_power": power,
            "units": self.units,
            "centers": self.centers,
            "map": {
                "provinces": ["LVP", "LON", "EDI", "PAR", "MAR", "BRE", "BER", "MUN", "KIE",
                             "VIE", "BUD", "TRI", "ROM", "VEN", "NAP", "MOS", "WAR", "STP", "SEV",
                             "CON", "ANK", "SMY"],
                "supply_centers": ["LVP", "LON", "EDI", "PAR", "MAR", "BRE", "BER", "MUN", "KIE",
                                  "VIE", "BUD", "TRI", "ROM", "VEN", "NAP", "MOS", "WAR", "STP", "SEV",
                                  "CON", "ANK", "SMY"],
                "coasts": []
            }
        }
    
    def advance_turn(self):
        """Advance to next turn."""
        self.turn += 1
        if self.phase.endswith("1901"):
            self.phase = "FALL 1901"
        elif self.phase.endswith("FALL"):
            year = int(self.phase.split()[-1])
            self.phase = f"SPRING {year + 1}"
        else:
            year = int(self.phase.split()[-1])
            self.phase = f"FALL {year}"
    
    def log_action(self, power: str, action: str, details: Dict[str, Any]):
        """Log an action."""
        self.game_log.append({
            "turn": self.turn,
            "phase": self.phase,
            "power": power,
            "action": action,
            "details": details,
            "timestamp": time.time()
        })


def generate_goals_for_power(power: str, turn: int) -> Dict[str, Any]:
    """Generate goals for a power based on turn and position."""
    base_goals = {
        "defend": [],
        "attack": [],
        "ally": [],
        "avoid": []
    }
    
    # Simple goal generation based on power
    if power == "ENGLAND":
        base_goals["defend"] = ["LVP", "LON"]
        base_goals["attack"] = ["EDI", "PAR"] if turn < 3 else ["BER", "MOS"]
        base_goals["ally"] = ["FRANCE"] if turn < 5 else []
    elif power == "FRANCE":
        base_goals["defend"] = ["PAR", "MAR"]
        base_goals["attack"] = ["BRE", "SPA"] if turn < 3 else ["MUN", "ROM"]
        base_goals["ally"] = ["ENGLAND"] if turn < 5 else []
    elif power == "GERMANY":
        base_goals["defend"] = ["BER", "MUN"]
        base_goals["attack"] = ["KIE", "DEN"] if turn < 3 else ["PAR", "WAR"]
        base_goals["ally"] = ["AUSTRIA"] if turn < 5 else []
    # Add more powers as needed...
    
    return base_goals


def run_game_turn(
    controller: MetaCognitiveController,
    adapter: DiplomacyAdapter,
    game_state: GameState,
    power: str,
    turn: int
) -> Dict[str, Any]:
    """Run a single turn for a power."""
    logger.info(f"Running turn {turn} for {power}")
    
    # Get state and goals
    state = game_state.get_state_for_power(power)
    goals = generate_goals_for_power(power, turn)
    
    # Generate plan
    plan = controller.generate_plan(
        state=state,
        goals=goals,
        unpredictability=0.3,
        max_iters=3,
        time_budget=15.0
    )
    
    # Log the action
    game_state.log_action(power, "plan_generated", {
        "plan": {
            "orders": plan.orders,
            "rationale": plan.rationale,
            "confidence": plan.confidence,
            "status": plan.status.value
        },
        "goals": goals
    })
    
    return {
        "power": power,
        "plan": plan,
        "goals": goals,
        "state": state
    }


def main():
    """Run a full game stub."""
    parser = argparse.ArgumentParser(description="Run a full Diplomacy game stub with CLAW")
    parser.add_argument("--model-path", required=True, help="Path to local model")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum turns to simulate")
    parser.add_argument("--powers", nargs="+", default=["ENGLAND", "FRANCE", "GERMANY"],
                       help="Powers to include in the game")
    parser.add_argument("--output-dir", default="runs", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / f"full_game_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    trace_file = run_dir / "trace.jsonl"
    
    try:
        logger.info("Initializing CLAW components...")
        
        # Initialize components
        llm = HuggingFaceLLM(model_path=args.model_path)
        adapter = DiplomacyAdapter()
        
        controller = MetaCognitiveController(
            neural_generator=llm,
            symbolic_planner=adapter,
            domain_adapter=adapter,
            trace_file=str(trace_file)
        )
        
        # Initialize game state
        game_state = GameState()
        
        # Run game
        game_results = []
        
        for turn in range(1, args.max_turns + 1):
            logger.info(f"Starting turn {turn}")
            
            turn_results = {}
            
            for power in args.powers:
                try:
                    result = run_game_turn(controller, adapter, game_state, power, turn)
                    turn_results[power] = result
                    
                    if args.verbose:
                        print(f"\n{power} (Turn {turn}):")
                        print(f"  Rationale: {result['plan'].rationale}")
                        print(f"  Confidence: {result['plan'].confidence:.2f}")
                        print(f"  Orders: {len(result['plan'].orders)}")
                        
                except Exception as e:
                    logger.error(f"Error in turn {turn} for {power}: {e}")
                    turn_results[power] = {
                        "power": power,
                        "error": str(e),
                        "plan": None
                    }
            
            game_results.append({
                "turn": turn,
                "phase": game_state.phase,
                "results": turn_results
            })
            
            # Advance turn
            game_state.advance_turn()
            
            # Simple stopping condition
            if turn >= 5 and all(
                result.get("plan") and result["plan"].confidence < 0.3
                for result in turn_results.values()
                if result.get("plan")
            ):
                logger.info("Stopping early due to low confidence")
                break
        
        # Save results
        final_results = {
            "game_log": game_state.game_log,
            "turn_results": game_results,
            "final_state": {
                "turn": game_state.turn,
                "phase": game_state.phase,
                "centers": game_state.centers,
                "units": game_state.units
            },
            "parameters": {
                "model_path": args.model_path,
                "max_turns": args.max_turns,
                "powers": args.powers
            },
            "timestamp": time.time()
        }
        
        results_file = run_dir / "game_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("CLAW DIPLOMACY GAME SUMMARY")
        print("="*60)
        print(f"Turns completed: {len(game_results)}")
        print(f"Final phase: {game_state.phase}")
        print(f"Powers: {', '.join(args.powers)}")
        print(f"\nResults saved to: {run_dir}")
        print(f"Trace file: {trace_file}")
        
        # Print turn-by-turn summary
        if args.verbose:
            print(f"\nTurn-by-turn summary:")
            for turn_result in game_results:
                print(f"\nTurn {turn_result['turn']} ({turn_result['phase']}):")
                for power, result in turn_result['results'].items():
                    if result.get('plan'):
                        print(f"  {power}: {result['plan'].rationale[:50]}... "
                              f"(conf: {result['plan'].confidence:.2f})")
                    else:
                        print(f"  {power}: ERROR - {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
