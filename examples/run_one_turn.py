#!/usr/bin/env python3
"""Example script: Run a single Diplomacy turn with CLAW."""

import argparse
import json
import time
from pathlib import Path

from claw.meta.controller import MetaCognitiveController
from claw.neural.hf_llm import HuggingFaceLLM
from claw.symbolic.diplomacy_adapter.adapter import DiplomacyAdapter
from claw.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run a single Diplomacy turn."""
    parser = argparse.ArgumentParser(description="Run a single Diplomacy turn with CLAW")
    parser.add_argument("--model-path", required=True, help="Path to local model")
    parser.add_argument("--unpredictability", type=float, default=0.3, 
                       help="Unpredictability setting (0.0-1.0)")
    parser.add_argument("--max-iters", type=int, default=5, 
                       help="Maximum iterations")
    parser.add_argument("--time-budget", type=float, default=30.0, 
                       help="Time budget in seconds")
    parser.add_argument("--output-dir", default="runs", 
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_dir) / f"single_turn_{timestamp}"
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
        
        # Get initial state
        state = adapter.get_initial_state()
        logger.info(f"Initial state: {adapter.serialize_for_prompt(state)}")
        
        # Define goals
        goals = {
            "defend": ["LVP", "LON"],
            "attack": ["EDI", "PAR"],
            "ally": ["FRANCE"]
        }
        
        logger.info(f"Goals: {goals}")
        
        # Generate plan
        logger.info("Generating plan...")
        start_time = time.time()
        
        plan = controller.generate_plan(
            state=state,
            goals=goals,
            unpredictability=args.unpredictability,
            max_iters=args.max_iters,
            time_budget=args.time_budget
        )
        
        generation_time = time.time() - start_time
        
        # Display results
        print("\n" + "="*60)
        print("CLAW DIPLOMACY TURN RESULTS")
        print("="*60)
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Plan confidence: {plan.confidence:.2f}")
        print(f"Plan status: {plan.status.value}")
        print(f"\nRationale: {plan.rationale}")
        print(f"\nOrders:")
        
        for order_key, order_data in plan.orders.items():
            if isinstance(order_data, dict):
                unit = order_data.get('unit', 'Unknown')
                action = order_data.get('action', 'Unknown')
                target = order_data.get('target', '')
                support = order_data.get('support', '')
                convoy = order_data.get('convoy', '')
                
                order_str = f"  {unit} {action}"
                if target:
                    order_str += f" {target}"
                if support:
                    order_str += f" (supporting {support})"
                if convoy:
                    order_str += f" (convoying {convoy})"
                
                print(order_str)
        
        # Save results
        results = {
            "plan": {
                "orders": plan.orders,
                "rationale": plan.rationale,
                "confidence": plan.confidence,
                "status": plan.status.value
            },
            "state": state,
            "goals": goals,
            "generation_time": generation_time,
            "timestamp": time.time(),
            "parameters": {
                "unpredictability": args.unpredictability,
                "max_iters": args.max_iters,
                "time_budget": args.time_budget
            }
        }
        
        results_file = run_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {run_dir}")
        print(f"Trace file: {trace_file}")
        
        if args.verbose:
            print(f"\nDetailed results: {results_file}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
