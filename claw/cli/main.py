"""Main CLI interface for CLAW."""

import json
import time
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from claw.meta.controller import MetaCognitiveController
from claw.neural.hf_llm import HuggingFaceLLM
from claw.symbolic.diplomacy_adapter.adapter import DiplomacyAdapter
from claw.logging import get_logger

# Initialize CLI
app = typer.Typer(
    name="claw",
    help="CLAW: Meta-Cognitive Neuro-Symbolic Planning Framework",
    no_args_is_help=True
)

console = Console()
logger = get_logger(__name__)


@app.command()
def demo_diplomacy(
    model_path: str = typer.Option(..., "--model-path", "-m", help="Path to local model"),
    unpredictability: float = typer.Option(0.3, "--unpredictability", "-u", help="Unpredictability setting (0.0-1.0)"),
    max_iters: int = typer.Option(5, "--max-iters", "-i", help="Maximum iterations"),
    time_budget: float = typer.Option(30.0, "--time-budget", "-t", help="Time budget in seconds"),
    output_dir: str = typer.Option("runs", "--output-dir", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run a Diplomacy demo with a single turn."""
    console.print(Panel.fit("CLAW Diplomacy Demo", style="bold blue"))
    
    # Create output directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_dir) / f"demo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    trace_file = run_dir / "trace.jsonl"
    
    try:
        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing components...", total=None)
            
            # Initialize LLM
            progress.update(task, description="Loading LLM...")
            llm = HuggingFaceLLM(model_path=model_path)
            
            # Initialize adapter
            progress.update(task, description="Initializing Diplomacy adapter...")
            adapter = DiplomacyAdapter()
            
            # Initialize controller
            progress.update(task, description="Initializing meta-cognitive controller...")
            controller = MetaCognitiveController(
                neural_generator=llm,
                symbolic_planner=adapter,
                domain_adapter=adapter,
                trace_file=str(trace_file)
            )
        
        # Get initial state
        state = adapter.get_initial_state()
        
        # Define goals
        goals = {
            "defend": ["LVP", "LON"],
            "attack": ["EDI", "PAR"],
            "ally": ["FRANCE"]
        }
        
        console.print(f"\n[bold]Game State:[/bold]")
        console.print(adapter.serialize_for_prompt(state))
        
        console.print(f"\n[bold]Goals:[/bold]")
        for goal_type, targets in goals.items():
            console.print(f"  {goal_type}: {', '.join(targets)}")
        
        # Generate plan
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating plan...", total=None)
            
            plan = controller.generate_plan(
                state=state,
                goals=goals,
                unpredictability=unpredictability,
                max_iters=max_iters,
                time_budget=time_budget
            )
        
        # Display results
        console.print(f"\n[bold green]Generated Plan:[/bold green]")
        display_plan(plan)
        
        # Save results
        results_file = run_dir / "results.json"
        save_results(results_file, plan, state, goals)
        
        console.print(f"\n[bold]Results saved to:[/bold] {run_dir}")
        console.print(f"[dim]Trace file: {trace_file}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def selfplay_diplomacy(
    model_path: str = typer.Option(..., "--model-path", "-m", help="Path to local model"),
    episodes: int = typer.Option(10, "--episodes", "-e", help="Number of episodes"),
    unpredictability: float = typer.Option(0.3, "--unpredictability", "-u", help="Unpredictability setting"),
    output_dir: str = typer.Option("runs", "--output-dir", "-o", help="Output directory")
):
    """Run self-play evaluation with multiple episodes."""
    console.print(Panel.fit("CLAW Diplomacy Self-Play", style="bold blue"))
    
    # Create output directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_dir) / f"selfplay_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize components
        llm = HuggingFaceLLM(model_path=model_path)
        adapter = DiplomacyAdapter()
        controller = MetaCognitiveController(
            neural_generator=llm,
            symbolic_planner=adapter,
            domain_adapter=adapter
        )
        
        results = []
        
        with Progress(console=console) as progress:
            task = progress.add_task("Running self-play episodes...", total=episodes)
            
            for episode in range(episodes):
                # Generate plan for this episode
                state = adapter.get_initial_state()
                goals = {
                    "defend": ["LVP"],
                    "attack": ["EDI"]
                }
                
                plan = controller.generate_plan(
                    state=state,
                    goals=goals,
                    unpredictability=unpredictability
                )
                
                # Store results
                results.append({
                    "episode": episode,
                    "plan": {
                        "orders": plan.orders,
                        "rationale": plan.rationale,
                        "confidence": plan.confidence,
                        "status": plan.status.value
                    }
                })
                
                progress.advance(task)
        
        # Save results
        results_file = run_dir / "selfplay_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[bold green]Self-play complete![/bold green]")
        console.print(f"Results saved to: {run_dir}")
        
        # Display summary
        display_selfplay_summary(results)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def inspect_trace(
    trace_file: str = typer.Argument(..., help="Path to trace file"),
    event_type: Optional[str] = typer.Option(None, "--event-type", "-t", help="Filter by event type"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum events to display")
):
    """Inspect a trace file."""
    console.print(Panel.fit("CLAW Trace Inspector", style="bold blue"))
    
    try:
        trace_path = Path(trace_file)
        if not trace_path.exists():
            console.print(f"[bold red]Trace file not found:[/bold red] {trace_file}")
            raise typer.Exit(1)
        
        # Load trace events
        events = []
        with open(trace_path, 'r') as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        
        # Filter by event type if specified
        if event_type:
            events = [e for e in events if e.get('event_type') == event_type]
        
        # Limit events
        events = events[-limit:] if limit > 0 else events
        
        console.print(f"\n[bold]Found {len(events)} events[/bold]")
        
        # Display events
        for i, event in enumerate(events):
            display_event(event, i + 1)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


def display_plan(plan):
    """Display a plan in a formatted table."""
    table = Table(title="Plan Details")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Rationale", plan.rationale)
    table.add_row("Confidence", f"{plan.confidence:.2f}")
    table.add_row("Status", plan.status.value)
    
    if plan.orders:
        table.add_row("Orders", "")
        for order_key, order_data in plan.orders.items():
            if isinstance(order_data, dict):
                unit = order_data.get('unit', 'Unknown')
                action = order_data.get('action', 'Unknown')
                target = order_data.get('target', '')
                order_str = f"{unit} {action}"
                if target:
                    order_str += f" {target}"
                table.add_row("", order_str)
    
    console.print(table)


def display_selfplay_summary(results: List[dict]):
    """Display self-play summary."""
    table = Table(title="Self-Play Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    total_episodes = len(results)
    avg_confidence = sum(r['plan']['confidence'] for r in results) / total_episodes
    legal_plans = sum(1 for r in results if r['plan']['status'] == 'legal')
    
    table.add_row("Total Episodes", str(total_episodes))
    table.add_row("Average Confidence", f"{avg_confidence:.2f}")
    table.add_row("Legal Plans", f"{legal_plans}/{total_episodes}")
    table.add_row("Success Rate", f"{legal_plans/total_episodes*100:.1f}%")
    
    console.print(table)


def display_event(event: dict, index: int):
    """Display a single trace event."""
    event_type = event.get('event_type', 'unknown')
    timestamp = event.get('timestamp', 0)
    data = event.get('data', {})
    
    # Format timestamp
    time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
    
    console.print(f"\n[bold]{index}. {event_type}[/bold] [{time_str}]")
    
    # Display event-specific data
    if event_type == 'plan_generation':
        plans = data.get('plans', [])
        console.print(f"  Generated {len(plans)} plans using {data.get('method', 'unknown')}")
        
    elif event_type == 'simulation':
        results = data.get('results', [])
        legal_count = sum(1 for r in results if r.get('is_legal', False))
        console.print(f"  Simulated {len(results)} samples, {legal_count} legal")
        
    elif event_type == 'decision':
        console.print(f"  Decision: {data.get('decision', 'unknown')}")
        console.print(f"  Reason: {data.get('reason', 'N/A')}")
        
    elif event_type == 'meta_decision':
        console.print(f"  Choice: {data.get('choice', 'unknown')}")
        params = data.get('parameters', {})
        if params:
            console.print(f"  Parameters: {params}")
    
    # Show raw data if verbose
    if len(str(data)) < 200:
        console.print(f"  Data: {data}")


def save_results(file_path: Path, plan, state: dict, goals: dict):
    """Save results to file."""
    results = {
        "plan": {
            "orders": plan.orders,
            "rationale": plan.rationale,
            "confidence": plan.confidence,
            "status": plan.status.value
        },
        "state": state,
        "goals": goals,
        "timestamp": time.time()
    }
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    app()
