# Diplomacy Quickstart Guide

This guide shows you how to get started with CLAW using the Diplomacy domain adapter.

## Prerequisites

1. Install CLAW in development mode:
   ```bash
   pip install -e ".[dev,llama]"
   ```

2. Download a local LLM model. For example, using Hugging Face:
   ```bash
   # Download a small model for testing
   huggingface-cli download microsoft/DialoGPT-medium --local-dir ./models/dialogpt-medium
   ```

## Basic Usage

### 1. Command Line Interface

The easiest way to get started is using the CLI:

```bash
# Run a single turn demo
claw demo-diplomacy --model-path ./models/dialogpt-medium --unpredictability 0.3

# Run self-play evaluation
claw selfplay-diplomacy --model-path ./models/dialogpt-medium --episodes 5

# Inspect trace files
claw inspect-trace runs/demo_2024-01-01_12-00-00/trace.jsonl
```

### 2. Python API

For more control, use the Python API:

```python
from claw.meta.controller import MetaCognitiveController
from claw.neural.hf_llm import HuggingFaceLLM
from claw.symbolic.diplomacy_adapter.adapter import DiplomacyAdapter

# Initialize components
llm = HuggingFaceLLM(model_path="./models/dialogpt-medium")
adapter = DiplomacyAdapter()

# Create controller
controller = MetaCognitiveController(
    neural_generator=llm,
    symbolic_planner=adapter,
    domain_adapter=adapter
)

# Generate a plan
state = adapter.get_initial_state()
goals = {
    "defend": ["LVP", "LON"],
    "attack": ["EDI", "PAR"],
    "ally": ["FRANCE"]
}

plan = controller.generate_plan(
    state=state,
    goals=goals,
    unpredictability=0.3,
    max_iters=5,
    time_budget=30.0
)

print(f"Generated plan: {plan.rationale}")
print(f"Confidence: {plan.confidence}")
print(f"Orders: {plan.orders}")
```

## Understanding the Output

### Plan Structure

A plan contains:
- **orders**: Dictionary of unit orders
- **rationale**: Human-readable explanation
- **confidence**: LLM confidence (0.0-1.0)
- **status**: Plan status (PENDING, LEGAL, ILLEGAL, etc.)

### Order Format

Orders follow the standard Diplomacy format:
- `A LVP HLD` - Army in Liverpool holds
- `F LON MTO ENG` - Fleet in London moves to English Channel
- `A PAR SUP A LVP EDI` - Army in Paris supports Army in Liverpool to Edinburgh
- `F ENG CVY A LVP EDI` - Fleet in English Channel convoys Army in Liverpool to Edinburgh

### Goals

Goals are specified as dictionaries:
- `defend`: List of locations to defend
- `attack`: List of locations to attack
- `ally`: List of powers to ally with
- `avoid`: List of locations to avoid

## Advanced Configuration

### Custom Meta-Policy

```python
from claw.meta.policy import HeuristicMetaPolicy

# Create custom policy
policy = HeuristicMetaPolicy(
    max_iters=10,
    time_budget=60.0,
    confidence_threshold=0.8,
    improvement_threshold=0.1
)

controller = MetaCognitiveController(
    neural_generator=llm,
    symbolic_planner=adapter,
    domain_adapter=adapter,
    meta_policy=policy
)
```

### Custom Scoring

```python
from claw.symbolic.diplomacy_adapter.score import DiplomacyScorer, ScoringWeights

# Custom scoring weights
weights = ScoringWeights(
    center_gain=2.0,      # Weight for gaining supply centers
    positional_advantage=1.0,  # Weight for positional moves
    risk_penalty=-0.5,    # Penalty for risky moves
    goal_fit=1.5,         # Weight for goal alignment
    confidence_bonus=0.3  # Bonus for high confidence
)

scorer = DiplomacyScorer(weights)
```

### Trace Analysis

```python
from claw.meta.tracing import TraceLogger

# Load trace
trace = TraceLogger()
trace.load_trace("runs/demo_2024-01-01_12-00-00/trace.jsonl")

# Get all events
events = trace.get_events()

# Get specific event types
plan_events = trace.get_events("plan_generation")
sim_events = trace.get_events("simulation")

# Analyze planning process
for event in events:
    if event.event_type == "iteration_summary":
        print(f"Iteration {event.data['iteration']}: "
              f"best_score={event.data['best_score']:.2f}, "
              f"improvement={event.data['improvement']:.2f}")
```

## Troubleshooting

### Common Issues

1. **Model not found**: Make sure the model path is correct and the model is downloaded
2. **CUDA out of memory**: Use CPU mode by setting `device="cpu"` in the LLM configuration
3. **Invalid orders**: Check that the model is generating valid Diplomacy orders
4. **Low confidence**: Try adjusting the temperature or using a different model

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tips

1. Use smaller models for faster iteration
2. Reduce `max_samples` for quicker generation
3. Lower `time_budget` for faster results
4. Use `opponent_model="hold"` for simpler simulations

## Next Steps

1. Try different models and compare results
2. Experiment with different goal combinations
3. Implement custom domain adapters
4. Add new symbolic planners
5. Create custom meta-policies

For more examples, see the `examples/` directory.
