<div align="center">
  <img src="CLAW_logo.png" alt="CLAW Logo" width="1000" height="1000">
</div>

# CLAW: Meta-Cognitive Neuro-Symbolic Planning Framework

A domain-general meta-cognitive neuro-symbolic planning framework that combines local LLMs with symbolic planners through a meta-cognitive controller (MCC). The first domain adapter is Diplomacy.

## Features

- **Meta-cognitive loop**: Monitor → Evaluate → Regulate
- **Modular design**: Plug any local LLM and symbolic planner
- **Domain-general**: Easy to add new domains via adapters
- **Controlled unpredictability**: Adjustable randomness and exploration
- **Structured outputs**: JSON-schema validated LLM responses
- **Comprehensive tracing**: Full decision audit trail

## Quick Start

### Installation

```bash
# Install in development mode
pip install -e .

# With optional dependencies
pip install -e ".[llama,quantization,dev]"
```

### Diplomacy Demo

```bash
# Run a single turn with a local model
claw demo-diplomacy --model-path /path/to/model --unpredictability 0.3

# Self-play evaluation
claw selfplay-diplomacy --episodes 10

# Inspect trace files
claw inspect-trace runs/2024-01-01_12-00-00/trace.jsonl
```

### Basic Usage

```python
from claw.meta.controller import MetaCognitiveController
from claw.neural.hf_llm import HuggingFaceLLM
from claw.symbolic.diplomacy_adapter.adapter import DiplomacyAdapter
from claw.symbolic.diplomacy_adapter.engine import DiplomacyEngine

# Initialize components
llm = HuggingFaceLLM(model_path="/path/to/model")
adapter = DiplomacyAdapter()
engine = DiplomacyEngine()

# Create controller
mcc = MetaCognitiveController(
    neural_generator=llm,
    symbolic_planner=adapter,
    domain_adapter=adapter
)

# Generate a plan
state = {"units": {...}, "centers": {...}}
goals = {"defend": ["LVP"], "attack": ["EDI"]}
plan = mcc.generate_plan(state, goals, unpredictability=0.3)
```

## Architecture

### Core Components

- **Meta-Cognitive Controller (MCC)**: Orchestrates the planning loop
- **Neural Strategy Generator**: Local LLM wrapper for plan generation
- **Symbolic Planner**: Domain-specific planning and simulation
- **Domain Adapter**: Converts between domain and framework representations
- **Meta Policy**: Decides when to use LLM vs symbolic search

### Meta-Cognitive Loop

1. **Monitor**: Track plan quality, legality, compute budget, confidence
2. **Evaluate**: Simulate candidate plans with opponent models
3. **Regulate**: Accept/repair/replan based on results and policy

## Project Structure

```
claw/
├── claw/
│   ├── interfaces/          # Core protocol definitions
│   ├── meta/               # Meta-cognitive controller
│   ├── neural/             # LLM wrappers and schemas
│   ├── symbolic/           # Symbolic planners and adapters
│   └── cli/                # Command-line interface
├── examples/               # Usage examples
└── tests/                  # Test suite
```

## Development

```bash
# Run tests
pytest

# Format code
black claw/
isort claw/

# Type checking
mypy claw/
```

## License

MIT License - see LICENSE file for details.