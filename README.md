# PROMETHEUS AGI Framework

## Overview
The PROMETHEUS AGI Framework is an advanced cognitive architecture designed to explore general intelligence capabilities based on the principles defined by François Chollet. It integrates multiple modules across perception, memory, learning, reasoning, planning, and social interaction to achieve adaptive, flexible, and meta-cognitive performance.

## Key Features
- **Modular Architecture**: Separated into functional components such as Memory, Learning, Perception, Planning, Reasoning, and Interaction.
- **Neural-Symbolic Integration**: Combines neural models with symbolic reasoning for improved abstraction and generalization.
- **Dynamic Meta-Learning**: Supports task adaptation through meta-learning techniques like MAML.
- **Multi-Agent Interaction**: Theory-of-Mind and social cognition modules for collaborative interactions.
- **Self-Assessment and Reflection**: Monitors performance, detects anomalies, and generates reflective explanations.

## Code Structure

```
prometheus_10/
├── benchmark/          # Evaluation and benchmarking tools
├── core/               # Core attention and global control modules
├── interaction/        # Social cognition and interaction management
├── interaction_multi_agent/ # Multi-agent interaction manager
├── knowledge/          # Knowledge graph and ontology builder
├── learning/           # Meta-learning and dynamic architecture components
├── memory/             # Memory modules (episodic, semantic, unsupervised)
├── meta/               # Meta-narrative and causal reasoning
├── perception/         # Perception modules for multimodal input
├── reasoning/          # Abstract and causal reasoning engines
├── planning/           # Task planning and scheduling modules
├── cognition/          # Cognitive control and curiosity-driven modules
├── rl/                 # Reinforcement learning components
├── debug_utils.py      # Utilities for debugging and visualization
├── requirements.txt    # Python dependencies
└── main.py             # Entry point for execution
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EliasofIX/prometheus-agi-framework.git
cd prometheus-agi-framework
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the framework, simply execute the main file:
```bash
python main.py
```

## Module Details

### 1. **Benchmark**
- `arc_benchmark.py`: Tests model performance on ARC-like abstract reasoning tasks.

### 2. **Core**
- `adaptive_attention.py`: Dynamically allocates attention to relevant streams.
- `global_controller.py`: Coordinates system-wide operations.

### 3. **Interaction**
- Manages agent interactions using Theory-of-Mind and social cognition mechanisms.

### 4. **Knowledge**
- `knowledge_graph.py`: Graph-based knowledge representation.
- `ontology_builder.py`: Generates abstract concepts from observations.

### 5. **Learning**
- Advanced meta-learning with architecture evolution capabilities.

### 6. **Memory**
- Episodic, semantic, and unsupervised memory models for information retention.

### 7. **Meta**
- High-level narrative generation and causal experiments for self-reflection.

### 8. **Perception**
- Multimodal sensory integration and language grounding.

### 9. **Reasoning**
- Symbolic and abstract reasoning with a neural-symbolic bridge.

### 10. **Planning**
- Hierarchical task planning and scheduling.

### 11. **Cognition**
- Cognitive control mechanisms driven by intrinsic and meta-curiosity.

### 12. **Reinforcement Learning**
- RL agent for task-independent adaptive learning.

## Example Output

Upon execution, the framework will:
- Generate knowledge graphs for visual inspection.
- Simulate task planning and reasoning.
- Demonstrate adaptive attention switching.
- Display meta-reflections and self-assessments.

## Contributing
Contributions are welcome! Please feel free to submit issues, suggest enhancements, or open pull requests.

## License
MIT License

## Contact
For inquiries, discussions, or collaborations, reach out to: [GitHub Profile](https://github.com/EliasofIX)

---

*PROMETHEUS: Pioneering the next step toward adaptable, reflective, and general artificial intelligence.*

