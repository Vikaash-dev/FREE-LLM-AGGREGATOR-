# AREA-7 Framework Documentation

## Core Identity: Autonomous Research & Engineering Agent (AREA-7)

AREA-7 is an autonomous AI agent that functions at the intersection of scientific discovery and software engineering. It systematically explores complex problems, forms and validates novel hypotheses, and translates validated research into production-quality code.

## Key Methodologies

1. **Sakuna AI Scientist Framework** - For systematic discovery and hypothesis validation
2. **Paper2Code Protocol** - For translating research into production code
3. **Multi-Agent Coordination** - 11 specialized agent personalities working together
4. **Structured Memory Management** - Short-term and long-term memory in markdown files
5. **Comprehensive Audit Logging** - All operations logged with structured JSON

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     AREA-7 Master Controller                 │
│              (Determines operational mode)                   │
└────────────┬───────────────────────────┬────────────────────┘
             │                           │
    ┌────────▼────────┐         ┌───────▼──────────┐
    │ Discovery Mode  │         │ Implementation   │
    │ (Sakuna AI      │         │ Mode (Paper2Code)│
    │  Scientist)     │         │                  │
    └────────┬────────┘         └───────┬──────────┘
             │                           │
    ┌────────▼───────────────────────────▼──────────┐
    │        Multi-Agent Coordination System        │
    │    (11 Specialized Agent Personalities)       │
    └────────┬──────────────────────────────────────┘
             │
    ┌────────▼──────────┬──────────────────┐
    │ Memory Management │  Audit Logging   │
    │  - Short-term     │  - Operation logs│
    │  - Long-term      │  - Statistics    │
    │  - Rules          │  - Metrics       │
    └───────────────────┴──────────────────┘
```

## Part 1: Master Directive (Guiding Loop)

The AREA-7 Master Controller determines the appropriate operational mode for any given goal:

### Operational Modes

1. **DISCOVERY Mode** - For exploratory problems requiring novel solutions
   - Uses Sakuna AI Scientist Protocol (Parts 2a & 2b)
   - Generates and validates hypotheses
   - Identifies promising phenomena

2. **IMPLEMENTATION Mode** - For known papers or pre-defined algorithms
   - Uses Paper2Code Protocol (Part 3)
   - Directly translates to production code
   - Creates comprehensive test suites

3. **HYBRID Mode** - Complex goals requiring both discovery and implementation
   - Runs full discovery phase first
   - Then implements validated findings
   - End-to-end solution

## Part 2: Sakuna AI Scientist Protocol

### Phase 1: Broad Exploration (Sakuna v1 Workflow)

**Objective**: Quickly generate and test a wide range of simple hypotheses

**Steps**:
1. **Ingest & Deconstruct** - Break problem into fundamental components
2. **Generate Diverse Hypotheses** - Create 5+ falsifiable "What if...?" hypotheses
3. **Design High-Throughput Experiments** - Minimal, low-cost experiments for each
4. **Execute & Observe** - Run experiments in parallel, identify anomalies
5. **Report Findings** - Select most promising phenomena for deep investigation

**Requirements**:
- Minimum 5 hypotheses per problem
- Each hypothesis must have novelty_score ≥ 0.7
- Run experiments in parallel
- Identify "surprising" or anomalous outcomes

### Phase 2: Deep Investigation (Sakuna v2 Workflow)

**Objective**: Take a single promising phenomenon and develop it into validated theory

**Steps**:
1. **Isolate the Phenomenon** - Form clear, specific hypothesis to explain it
2. **Design Rigorous Experiment** - Controlled experiment with proper baselines
3. **Execute and Analyze** - Collect detailed data and analyze results
4. **Iterate and Refine** - Loop: Hypothesize → Experiment → Analyze → Refine
5. **Synthesize & Propose Implementation** - Summarize validated findings

**Validation Criteria**:
- Confidence score > 0.8
- Statistical significance (p < 0.05)
- Reproducibility across 3+ runs
- Clear mechanism identified

## Part 3: Paper2Code Protocol

**Objective**: Translate validated findings or research papers into production-quality code

### 6-Step Implementation Pipeline

**Step 1: Full Paper Deconstruction**
- Extract core algorithm
- Identify data structures
- Map mathematical equations
- Define experimental setup
- Note expected results

**Step 2: Create Language-Agnostic Blueprint**
- Write detailed pseudocode
- Specify input/output interfaces
- Define helper functions
- Include complexity analysis

**Step 3: Environment & Dependency Scaffolding**
- Identify required libraries
- Generate dependency files
- Create setup scripts
- Configure virtual environment

**Step 4: Modular Code Implementation**
- Translate pseudocode to clean code
- Single-purpose functions/classes
- Inline comments linking to source
- Comprehensive error handling

**Step 5: Write Replication & Unit Tests**
- Unit tests for each function
- Integration tests for pipelines
- Replication tests matching paper results
- Performance benchmarks

**Step 6: Final Packaging & Documentation**
- Organize directory structure
- Write comprehensive README
- Document setup/execution
- Summarize replication results

## Multi-Agent Personality System

AREA-7 employs 11 specialized agent personalities:

### Implemented Agents

1. **Hypothesis Generator** - Creates novel, diverse hypotheses
   - Strategies: combination, relaxation, inverse thinking, simplification, learning
   - Scores hypotheses on novelty, feasibility, and impact

2. **Experiment Designer** - Designs rigorous controlled experiments
   - Defines control and experimental groups
   - Specifies metrics and success criteria
   - Creates detailed procedures

3. **Data Analyzer** - Analyzes experimental results
   - Identifies patterns and anomalies
   - Performs statistical analysis
   - Generates recommendations

4. **Code Architect** - Designs system architecture
   - Defines components and interfaces
   - Maps data flows
   - Applies design patterns

5. **Research Paper Generator** - Creates formal research papers
   - Follows academic paper structure
   - Includes abstract, methodology, results
   - Generates LaTeX-ready content

### Additional Agent Personalities (Scaffolded)

6. **Implementation Engineer** - Writes production code
7. **Test Engineer** - Creates comprehensive test suites
8. **Performance Optimizer** - Optimizes for efficiency
9. **Security Auditor** - Ensures security best practices
10. **Documentation Specialist** - Maintains documentation
11. **Quality Reviewer** - Reviews and validates work

## Memory Management

### Short-Term Memory (`memory/short_term_memory.md`)

Stores current session context:
- Active hypotheses
- Recent experiments
- Pending actions
- Current problem statement

**Management**:
- Updated after each operation
- Cleared when tasks complete
- Max size: 1000 entries

### Long-Term Memory (`memory/long_term_memory.md`)

Archives validated findings:
- Validated hypotheses
- Successful patterns
- Failed patterns (lessons learned)
- Implementation history

**Management**:
- Append-only (no deletions)
- Organized by category
- No size limit

### Operational Rules (`memory/rules.md`)

Defines framework behavior:
- Core principles (scientific rigor, minimal changes)
- Discovery phase rules
- Implementation phase rules
- Agent behavior rules
- Quality standards
- Emergency protocols

## Audit Logging

All operations are logged to `audit_logs/` with structured JSON format:

```json
{
  "timestamp": "2025-10-02T07:47:36.123456",
  "operation": "generate_hypotheses",
  "agent_id": "hypothesis_generator_1759391256",
  "operation_type": "hypothesis_generation",
  "input_data": {...},
  "output_data": {...},
  "metrics": {"hypothesis_count": 5},
  "success": true,
  "error_message": null
}
```

### Audit Statistics

Access operation statistics:
- Total operations
- Success rate
- Operations by type
- Operations by agent
- Average metrics

## Usage Examples

### Example 1: Discovery Mode

```python
from area7_framework import AREA7Master

# Initialize framework
area7 = AREA7Master()

# Define exploratory goal
goal = {
    "description": "Optimize LLM routing for minimal latency",
    "exploratory": True,
    "novel": True,
    "constraints": ["Maintain accuracy", "Max 100ms overhead"],
    "success_criteria": ["20% latency reduction", "No accuracy loss"]
}

# Execute
results = await area7.execute(goal)
```

### Example 2: Implementation Mode

```python
# Define implementation goal
goal = {
    "description": "Implement validated routing algorithm",
    "paper_reference": "Validated routing mechanism",
    "exploratory": False,
    "mechanism": "Priority-based routing with learned weights",
    "metrics": {"accuracy": 0.92, "latency": 0.85}
}

# Execute Paper2Code pipeline
results = await area7.execute(goal)
```

### Example 3: Multi-Agent Pipeline

```python
from multi_agent_personalities import MultiAgentCoordinator
from area7_framework import MemoryManager, AuditLogger

# Initialize coordinator
memory = MemoryManager()
audit = AuditLogger()
coordinator = MultiAgentCoordinator(memory, audit)

# Define problem
problem = {
    "description": "Optimize system performance",
    "constraints": ["Limited resources"],
    "success_criteria": ["30% improvement"]
}

# Execute multi-agent pipeline
results = await coordinator.execute_pipeline(problem)
```

## Testing

Run comprehensive test suite:

```bash
# Run all tests
pytest tests/test_area7_framework.py -v

# Run specific test class
pytest tests/test_area7_framework.py::TestMemoryManager -v

# Run with coverage
pytest tests/test_area7_framework.py --cov=area7_framework --cov-report=html
```

**Test Coverage**: 24 tests covering:
- Memory management (4 tests)
- Audit logging (3 tests)
- Sakuna AI Scientist (4 tests)
- Paper2Code protocol (4 tests)
- AREA7 Master (3 tests)
- Multi-agent system (4 tests)
- Integration workflows (2 tests)

## Quality Assurance

### Code Quality Standards
- All code must be modular and well-documented
- Follow PEP 8 for Python
- Include inline comments linking to source
- Maintain 80%+ test coverage

### Validation Requirements
- Unit tests for all functions
- Integration tests for pipelines
- Replication tests matching results
- Performance benchmarks documented

### Scientific Rigor Checklist
- ✓ Hypothesis clearly stated and falsifiable
- ✓ Experimental design controls for confounding variables
- ✓ Statistical tests appropriate for data distribution
- ✓ Baseline comparisons fair and comprehensive
- ✓ Results reproducible with provided code
- ✓ Limitations and assumptions explicitly stated

## File Structure

```
area7_framework/
├── area7_framework.py           # Core framework implementation
├── multi_agent_personalities.py # Multi-agent system
├── tests/
│   └── test_area7_framework.py # Comprehensive test suite
├── memory/                      # Auto-generated
│   ├── short_term_memory.md    # Current session context
│   ├── long_term_memory.md     # Validated findings archive
│   ├── rules.md                # Operational guidelines
│   └── research_paper_*.md     # Generated research papers
├── audit_logs/                  # Auto-generated
│   └── audit_log_*.jsonl       # Structured operation logs
└── AREA7_FRAMEWORK.md          # This documentation
```

## Integration with Existing Systems

To integrate AREA-7 with existing experimental optimizer:

```python
from area7_framework import AREA7Master
from experimental_optimizer import ExperimentalAggregator

# Initialize both systems
area7 = AREA7Master()
experimental = ExperimentalAggregator()

# Use AREA-7 for research-driven optimization
async def research_driven_optimization():
    # Define research goal
    goal = {
        "description": "Discover novel optimization strategies",
        "exploratory": True,
        "novel": True
    }
    
    # Run discovery
    results = await area7.execute(goal)
    
    # Apply findings to experimental system
    if results.get("validated"):
        await experimental.apply_optimization(results["discovery"])
```

## Performance Characteristics

- **Discovery Phase**: 0.5-2.0 seconds for 5 hypotheses
- **Implementation Phase**: 0.01-0.1 seconds for code generation
- **Multi-Agent Pipeline**: 0.5-1.0 seconds for complete workflow
- **Memory Operations**: <1ms for read/write
- **Audit Logging**: <1ms per operation

## Limitations and Future Work

### Current Limitations
- Hypothesis generation uses templates (could use LLM)
- Experiments are simulated (need real execution)
- Limited to Python for code generation
- No parallel agent execution yet

### Future Enhancements
1. LLM integration for hypothesis generation
2. Real experiment execution engine
3. Multi-language code generation
4. Parallel agent execution
5. Continuous learning from outcomes
6. Integration with external research databases
7. Automatic paper submission pipeline

## References

1. **Sakuna AI Scientist Framework**: "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery"
2. **Paper2Code Methodology**: Systematic translation of research to production code
3. **Multi-Agent Systems**: Coordination and collaboration patterns
4. **Structured Logging**: Best practices for audit trails
5. **Scientific Method**: Hypothesis generation and validation

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please follow:
1. Code quality standards (PEP 8)
2. Comprehensive tests for new features
3. Documentation updates
4. Audit logging for new operations

## Contact

For questions or issues, please open an issue on GitHub.

---

**Generated by**: AREA-7 Framework Documentation System
**Version**: 1.0.0
**Last Updated**: 2025-10-02
