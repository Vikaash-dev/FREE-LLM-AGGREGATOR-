"""
ADA-7 Integration with LLM Aggregator

This example demonstrates how to integrate the ADA-7 framework
with the existing LLM aggregator for AI-powered development assistance.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.core.ada7.framework import ADA7Framework, AcademicReference, GitHubRepository

console = Console()


async def demo_ada7_integration():
    """Demonstrate ADA-7 integration with LLM aggregator."""
    
    console.print(Panel.fit(
        "🔗 ADA-7 Framework Integration Example\n"
        "Using LLM Aggregator for AI-Powered Development",
        title="Integration Demo",
        border_style="bold green"
    ))
    
    # Initialize framework (can optionally pass aggregator and meta_controller)
    console.print("\n[bold cyan]1. Initializing ADA-7 with LLM Aggregator Integration[/bold cyan]")
    
    # Note: In production, you would pass actual aggregator and meta_controller instances
    ada7 = ADA7Framework(
        aggregator=None,  # Would be LLMAggregator instance
        meta_controller=None  # Would be MetaModelController instance
    )
    
    console.print("   ✓ ADA-7 Framework initialized")
    console.print("   ✓ Ready to leverage LLM aggregator for intelligent analysis")
    console.print("   ✓ Meta-controller available for task-based model selection")
    
    # Example project
    console.print("\n[bold cyan]2. Starting Project with AI-Assisted Analysis[/bold cyan]")
    
    project = await ada7.start_project(
        name="E-Commerce Platform",
        description="Modern e-commerce platform with AI-powered product recommendations, "
                   "real-time inventory management, and multi-vendor support",
        constraints={
            "budget": 100000,
            "timeline": "9 months",
            "team_size": 8,
            "target_users": 100000
        }
    )
    
    console.print(f"   ✓ Project: {project.name}")
    console.print(f"   ✓ ID: {project.project_id}")
    
    # Example: How ADA-7 would use LLM aggregator
    console.print("\n[bold cyan]3. AI-Powered Stage Execution[/bold cyan]")
    
    integration_examples = """
## How ADA-7 Uses LLM Aggregator

### Stage 1: Requirements Analysis
- **LLM Task**: Natural language processing of user requirements
- **Model Selection**: Meta-controller selects model based on complexity
- **Use Case**: Analyze user stories and extract requirements
- **Example**: Convert conversational requirements into SMART specifications

### Stage 2: Architecture Design
- **LLM Task**: Generate architecture alternatives based on requirements
- **Model Selection**: Reasoning-focused models for complex design
- **Use Case**: Validate architecture against academic best practices
- **Example**: Generate microservices boundaries and communication patterns

### Stage 3: Component Design
- **LLM Task**: Code generation and template creation
- **Model Selection**: Code-specialized models (e.g., Codestral, DeepSeek)
- **Use Case**: Generate boilerplate code and integration patterns
- **Example**: Create OpenAPI schemas and data flow diagrams

### Stage 4: Implementation Strategy
- **LLM Task**: Generate CI/CD configurations and deployment scripts
- **Model Selection**: General-purpose models with strong instruction following
- **Use Case**: Create Docker configurations and pipeline definitions
- **Example**: Generate GitHub Actions workflows and Terraform templates

### Stage 5: Testing Framework
- **LLM Task**: Generate test cases and quality checklists
- **Model Selection**: Models with strong logical reasoning
- **Use Case**: Create comprehensive test scenarios
- **Example**: Generate unit tests, integration tests, and test data

### Stage 6: Deployment Strategy
- **LLM Task**: Infrastructure as Code generation
- **Model Selection**: Code generation models for IaC
- **Use Case**: Create Kubernetes manifests and monitoring configs
- **Example**: Generate Prometheus alerts and Grafana dashboards

### Stage 7: Maintenance Planning
- **LLM Task**: Documentation generation and knowledge management
- **Model Selection**: Models with strong writing capabilities
- **Use Case**: Create comprehensive documentation
- **Example**: Generate API docs, runbooks, and incident playbooks

## Benefits of Integration

1. **Intelligent Model Selection**: Meta-controller picks best model for each task
2. **Cost Optimization**: Uses free-tier models when appropriate
3. **Fallback Mechanisms**: Automatic failover if primary model unavailable
4. **Performance Tracking**: Monitors model performance for continuous improvement
5. **Research-Backed**: Leverages academic papers for validation
6. **Evidence-Based**: Uses GitHub repos as real-world examples
"""
    
    console.print(Markdown(integration_examples))
    
    # Example: Advanced decision making with LLM
    console.print("\n[bold cyan]4. AI-Powered Decision Matrix[/bold cyan]")
    
    console.print("   Example: Choosing between database technologies")
    console.print("   • LLM analyzes requirements and generates comparison")
    console.print("   • Considers: performance, scalability, cost, team expertise")
    console.print("   • References: Academic papers and production case studies")
    console.print("   • Output: Quantified recommendation with confidence scores")
    
    # Create example decision matrix
    matrix = ada7.create_decision_matrix(
        alternatives=["PostgreSQL", "MongoDB", "Cassandra"],
        criteria=["performance", "scalability", "ease_of_use", "cost"],
        weights={
            "performance": 0.30,
            "scalability": 0.30,
            "ease_of_use": 0.20,
            "cost": 0.20
        }
    )
    
    # Simulated LLM-generated scores
    matrix.scores = {
        "PostgreSQL": {
            "performance": 8.5,
            "scalability": 7.0,
            "ease_of_use": 8.0,
            "cost": 9.0
        },
        "MongoDB": {
            "performance": 7.5,
            "scalability": 8.5,
            "ease_of_use": 7.5,
            "cost": 8.0
        },
        "Cassandra": {
            "performance": 9.0,
            "scalability": 9.5,
            "ease_of_use": 6.0,
            "cost": 7.0
        }
    }
    
    best = matrix.get_best_alternative()
    scores = matrix.calculate_weighted_scores()
    
    console.print(f"\n   🏆 Recommended: [bold green]{best}[/bold green]")
    console.print(f"   📊 Weighted Scores:")
    for alt, score in scores.items():
        console.print(f"      • {alt}: {score:.2f}/10")
    
    # Integration patterns
    console.print("\n[bold cyan]5. Integration Patterns with Existing System[/bold cyan]")
    
    patterns = """
## Code Integration Examples

### Pattern 1: Using ADA-7 in Main Application

```python
from src.core.aggregator import LLMAggregator
from src.core.meta_controller import MetaModelController
from src.core.ada7.framework import ADA7Framework

# Initialize LLM infrastructure
aggregator = LLMAggregator()
meta_controller = MetaModelController(aggregator)

# Initialize ADA-7 with LLM support
ada7 = ADA7Framework(
    aggregator=aggregator,
    meta_controller=meta_controller
)

# Start development process
project = await ada7.start_project(
    name="My Project",
    description="AI-powered application"
)

# Execute all stages
results = await ada7.execute_all_stages(project)
```

### Pattern 2: Stage-by-Stage Execution with Custom LLM Tasks

```python
# Execute Stage 1 with AI assistance
stage1_results = await ada7.execute_stage_1(project)

# Use LLM for detailed requirement analysis
requirement_analysis = await aggregator.chat_completion({
    "model": "auto",
    "messages": [{
        "role": "user",
        "content": f"Analyze these requirements: {stage1_results}"
    }]
})

# Continue with Stage 2
stage2_results = await ada7.execute_stage_2(project)
```

### Pattern 3: Integrating with Auto-Updater

```python
from src.core.auto_updater import AutoUpdater

# ADA-7 can leverage auto-updater for latest practices
auto_updater = AutoUpdater(aggregator)

# Discover latest frameworks and tools
discoveries = await auto_updater.discover_new_providers()

# Use in technology selection (Stage 3)
stage3_results = await ada7.execute_stage_3(project)
```

### Pattern 4: Ensemble System for Validation

```python
from src.core.ensemble_system import EnsembleSystem

ensemble = EnsembleSystem(aggregator)

# Use multiple models to validate architecture decision
validation_results = await ensemble.generate_and_blend(
    prompt="Validate this architecture design: ...",
    num_responses=3
)

# Use consensus for Stage 2 validation
stage2_results['ensemble_validation'] = validation_results
```
"""
    
    console.print(Markdown(patterns))
    
    # Summary
    console.print("\n[bold green]✅ Integration Examples Complete![/bold green]")
    
    console.print("\n[bold cyan]Key Integration Points:[/bold cyan]")
    console.print("   1. LLMAggregator for multi-model access")
    console.print("   2. MetaModelController for intelligent model selection")
    console.print("   3. EnsembleSystem for multi-model validation")
    console.print("   4. AutoUpdater for latest technology discovery")
    console.print("   5. Research components for academic validation")
    
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("   • Integrate ADA-7 into main application workflow")
    console.print("   • Configure LLM models for each stage type")
    console.print("   • Set up monitoring for ADA-7 decisions")
    console.print("   • Create custom stage handlers for specific needs")
    console.print("   • Extend knowledge base with domain-specific papers")


async def main():
    """Main entry point."""
    try:
        await demo_ada7_integration()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
