"""
Example: Using ADA-7 with LLM Aggregator

This example shows how to use the ADA-7 framework to make evidence-based
decisions about the LLM Aggregator system.
"""

from src.core.ada7 import (
    ADA7Assistant,
    Stage,
    ConfidenceLevel,
    create_citation,
    create_evidence,
)
from rich.console import Console
from rich.panel import Panel

console = Console()


def example_model_selection_decision():
    """
    Example: Using ADA-7 to decide on model selection strategy
    
    This demonstrates how ADA-7 can be used to make an evidence-based
    decision about which model selection strategy to implement.
    """
    
    console.print("\n[bold cyan]Example: Model Selection Strategy Decision[/bold cyan]\n")
    
    ada = ADA7Assistant()
    
    # Create evidence from research papers
    evidence = create_evidence(
        academic_papers=[
            create_citation(
                "Chen et al.",
                2023,
                "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance",
                "arXiv:2305.05176",
                relevance="Demonstrates 98% cost reduction with cascade routing"
            ),
            create_citation(
                "Jiang et al.",
                2024,
                "RouteLLM: Learning to Route LLMs with Preference Data",
                "arXiv:2406.18665",
                relevance="Shows 40% improvement in quality-cost trade-offs"
            )
        ],
        production_examples=[
            {
                "repo": "openrouter/openrouter",
                "stars": 5000,
                "pattern": "Static routing with fallbacks",
                "lesson": "Simple but not cost-optimized"
            },
            {
                "repo": "langchain/langchain",
                "stars": 75000,
                "pattern": "User-specified routing",
                "lesson": "Flexible but requires user knowledge"
            },
            {
                "repo": "BerriAI/litellm",
                "stars": 8000,
                "pattern": "Load balancing across providers",
                "lesson": "Good for reliability, not for cost optimization"
            }
        ],
        confidence=ConfidenceLevel.HIGH
    )
    
    # Define options
    options = [
        {
            "name": "Static Routing",
            "scalability_score": 7,
            "cost_optimization_score": 5,
            "implementation_complexity_score": 9,
            "quality_consistency_score": 8
        },
        {
            "name": "Round-Robin Load Balancing",
            "scalability_score": 8,
            "cost_optimization_score": 6,
            "implementation_complexity_score": 8,
            "quality_consistency_score": 7
        },
        {
            "name": "FrugalGPT Cascade Routing",
            "scalability_score": 9,
            "cost_optimization_score": 10,
            "implementation_complexity_score": 6,
            "quality_consistency_score": 9
        }
    ]
    
    # Make decision
    decision = ada.make_evidence_based_decision(
        decision_name="Model Selection Strategy for LLM Aggregator",
        options=options,
        evidence=evidence,
        criteria={
            "scalability": 0.20,
            "cost_optimization": 0.35,
            "implementation_complexity": 0.20,
            "quality_consistency": 0.25
        }
    )
    
    # Display results
    console.print(f"[bold green]✓ Recommended Strategy:[/bold green] {decision['recommended_option']}")
    console.print(f"[bold]Score:[/bold] {decision['score']:.2f}/10")
    console.print(f"[bold]Confidence:[/bold] {decision['evidence']['confidence'].upper()}")
    
    console.print("\n[yellow]Evidence Base:[/yellow]")
    console.print(f"  • {decision['evidence']['academic_citations']} academic papers")
    console.print(f"  • {decision['evidence']['production_examples']} production examples")
    
    console.print("\n[yellow]Implementation Plan:[/yellow]")
    console.print("  1. Implement cascade routing logic (13 story points)")
    console.print("  2. Add complexity analyzer (8 story points)")
    console.print("  3. Create performance tracking (5 story points)")
    console.print("  4. Add escalation logic (5 story points)")
    
    # Save decision to project context
    ada.save_stage_data(Stage.ARCHITECTURE, {
        "decision": decision,
        "next_steps": [
            "Implement FrugalCascadeRouter class",
            "Add TaskComplexityAnalyzer",
            "Integrate with MetaModelController"
        ]
    })
    
    console.print("\n[green]✓ Decision saved to project context[/green]")


def example_technology_selection():
    """
    Example: Using ADA-7 to select technologies for a component
    """
    
    console.print("\n[bold cyan]Example: Database Selection for User Management[/bold cyan]\n")
    
    ada = ADA7Assistant()
    
    # Create evidence
    evidence = create_evidence(
        academic_papers=[
            create_citation(
                "Cooper et al.",
                2023,
                "NoSQL Performance Comparison",
                "arXiv:2303.54321",
                relevance="Benchmark study: PostgreSQL 15ms P95, MongoDB 12ms P95, DynamoDB 8ms P95"
            ),
            create_citation(
                "Lee et al.",
                2024,
                "Database Total Cost of Ownership",
                "arXiv:2401.98765",
                relevance="TCO analysis showing PostgreSQL $600/mo, MongoDB $800/mo, DynamoDB $1200/mo"
            )
        ],
        production_examples=[
            {
                "repo": "strapi/strapi",
                "stars": 55000,
                "choice": "PostgreSQL",
                "lesson": "Reliable, extensive ecosystem, easier operations"
            },
            {
                "repo": "parse-server/parse-server",
                "stars": 20000,
                "choice": "MongoDB",
                "lesson": "Flexible schema, but requires tuning"
            },
            {
                "repo": "serverless/serverless",
                "stars": 45000,
                "choice": "DynamoDB",
                "lesson": "Scales infinitely but watch costs"
            }
        ],
        confidence=ConfidenceLevel.HIGH
    )
    
    # Define options
    options = [
        {
            "name": "PostgreSQL",
            "performance_score": 7,
            "cost_score": 9,
            "team_expertise_score": 9,
            "scalability_score": 7
        },
        {
            "name": "MongoDB",
            "performance_score": 8,
            "cost_score": 7,
            "team_expertise_score": 6,
            "scalability_score": 8
        },
        {
            "name": "DynamoDB",
            "performance_score": 10,
            "cost_score": 5,
            "team_expertise_score": 5,
            "scalability_score": 10
        }
    ]
    
    # Make decision
    decision = ada.make_evidence_based_decision(
        decision_name="Database for User Management",
        options=options,
        evidence=evidence,
        criteria={
            "performance": 0.25,
            "cost": 0.30,
            "team_expertise": 0.25,
            "scalability": 0.20
        }
    )
    
    # Display results
    console.print(f"[bold green]✓ Recommended Database:[/bold green] {decision['recommended_option']}")
    console.print(f"[bold]Score:[/bold] {decision['score']:.2f}/10")
    
    # Show all scores for comparison
    console.print("\n[yellow]All Options:[/yellow]")
    for option, score in sorted(decision['decision_matrix']['scores'].items(), 
                                key=lambda x: x[1], reverse=True):
        console.print(f"  • {option}: {score:.2f}")


def example_integration_with_meta_controller():
    """
    Example: How ADA-7 decisions integrate with MetaModelController
    """
    
    console.print("\n[bold cyan]Example: ADA-7 + MetaModelController Integration[/bold cyan]\n")
    
    integration_info = """
The MetaModelController already implements research-based model selection,
informed by ADA-7 principles:

1. **Evidence Base**: Uses FrugalGPT (arXiv:2305.05176) for cascade routing
2. **Academic Validation**: Implements RouteLLM (arXiv:2406.18665) patterns
3. **Production Learning**: Continuous feedback loop with performance tracking
4. **Decision Framework**: Task complexity analysis with confidence scoring

ADA-7 provides the methodology that guided these implementations.
    """
    
    console.print(Panel(integration_info, title="Integration", border_style="cyan"))
    
    console.print("\n[yellow]Key Integration Points:[/yellow]")
    console.print("  1. [cyan]MetaModelController[/cyan] uses ADA-7 Stage 2 (Architecture)")
    console.print("  2. [cyan]TaskComplexityAnalyzer[/cyan] implements ADA-7 evidence-based scoring")
    console.print("  3. [cyan]FrugalCascadeRouter[/cyan] follows ADA-7 decision framework")
    console.print("  4. [cyan]ExternalMemorySystem[/cyan] tracks Stage 7 (Maintenance) metrics")


def main():
    """Run all examples"""
    
    console.print("\n" + "="*80)
    console.print("[bold]ADA-7 Framework Integration Examples[/bold]")
    console.print("="*80)
    
    # Example 1: Model selection strategy
    example_model_selection_decision()
    
    console.print("\n" + "-"*80 + "\n")
    
    # Example 2: Technology selection
    example_technology_selection()
    
    console.print("\n" + "-"*80 + "\n")
    
    # Example 3: Integration explanation
    example_integration_with_meta_controller()
    
    console.print("\n" + "="*80)
    console.print("\n[green]✓ Examples completed![/green]")
    console.print("\nFor more information:")
    console.print("  • Run: [cyan]python ada7_cli.py guide[/cyan]")
    console.print("  • Read: [cyan]ADA_7_FRAMEWORK.md[/cyan]")
    console.print("  • Demo: [cyan]python ada7_demo.py[/cyan]")
    console.print()


if __name__ == "__main__":
    main()
