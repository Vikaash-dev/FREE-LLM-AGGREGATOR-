"""
ADA-7 Framework Demo

Demonstrates the Advanced Development Assistant (ADA-7) framework
with all 7 evolutionary stages for multi-project development.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import print as rprint

# Import ADA-7 components
from src.core.ada7.framework import ADA7Framework, AcademicReference, GitHubRepository

console = Console()


async def demo_ada7_framework():
    """Demonstrate ADA-7 framework capabilities."""
    
    console.print(Panel.fit(
        "🎯 Advanced Development Assistant (ADA-7) Framework\n"
        "7-Stage Evolutionary Development Methodology\n"
        "with Academic Research Integration",
        title="ADA-7 Demo",
        border_style="bold blue"
    ))
    
    # Initialize framework
    console.print("\n[bold cyan]Initializing ADA-7 Framework...[/bold cyan]")
    ada7 = ADA7Framework()
    
    # Populate knowledge base
    console.print("\n[bold blue]1. Building Knowledge Base[/bold blue]")
    
    # Add academic papers
    papers = [
        AcademicReference(
            authors="Chen et al.",
            year=2023,
            title="DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines",
            arxiv_id="2310.03714",
            citation_count=150
        ),
        AcademicReference(
            authors="Wu et al.",
            year=2023,
            title="AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
            arxiv_id="2308.08155",
            citation_count=200
        ),
        AcademicReference(
            authors="Newman",
            year=2015,
            title="Building Microservices: Designing Fine-Grained Systems",
            arxiv_id="ISBN-978-1491950357",
            citation_count=5000
        )
    ]
    
    for paper in papers:
        ada7.add_academic_reference(paper)
    
    console.print(f"   ✓ Added {len(papers)} academic papers to knowledge base")
    
    # Add GitHub repositories
    repos = [
        GitHubRepository(
            owner="microsoft",
            name="autogen",
            url="https://github.com/microsoft/autogen",
            stars=25500,
            last_commit_days_ago=1,
            description="Multi-agent conversation framework"
        ),
        GitHubRepository(
            owner="stanfordnlp",
            name="dspy",
            url="https://github.com/stanfordnlp/dspy",
            stars=12800,
            last_commit_days_ago=2,
            description="Programming framework for LMs"
        ),
        GitHubRepository(
            owner="langchain-ai",
            name="langchain",
            url="https://github.com/langchain-ai/langchain",
            stars=87200,
            last_commit_days_ago=0,
            description="LLM application framework"
        )
    ]
    
    for repo in repos:
        ada7.add_github_repository(repo)
    
    console.print(f"   ✓ Added {len(repos)} GitHub repositories to knowledge base")
    
    # Start new project
    console.print("\n[bold blue]2. Starting New Project[/bold blue]")
    
    project = await ada7.start_project(
        name="AI-Powered Task Manager",
        description="Intelligent task management system with AI-powered prioritization, "
                   "natural language processing, and team collaboration features",
        constraints={
            "budget": 50000,
            "timeline": "6 months",
            "team_size": 5,
            "technology_preference": "Python/React"
        }
    )
    
    console.print(f"   ✓ Project created: {project.name}")
    console.print(f"   • Project ID: {project.project_id}")
    console.print(f"   • Budget: ${project.constraints['budget']:,}")
    console.print(f"   • Timeline: {project.constraints['timeline']}")
    console.print(f"   • Team Size: {project.constraints['team_size']} developers")
    
    # Execute stages
    console.print("\n[bold blue]3. Executing 7 Development Stages[/bold blue]")
    
    # Stage 1: Requirements Analysis
    console.print("\n[cyan]Stage 1: Requirements Analysis & Competitive Intelligence[/cyan]")
    stage1_results = await ada7.execute_stage_1(project)
    
    console.print(f"   ✓ Created {len(stage1_results['user_personas'])} user personas")
    console.print(f"   ✓ Generated {len(stage1_results['user_stories'])} user stories")
    console.print(f"   ✓ Analyzed {len(stage1_results['competitors'])} competitors")
    console.print(f"   ✓ Identified {len(stage1_results['feature_gaps'])} feature gaps")
    console.print(f"   ✓ Defined {len(stage1_results['requirements'])} requirements")
    
    # Display sample requirement
    if stage1_results['requirements']:
        req = stage1_results['requirements'][0]
        console.print(f"\n   📋 Sample Requirement: [bold]{req['id']}[/bold] - {req['title']}")
        console.print(f"      Priority: {req['priority']}")
        console.print(f"      Type: {req['type']}")
    
    # Stage 2: Architecture Design
    console.print("\n[cyan]Stage 2: Architecture Design & Academic Validation[/cyan]")
    stage2_results = await ada7.execute_stage_2(project)
    
    console.print(f"   ✓ Generated {len(stage2_results['architecture_variants'])} architecture variants")
    console.print(f"   ✓ Performed academic validation with research papers")
    console.print(f"   ✓ Created decision matrix with weighted scoring")
    console.print(f"   ✓ Conducted risk assessment")
    console.print(f"   ✓ Recommended: {stage2_results['recommended_architecture']}")
    
    # Display architecture comparison
    arch_table = Table(title="Architecture Comparison")
    arch_table.add_column("Architecture", style="cyan")
    arch_table.add_column("Type", style="yellow")
    arch_table.add_column("Latency", justify="right")
    arch_table.add_column("Throughput", justify="right")
    arch_table.add_column("Score", justify="right", style="green")
    
    for variant in stage2_results['architecture_variants']:
        weighted_score = stage2_results['decision_matrix']['weighted_scores'][variant['name']]
        arch_table.add_row(
            variant['name'],
            variant['type'],
            f"{variant['performance_benchmarks']['latency_ms']}ms",
            f"{variant['performance_benchmarks']['throughput_rps']} rps",
            f"{weighted_score:.2f}"
        )
    
    console.print("\n", arch_table)
    
    # Stage 3: Component Design
    console.print("\n[cyan]Stage 3: Component Design & Technology Stack[/cyan]")
    stage3_results = await ada7.execute_stage_3(project)
    
    console.print(f"   ✓ Defined {len(stage3_results['component_breakdown']['components'])} components")
    console.print(f"   ✓ Selected technology stack with alternatives")
    console.print(f"   ✓ Defined {len(stage3_results['integration_patterns']['patterns'])} integration patterns")
    console.print(f"   ✓ Created development estimates")
    
    tech_stack = stage3_results['technology_stack']
    console.print(f"\n   🔧 Technology Stack:")
    console.print(f"      Backend: {tech_stack['backend']['primary']['framework']}")
    console.print(f"      Frontend: {tech_stack['frontend']['primary']['name']} {tech_stack['frontend']['primary']['version']}")
    console.print(f"      Database: {tech_stack['database']['primary']['name']} {tech_stack['database']['primary']['version']}")
    
    # Stage 4: Implementation Strategy
    console.print("\n[cyan]Stage 4: Implementation Strategy & Development Pipeline[/cyan]")
    stage4_results = await ada7.execute_stage_4(project)
    
    console.print(f"   ✓ Created phased development plan with MVP definition")
    console.print(f"   ✓ Setup development environment (Docker + local)")
    console.print(f"   ✓ Defined CI/CD pipeline with {len(stage4_results['ci_cd_pipeline']['ci']['steps'])} steps")
    console.print(f"   ✓ Created code templates")
    
    mvp = stage4_results['phased_plan']['mvp']
    console.print(f"\n   🎯 MVP Features: {', '.join(mvp['features'])}")
    console.print(f"   ⏱️  MVP Duration: {mvp['duration']}")
    
    # Stage 5: Testing Framework
    console.print("\n[cyan]Stage 5: Testing Framework & Quality Assurance[/cyan]")
    stage5_results = await ada7.execute_stage_5(project)
    
    strategy = stage5_results['testing_strategy']
    console.print(f"   ✓ Defined testing pyramid with 4 levels")
    console.print(f"   ✓ Unit test coverage target: {int(strategy['unit_tests']['coverage_target']*100)}%")
    console.print(f"   ✓ Created quality gates (code, security, performance)")
    console.print(f"   ✓ Defined failure response protocol")
    
    # Stage 6: Deployment
    console.print("\n[cyan]Stage 6: Deployment & Infrastructure Management[/cyan]")
    stage6_results = await ada7.execute_stage_6(project)
    
    console.print(f"   ✓ Defined 3 environments (dev, staging, production)")
    console.print(f"   ✓ Created Infrastructure as Code (Terraform + K8s)")
    console.print(f"   ✓ Implemented security (OAuth 2.0, encryption, network)")
    console.print(f"   ✓ Setup monitoring (Prometheus, ELK, APM)")
    
    # Stage 7: Maintenance
    console.print("\n[cyan]Stage 7: Maintenance & Continuous Evolution[/cyan]")
    stage7_results = await ada7.execute_stage_7(project)
    
    console.print(f"   ✓ Defined operational excellence metrics")
    console.print(f"   ✓ Created evolution roadmap (4 quarters)")
    console.print(f"   ✓ Setup knowledge management system")
    console.print(f"   ✓ Created incident response playbooks")
    
    # Project status
    console.print("\n[bold blue]4. Project Status Summary[/bold blue]")
    
    status = ada7.get_project_status(project.project_id)
    
    status_tree = Tree("📊 Project Status")
    status_tree.add(f"Name: {status['name']}")
    status_tree.add(f"Current Stage: {status['current_stage']}/7")
    status_tree.add(f"Stages Completed: {len(status['stages_completed'])}")
    
    constraints_node = status_tree.add("Constraints")
    for key, value in status['constraints'].items():
        constraints_node.add(f"{key}: {value}")
    
    console.print("\n", status_tree)
    
    # Knowledge base
    console.print("\n[bold blue]5. Knowledge Base Search[/bold blue]")
    
    # Search papers
    architecture_papers = ada7.search_academic_papers(
        keywords=["microservices", "architecture"],
        min_relevance=0.3
    )
    console.print(f"\n   📚 Found {len(architecture_papers)} papers on microservices architecture")
    
    if architecture_papers:
        console.print(f"   • {architecture_papers[0].format_citation()}")
    
    # Search repos
    llm_repos = ada7.search_github_repositories(
        keywords=["llm", "language"],
        min_stars=1000
    )
    console.print(f"\n   🔍 Found {len(llm_repos)} GitHub repos related to LLM")
    
    if llm_repos:
        console.print(f"   • {llm_repos[0].format_reference()}")
    
    # Decision matrix example
    console.print("\n[bold blue]6. Decision Matrix Example[/bold blue]")
    
    matrix = ada7.create_decision_matrix(
        alternatives=["Option A", "Option B", "Option C"],
        criteria=["cost", "performance", "scalability"],
        weights={"cost": 0.3, "performance": 0.4, "scalability": 0.3}
    )
    
    # Add some scores
    matrix.scores = {
        "Option A": {"cost": 8.0, "performance": 7.0, "scalability": 6.0},
        "Option B": {"cost": 6.0, "performance": 9.0, "scalability": 8.0},
        "Option C": {"cost": 7.0, "performance": 6.0, "scalability": 9.0}
    }
    
    weighted_scores = matrix.calculate_weighted_scores()
    best = matrix.get_best_alternative()
    
    console.print(f"\n   📊 Decision Matrix: {', '.join(matrix.alternatives)}")
    console.print(f"   🏆 Best Alternative: {best}")
    console.print(f"   💯 Scores: {', '.join([f'{k}: {v:.2f}' for k, v in weighted_scores.items()])}")
    
    # Summary
    console.print("\n[bold green]✅ ADA-7 Framework Demo Complete![/bold green]")
    console.print("\n📈 Framework Capabilities Demonstrated:")
    console.print("   • 7-stage evolutionary development methodology")
    console.print("   • Academic research integration (arXiv papers)")
    console.print("   • GitHub repository analysis and integration")
    console.print("   • Evidence-based decision making with quantified metrics")
    console.print("   • SMART requirements specification")
    console.print("   • Architecture validation with multiple alternatives")
    console.print("   • Comprehensive technology stack selection")
    console.print("   • Testing pyramid and quality gates")
    console.print("   • Production-ready deployment strategies")
    console.print("   • Continuous evolution and maintenance planning")
    
    console.print("\n[bold cyan]Integration with LLM Aggregator:[/bold cyan]")
    console.print("   • Uses meta-controller for intelligent model selection")
    console.print("   • Leverages research components for academic validation")
    console.print("   • Can use aggregator for AI-powered analysis")
    console.print("   • Integrates with ensemble system for multi-model decisions")


async def main():
    """Main entry point."""
    try:
        await demo_ada7_framework()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
