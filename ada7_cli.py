#!/usr/bin/env python3
"""
ADA-7 Command-Line Interface

Provides access to the Advanced Development Assistant (ADA-7) framework
for structured, evidence-based software development.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.ada7 import (
    ADA7Assistant,
    Stage,
    ConfidenceLevel,
    create_citation,
    create_evidence,
)

console = Console()


def cmd_new_project(args):
    """Start a new project with ADA-7"""
    console.print("\n[bold cyan]🎯 ADA-7: Starting New Project[/bold cyan]\n")
    
    ada = ADA7Assistant()
    
    project_name = args.name or console.input("[yellow]Project name:[/yellow] ")
    
    console.print(f"\n[green]✓[/green] Initialized ADA-7 project: [bold]{project_name}[/bold]")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("  1. Define user personas: [cyan]ada7 persona add[/cyan]")
    console.print("  2. Analyze competitors: [cyan]ada7 competitor add[/cyan]")
    console.print("  3. Design architecture: [cyan]ada7 architecture design[/cyan]")
    console.print("\nFor full workflow, run: [cyan]ada7 guide[/cyan]")


def cmd_guide(args):
    """Show ADA-7 framework guide"""
    guide_text = """
# 🎯 ADA-7 Framework Guide

## Quick Start

1. **Initialize Project**
   ```bash
   ada7 new --name "My Project"
   ```

2. **Stage 1: Requirements Analysis**
   ```bash
   ada7 persona add --name "Power User" --description "..."
   ada7 competitor add --name "LangChain" --url "github.com/..."
   ada7 gaps analyze
   ```

3. **Stage 2: Architecture Design**
   ```bash
   ada7 architecture design --options 3
   ada7 architecture decide
   ```

4. **Stage 3: Component Design**
   ```bash
   ada7 component add --name "Router" --points 13
   ada7 component estimate
   ```

5. **Stage 4: Implementation**
   ```bash
   ada7 mvp define
   ada7 sprint plan
   ```

6. **Stage 5: Testing**
   ```bash
   ada7 test strategy
   ada7 quality-gate add
   ```

7. **Stage 6: Deployment**
   ```bash
   ada7 deploy spec --env production
   ada7 sla define
   ```

8. **Stage 7: Maintenance**
   ```bash
   ada7 debt track
   ada7 roadmap plan
   ```

## Key Principles

✓ **Evidence-Based** - Back decisions with academic papers and industry data
✓ **Structured** - Follow 7 evolutionary stages
✓ **Practical** - Focus on real-world constraints
✓ **Quality** - Built-in testing and validation
✓ **Evolution** - Continuous improvement

## Documentation

For complete documentation, see: **ADA_7_FRAMEWORK.md**

For interactive demo: `python ada7_demo.py`
    """
    
    console.print(Panel(Markdown(guide_text), title="ADA-7 Guide", border_style="cyan"))


def cmd_demo(args):
    """Run the ADA-7 demo"""
    import subprocess
    
    console.print("\n[bold cyan]Running ADA-7 Demo...[/bold cyan]\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "ada7_demo.py"],
            cwd=Path(__file__).parent,
            capture_output=False
        )
        sys.exit(result.returncode)
    except Exception as e:
        console.print(f"[red]Error running demo: {e}[/red]")
        sys.exit(1)


def cmd_status(args):
    """Show project status"""
    console.print("\n[bold cyan]ADA-7 Project Status[/bold cyan]\n")
    
    ada = ADA7Assistant()
    summary = ada.get_project_summary()
    
    console.print(f"[yellow]Current Stage:[/yellow] {summary['current_stage']}")
    console.print(f"[yellow]Stages Completed:[/yellow] {summary['stages_completed']}/7")
    
    if summary['stages_completed'] == 0:
        console.print("\n[yellow]💡 Tip:[/yellow] Start with: [cyan]ada7 new[/cyan]")
    else:
        console.print(f"\n[green]✓[/green] Progress: {summary['stages_completed'] / 7 * 100:.1f}%")


def cmd_citation(args):
    """Create a citation"""
    console.print("\n[bold cyan]Create Citation[/bold cyan]\n")
    
    citation = create_citation(
        authors=args.authors,
        year=args.year,
        title=args.title,
        identifier=args.identifier,
        citation_type=args.type,
        relevance=args.relevance or ""
    )
    
    console.print(f"\n[green]✓ Citation created:[/green]")
    console.print(f"  {citation}")


def cmd_framework_info(args):
    """Show ADA-7 framework information"""
    info_text = """
# 🎯 ADA-7: Advanced Development Assistant

## Overview

ADA-7 is a specialized AI system that creates high-quality software applications
through structured, evidence-based development.

## 7 Evolutionary Stages

1. **Requirements Analysis** - User personas, competitive analysis, feature gaps
2. **Architecture Design** - 3 options with academic validation and decision matrix
3. **Component Design** - Technology selection, development estimates
4. **Implementation** - MVP definition, sprint planning, CI/CD setup
5. **Testing Framework** - Test pyramid, quality gates, failure protocols
6. **Deployment** - Infrastructure as code, security, monitoring
7. **Maintenance** - Technical debt tracking, evolution roadmap

## Evidence Requirements

Every major decision must have:
- 2+ academic papers (arXiv with citations)
- 3+ production examples (GitHub repos with metrics)
- Quantified performance data
- Risk assessment with mitigations

## Quality Standards

✓ Cross-validation with 3+ sources
✓ Production readiness testing
✓ Performance benchmarking
✓ Maintenance assessment
✓ Comprehensive documentation

## Getting Started

```bash
# Run interactive demo
ada7 demo

# Start new project
ada7 new --name "My Project"

# View full guide
ada7 guide
```

For complete documentation: **ADA_7_FRAMEWORK.md**
    """
    
    console.print(Panel(Markdown(info_text), title="ADA-7 Framework", border_style="green", box=box.DOUBLE))


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ADA-7: Advanced Development Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ada7 info              Show framework information
  ada7 demo              Run interactive demo
  ada7 guide             Show complete guide
  ada7 new               Start new project
  ada7 status            Show project status
  
For more information: ada7 --help
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show framework information")
    info_parser.set_defaults(func=cmd_framework_info)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.set_defaults(func=cmd_demo)
    
    # Guide command
    guide_parser = subparsers.add_parser("guide", help="Show complete guide")
    guide_parser.set_defaults(func=cmd_guide)
    
    # New project command
    new_parser = subparsers.add_parser("new", help="Start new project")
    new_parser.add_argument("--name", help="Project name")
    new_parser.set_defaults(func=cmd_new_project)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.set_defaults(func=cmd_status)
    
    # Citation command
    citation_parser = subparsers.add_parser("citation", help="Create a citation")
    citation_parser.add_argument("--authors", required=True, help="Authors (e.g., 'Chen et al.')")
    citation_parser.add_argument("--year", type=int, required=True, help="Publication year")
    citation_parser.add_argument("--title", required=True, help="Paper/Project title")
    citation_parser.add_argument("--identifier", required=True, help="arXiv ID, DOI, or GitHub URL")
    citation_parser.add_argument("--type", choices=["academic", "industry"], default="academic")
    citation_parser.add_argument("--relevance", help="Relevance description")
    citation_parser.set_defaults(func=cmd_citation)
    
    args = parser.parse_args()
    
    if not args.command:
        # Show info by default
        cmd_framework_info(args)
        console.print("\n[yellow]💡 Tip:[/yellow] Use [cyan]ada7 --help[/cyan] to see all commands")
        return
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
