"""
ADA-7 Framework Demo

Demonstrates the Advanced Development Assistant (ADA-7) framework
for structured, evidence-based software development.
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.markdown import Markdown
from rich import box

from src.core.ada7 import (
    ADA7Assistant,
    Stage,
    ConfidenceLevel,
    create_citation,
    create_evidence,
    Citation,
)

console = Console()


def print_header():
    """Print demo header"""
    header_text = """
    # 🎯 ADA-7: Advanced Development Assistant
    
    A specialized AI system for creating high-quality software
    through structured, evidence-based development.
    """
    console.print(Panel(Markdown(header_text), border_style="blue", box=box.DOUBLE))


def demo_stage_1_requirements():
    """Demonstrate Stage 1: Requirements Analysis"""
    console.print("\n[bold cyan]Stage 1: Requirements Analysis & Competitive Intelligence[/bold cyan]\n")
    
    ada = ADA7Assistant()
    
    # Create user personas
    console.print("[yellow]Creating User Personas...[/yellow]")
    
    persona1 = ada.create_user_persona(
        name="Power User - AI Researcher",
        description="PhD student experimenting with multiple LLM models",
        pain_points=[
            "High costs when testing various models",
            "Complex setup for multiple provider APIs",
            "No intelligent model selection",
            "Slow response times from single providers"
        ],
        success_metrics=[
            {"metric": "Cost per experiment", "target": "<$5", "measurement": "Monthly spend"},
            {"metric": "Setup time", "target": "<5 minutes", "measurement": "Time to first API call"},
            {"metric": "Response time", "target": "<2 seconds", "measurement": "P95 latency"}
        ]
    )
    
    persona2 = ada.create_user_persona(
        name="Production User - Startup CTO",
        description="Running LLM-powered features in production",
        pain_points=[
            "Vendor lock-in concerns",
            "Rate limiting issues",
            "No fallback when primary provider fails",
            "Unpredictable costs"
        ],
        success_metrics=[
            {"metric": "Uptime", "target": ">99.9%", "measurement": "Monthly availability"},
            {"metric": "Cost predictability", "target": "±10%", "measurement": "Monthly variance"},
            {"metric": "Latency", "target": "<100ms", "measurement": "P95"}
        ]
    )
    
    # Display personas
    persona_table = Table(title="User Personas", box=box.ROUNDED)
    persona_table.add_column("Persona", style="cyan")
    persona_table.add_column("Pain Points", style="yellow")
    persona_table.add_column("Success Metrics", style="green")
    
    for persona in [persona1, persona2]:
        persona_table.add_row(
            persona.name,
            "\n".join(f"• {p}" for p in persona.pain_points[:3]),
            "\n".join(f"• {m['metric']}: {m['target']}" for m in persona.success_metrics[:2])
        )
    
    console.print(persona_table)
    
    # Competitive analysis
    console.print("\n[yellow]Analyzing Competitors (10 similar applications)...[/yellow]")
    
    competitors = [
        ada.analyze_competitor("LangChain", "https://github.com/langchain-ai/langchain", 
                             ["Multi-provider", "Chaining", "Memory", "Agents"], 
                             is_open_source=True, stars=75000),
        ada.analyze_competitor("LlamaIndex", "https://github.com/run-llama/llama_index",
                             ["Data connectors", "Indexes", "Query engines"],
                             is_open_source=True, stars=25000),
        ada.analyze_competitor("OpenRouter", None,
                             ["50+ models", "Unified API", "Rate limiting"],
                             is_open_source=False),
        ada.analyze_competitor("Portkey", "https://github.com/Portkey-AI/gateway",
                             ["Multi-provider gateway", "Caching", "Load balancing"],
                             is_open_source=True, stars=3500),
        ada.analyze_competitor("LiteLLM", "https://github.com/BerriAI/litellm",
                             ["100+ LLMs", "Unified interface", "Load balancing"],
                             is_open_source=True, stars=8000),
        ada.analyze_competitor("AI Gateway (Cloudflare)", None,
                             ["Caching", "Rate limiting", "Analytics"],
                             is_open_source=False),
        ada.analyze_competitor("Helicone", "https://github.com/Helicone/helicone",
                             ["Observability", "Caching", "Rate limiting"],
                             is_open_source=True, stars=1200),
        ada.analyze_competitor("Humanloop", None,
                             ["Prompt management", "Evaluation", "Monitoring"],
                             is_open_source=False),
        ada.analyze_competitor("PromptLayer", "https://github.com/MagnivOrg/prompt-layer-library",
                             ["Request tracking", "Analytics", "Prompt versioning"],
                             is_open_source=True, stars=500),
        ada.analyze_competitor("BerriAI Proxy", "https://github.com/BerriAI/litellm-proxy",
                             ["Load balancing", "Fallbacks", "Rate limiting"],
                             is_open_source=True, stars=1500),
    ]
    
    comp_table = Table(title="Competitive Analysis (10 Applications)", box=box.ROUNDED)
    comp_table.add_column("Application", style="cyan")
    comp_table.add_column("Type", style="yellow")
    comp_table.add_column("Stars", style="magenta")
    comp_table.add_column("Key Features", style="green")
    
    for comp in competitors:
        comp_table.add_row(
            comp.name,
            "Open Source" if comp.is_open_source else "Commercial",
            str(comp.stars) if comp.stars else "N/A",
            ", ".join(comp.features[:3])
        )
    
    console.print(comp_table)
    
    # Feature gap analysis
    console.print("\n[yellow]Identifying Feature Gaps...[/yellow]")
    
    user_needs = [
        "Intelligent model routing based on task complexity",
        "Cost optimization with cascade routing",
        "Research-backed model selection",
        "Multi-model ensemble responses",
        "Automatic credential rotation"
    ]
    
    gaps = ada.identify_feature_gaps(competitors, user_needs)
    
    gap_table = Table(title="Feature Gap Analysis", box=box.ROUNDED)
    gap_table.add_column("User Need", style="cyan")
    gap_table.add_column("Priority", style="yellow")
    gap_table.add_column("Evidence", style="green")
    
    for gap in gaps:
        gap_table.add_row(gap["need"], gap["priority"], gap["evidence"])
    
    console.print(gap_table)
    
    return ada


def demo_stage_2_architecture(ada: ADA7Assistant):
    """Demonstrate Stage 2: Architecture Design"""
    console.print("\n[bold cyan]Stage 2: Architecture Design & Academic Validation[/bold cyan]\n")
    
    # Create architecture options
    console.print("[yellow]Designing 3 Architecture Options...[/yellow]")
    
    # Option 1: Monolithic
    evidence_monolithic = create_evidence(
        academic_papers=[
            create_citation("Chen et al.", 2023, "Monolithic vs Microservices Performance", 
                          "arXiv:2301.12345",
                          relevance="Shows monolithic has 20% lower latency for simple systems"),
            create_citation("Smith et al.", 2024, "System Architecture Trade-offs",
                          "arXiv:2401.54321",
                          relevance="Demonstrates faster development time for small teams")
        ],
        production_examples=[
            {"repo": "fastapi/fastapi", "stars": 65000, "pattern": "Single-process async server",
             "lesson": "Excellent for startups, scales to 10k req/s"},
            {"repo": "django/django", "stars": 70000, "pattern": "Monolithic MVC",
             "lesson": "Proven reliability, extensive ecosystem"},
            {"repo": "flask/flask", "stars": 65000, "pattern": "Minimal monolith",
             "lesson": "Simple to understand and deploy"}
        ],
        confidence=ConfidenceLevel.HIGH
    )
    
    opt1 = ada.create_architecture_option(
        name="Monolithic Architecture",
        description="Single application with all components in one codebase",
        components=["API Gateway", "Router", "Provider Manager", "Account Manager", "Cache"],
        evidence=evidence_monolithic
    )
    opt1.pros = ["Fast development", "Simple deployment", "Easy debugging", "Low overhead"]
    opt1.cons = ["Scaling limitations", "Deployment risk", "Technology coupling"]
    opt1.performance_metrics = {"latency_p95": "50ms", "throughput": "5000 req/s"}
    
    # Option 2: Microservices
    evidence_microservices = create_evidence(
        academic_papers=[
            create_citation("Liu et al.", 2024, "Microservices in Production",
                          "arXiv:2402.11111",
                          relevance="Shows 3x better scalability but 50% more complexity"),
            create_citation("Wang et al.", 2023, "Service Mesh Performance",
                          "arXiv:2308.98765",
                          relevance="Demonstrates <10ms service mesh overhead")
        ],
        production_examples=[
            {"repo": "netflix/conductor", "stars": 12000, "pattern": "Workflow microservices",
             "lesson": "Scales to millions but requires DevOps expertise"},
            {"repo": "uber/cadence", "stars": 7000, "pattern": "Distributed orchestration",
             "lesson": "Complex but highly resilient"},
            {"repo": "temporal-io/temporal", "stars": 8000, "pattern": "Durable execution",
             "lesson": "Excellent fault tolerance"}
        ],
        confidence=ConfidenceLevel.MEDIUM
    )
    
    opt2 = ada.create_architecture_option(
        name="Microservices Architecture",
        description="Multiple independent services communicating via APIs",
        components=["API Gateway Service", "Router Service", "Provider Service", 
                   "Account Service", "Cache Service", "Meta-Controller Service"],
        evidence=evidence_microservices
    )
    opt2.pros = ["Independent scaling", "Technology diversity", "Fault isolation"]
    opt2.cons = ["Complex deployment", "Network overhead", "Distributed debugging"]
    opt2.performance_metrics = {"latency_p95": "70ms", "throughput": "15000 req/s"}
    
    # Option 3: Hybrid/Modular
    evidence_hybrid = create_evidence(
        academic_papers=[
            create_citation("Zhang et al.", 2023, "Modular System Design",
                          "arXiv:2308.99999",
                          relevance="Shows 40% reduction in coupling vs monolith"),
            create_citation("Kumar et al.", 2024, "Plugin Architecture Performance",
                          "arXiv:2402.77777",
                          relevance="Demonstrates <5ms plugin overhead")
        ],
        production_examples=[
            {"repo": "fastapi/fastapi", "stars": 65000, "pattern": "Modular middleware",
             "lesson": "Clean boundaries without microservice complexity"},
            {"repo": "starlette/starlette", "stars": 8500, "pattern": "ASGI components",
             "lesson": "Highly composable and performant"},
            {"repo": "encode/httpx", "stars": 11000, "pattern": "Pluggable transports",
             "lesson": "Extension points enable flexibility"}
        ],
        confidence=ConfidenceLevel.HIGH
    )
    
    opt3 = ada.create_architecture_option(
        name="Hybrid/Modular Architecture",
        description="Monolithic deployment with modular component design",
        components=["Core Module", "Provider Plugins", "Router Module", 
                   "Account Module", "Cache Module", "Meta-Controller Module"],
        evidence=evidence_hybrid
    )
    opt3.pros = ["Clean boundaries", "Easy testing", "Simple deployment", "Extensible"]
    opt3.cons = ["Still single deployment", "Shared resources"]
    opt3.performance_metrics = {"latency_p95": "55ms", "throughput": "8000 req/s"}
    
    # Display options
    arch_tree = Tree("🏗️ Architecture Options", guide_style="bold cyan")
    
    for opt in [opt1, opt2, opt3]:
        opt_node = arch_tree.add(f"[bold]{opt.name}[/bold]")
        opt_node.add(f"[green]Pros:[/green] {', '.join(opt.pros[:3])}")
        opt_node.add(f"[red]Cons:[/red] {', '.join(opt.cons[:2])}")
        metrics_str = f"Latency: {opt.performance_metrics['latency_p95']}, "
        metrics_str += f"Throughput: {opt.performance_metrics['throughput']}"
        opt_node.add(f"[yellow]Performance:[/yellow] {metrics_str}")
        opt_node.add(f"[blue]Evidence:[/blue] {len(opt.evidence.citations)} citations, "
                    f"{len(opt.evidence.production_examples)} examples")
    
    console.print(arch_tree)
    
    # Decision matrix
    console.print("\n[yellow]Creating Decision Matrix...[/yellow]")
    
    criteria = {
        "scalability": 0.20,
        "maintainability": 0.20,
        "performance": 0.15,
        "cost": 0.15,
        "team_expertise": 0.15,
        "time_to_market": 0.15
    }
    
    matrix = ada.create_decision_matrix(criteria, [opt1, opt2, opt3])
    
    # Add scores
    matrix.scores = {
        "Monolithic Architecture": {
            "scalability": 6, "maintainability": 7, "performance": 8,
            "cost": 9, "team_expertise": 9, "time_to_market": 9
        },
        "Microservices Architecture": {
            "scalability": 9, "maintainability": 6, "performance": 7,
            "cost": 5, "team_expertise": 5, "time_to_market": 4
        },
        "Hybrid/Modular Architecture": {
            "scalability": 8, "maintainability": 8, "performance": 8,
            "cost": 7, "team_expertise": 7, "time_to_market": 7
        }
    }
    
    # Display matrix
    matrix_table = Table(title="Decision Matrix (Weighted Scoring)", box=box.ROUNDED)
    matrix_table.add_column("Criterion (Weight)", style="cyan")
    matrix_table.add_column("Monolithic", style="yellow")
    matrix_table.add_column("Microservices", style="magenta")
    matrix_table.add_column("Hybrid/Modular", style="green")
    
    for criterion, weight in criteria.items():
        matrix_table.add_row(
            f"{criterion.title()} ({weight:.2f})",
            str(matrix.scores["Monolithic Architecture"][criterion]),
            str(matrix.scores["Microservices Architecture"][criterion]),
            str(matrix.scores["Hybrid/Modular Architecture"][criterion])
        )
    
    weighted_scores = matrix.calculate_weighted_scores()
    matrix_table.add_row(
        "[bold]Total Score[/bold]",
        f"[bold]{weighted_scores['Monolithic Architecture']:.2f}[/bold]",
        f"[bold]{weighted_scores['Microservices Architecture']:.2f}[/bold]",
        f"[bold]{weighted_scores['Hybrid/Modular Architecture']:.2f}[/bold]"
    )
    
    console.print(matrix_table)
    
    # Recommendation
    recommended, score = matrix.get_recommendation()
    console.print(f"\n[bold green]✓ Recommended:[/bold green] {recommended} (Score: {score:.2f})")


def demo_stage_3_components(ada: ADA7Assistant):
    """Demonstrate Stage 3: Component Design"""
    console.print("\n[bold cyan]Stage 3: Component Design & Technology Stack[/bold cyan]\n")
    
    console.print("[yellow]Defining Core Components...[/yellow]")
    
    # Define components
    components = [
        ada.define_component(
            name="IntelligentRouter",
            responsibility="Analyze requests and select optimal provider/model",
            technology={
                "framework": "FastAPI",
                "version": "0.104.1",
                "language": "Python 3.11",
                "justification": "Async support, automatic OpenAPI docs, high performance"
            },
            story_points=13
        ),
        ada.define_component(
            name="ProviderManager",
            responsibility="Manage connections to 25+ LLM providers",
            technology={
                "framework": "httpx",
                "version": "0.25.2",
                "language": "Python 3.11",
                "justification": "Async HTTP client, connection pooling, HTTP/2 support"
            },
            story_points=21
        ),
        ada.define_component(
            name="AccountManager",
            responsibility="Securely manage and rotate API credentials",
            technology={
                "framework": "cryptography",
                "version": "41.0.8",
                "language": "Python 3.11",
                "justification": "Industry-standard encryption, key management"
            },
            story_points=8
        ),
        ada.define_component(
            name="MetaController",
            responsibility="ML-based model selection using research papers",
            technology={
                "framework": "PyTorch",
                "version": "2.0.0+",
                "language": "Python 3.11",
                "justification": "Research implementation, gradient-based optimization"
            },
            story_points=21
        ),
        ada.define_component(
            name="CacheLayer",
            responsibility="Cache responses to reduce costs and latency",
            technology={
                "framework": "Redis",
                "version": "5.0.1",
                "language": "Python 3.11",
                "justification": "Fast in-memory storage, TTL support, pub/sub"
            },
            story_points=5
        )
    ]
    
    # Display components
    comp_table = Table(title="Component Specifications", box=box.ROUNDED)
    comp_table.add_column("Component", style="cyan")
    comp_table.add_column("Responsibility", style="yellow")
    comp_table.add_column("Technology", style="green")
    comp_table.add_column("Story Points", style="magenta")
    
    for comp in components:
        tech_str = f"{comp.technology['framework']} {comp.technology['version']}"
        comp_table.add_row(
            comp.name,
            comp.responsibility[:50] + "..." if len(comp.responsibility) > 50 else comp.responsibility,
            tech_str,
            str(comp.story_points)
        )
    
    console.print(comp_table)
    
    # Development estimate
    console.print("\n[yellow]Calculating Development Estimates...[/yellow]")
    
    estimate = ada.estimate_development_time(components, team_velocity=40)
    
    estimate_panel = Panel(
        f"""[bold]Total Story Points:[/bold] {estimate['total_story_points']}
[bold]Team Velocity:[/bold] {estimate['team_velocity']} points/sprint
[bold]Estimated Sprints:[/bold] {estimate['estimated_sprints']}
[bold]Estimated Weeks:[/bold] {estimate['estimated_weeks']}
[bold]Confidence:[/bold] {estimate['confidence']}""",
        title="Development Time Estimate",
        border_style="green"
    )
    
    console.print(estimate_panel)


def demo_stage_4_implementation(ada: ADA7Assistant):
    """Demonstrate Stage 4: Implementation Strategy"""
    console.print("\n[bold cyan]Stage 4: Implementation Strategy & Development Pipeline[/bold cyan]\n")
    
    # Define MVP
    console.print("[yellow]Defining MVP Scope...[/yellow]")
    
    mvp = ada.define_mvp(
        core_features=[
            "Multi-provider routing (5+ providers)",
            "Basic account management",
            "Simple fallback mechanism",
            "OpenAI-compatible API",
            "Rate limit handling"
        ],
        success_metrics=[
            {"metric": "API response time", "target": "<100ms P95", "measurement": "Prometheus"},
            {"metric": "Error rate", "target": "<1%", "measurement": "Error tracking"},
            {"metric": "Provider coverage", "target": "5+ free providers", "measurement": "Config"}
        ],
        launch_criteria=[
            "All MVP features tested with >80% coverage",
            "Security audit passed",
            "Load testing completed (1000 req/s)",
            "Documentation complete",
            "Deployment automation working"
        ]
    )
    
    mvp_table = Table(title="MVP Definition", box=box.ROUNDED)
    mvp_table.add_column("Category", style="cyan")
    mvp_table.add_column("Items", style="green")
    
    mvp_table.add_row(
        "Core Features",
        "\n".join(f"• {f}" for f in mvp.core_features)
    )
    mvp_table.add_row(
        "Success Metrics",
        "\n".join(f"• {m['metric']}: {m['target']}" for m in mvp.success_metrics)
    )
    mvp_table.add_row(
        "Launch Criteria",
        "\n".join(f"• {c}" for c in mvp.launch_criteria[:3])
    )
    
    console.print(mvp_table)
    
    # Sprint planning
    console.print("\n[yellow]Planning Development Sprints...[/yellow]")
    
    sprints = [
        ada.plan_sprint(
            sprint_number=1,
            theme="Core Infrastructure",
            features=[
                {"name": "Provider abstraction layer", "story_points": 13},
                {"name": "Basic router implementation", "story_points": 21},
                {"name": "Error handling framework", "story_points": 8}
            ]
        ),
        ada.plan_sprint(
            sprint_number=2,
            theme="Account & Security",
            features=[
                {"name": "Credential encryption", "story_points": 8},
                {"name": "Account rotation logic", "story_points": 5},
                {"name": "Rate limit tracking", "story_points": 8}
            ]
        )
    ]
    
    sprint_table = Table(title="Sprint Plan", box=box.ROUNDED)
    sprint_table.add_column("Sprint", style="cyan")
    sprint_table.add_column("Theme", style="yellow")
    sprint_table.add_column("Story Points", style="magenta")
    sprint_table.add_column("Features", style="green")
    
    for sprint in sprints:
        features_str = "\n".join(f"• {f['name']} ({f['story_points']} pts)" 
                                for f in sprint.features)
        sprint_table.add_row(
            f"Sprint {sprint.sprint_number}",
            sprint.theme,
            str(sprint.story_points),
            features_str
        )
    
    console.print(sprint_table)


def demo_stage_5_testing(ada: ADA7Assistant):
    """Demonstrate Stage 5: Testing Framework"""
    console.print("\n[bold cyan]Stage 5: Testing Framework & Quality Assurance[/bold cyan]\n")
    
    console.print("[yellow]Creating Test Strategy...[/yellow]")
    
    strategy = ada.create_test_strategy(
        target_coverage=0.80,
        integration_tests=[
            "Provider API integration",
            "Database operations",
            "Cache layer",
            "Account management"
        ],
        e2e_scenarios=[
            "User makes successful request",
            "Primary provider fails, fallback succeeds",
            "Rate limit hit, automatic retry",
            "Invalid credentials handled gracefully"
        ]
    )
    
    # Testing pyramid visualization
    console.print("\n[bold]Testing Strategy Pyramid:[/bold]")
    pyramid = """
           /\\
          /E2E\\      10% - User journey tests (4 scenarios)
         /------\\
        /  INT  \\    20% - API contract, DB integration (4 test suites)
       /----------\\
      /   UNIT    \\ 70% - Function/class level (>80% coverage)
     /--------------\\
    """
    console.print(pyramid, style="cyan")
    
    # Quality gates
    console.print("[yellow]Defining Quality Gates...[/yellow]")
    
    gates = [
        ada.define_quality_gate("Code Coverage", "≥80%", blocking=True),
        ada.define_quality_gate("Security Scan", "No High/Critical issues", blocking=True),
        ada.define_quality_gate("Performance Regression", "±10% from baseline", blocking=True),
        ada.define_quality_gate("Code Complexity", "Cyclomatic complexity ≤10", blocking=False)
    ]
    
    gates_table = Table(title="Quality Gates", box=box.ROUNDED)
    gates_table.add_column("Gate", style="cyan")
    gates_table.add_column("Threshold", style="yellow")
    gates_table.add_column("Blocking", style="magenta")
    
    for gate in gates:
        blocking_str = "✓ Yes" if gate["blocking"] else "○ No"
        gates_table.add_row(gate["name"], str(gate["threshold"]), blocking_str)
    
    console.print(gates_table)


def demo_stage_6_deployment(ada: ADA7Assistant):
    """Demonstrate Stage 6: Deployment"""
    console.print("\n[bold cyan]Stage 6: Deployment & Infrastructure Management[/bold cyan]\n")
    
    console.print("[yellow]Creating Deployment Specifications...[/yellow]")
    
    # Production deployment
    prod_spec = ada.create_deployment_spec(
        environment="production",
        provider="AWS",
        compute_config={
            "type": "ECS Fargate",
            "cpu": "2 vCPU",
            "memory": "4 GB",
            "min_instances": 2,
            "max_instances": 10
        }
    )
    prod_spec.security_config = {
        "authentication": "OAuth 2.0",
        "encryption_at_rest": "AES-256",
        "encryption_in_transit": "TLS 1.3",
        "network": "VPC with private subnets"
    }
    prod_spec.monitoring_config = {
        "metrics": "Prometheus + Grafana",
        "logs": "CloudWatch Logs",
        "alerting": "PagerDuty"
    }
    prod_spec.auto_scaling = {
        "metric": "CPU Utilization",
        "threshold": "70%",
        "scale_out_cooldown": "300s"
    }
    
    deploy_table = Table(title="Production Deployment Specification", box=box.ROUNDED)
    deploy_table.add_column("Category", style="cyan")
    deploy_table.add_column("Configuration", style="green")
    
    deploy_table.add_row(
        "Compute",
        f"{prod_spec.compute_config['type']}\n"
        f"{prod_spec.compute_config['cpu']}, {prod_spec.compute_config['memory']}\n"
        f"Instances: {prod_spec.compute_config['min_instances']}-{prod_spec.compute_config['max_instances']}"
    )
    deploy_table.add_row(
        "Security",
        f"Auth: {prod_spec.security_config['authentication']}\n"
        f"Encryption: {prod_spec.security_config['encryption_at_rest']} (rest), "
        f"{prod_spec.security_config['encryption_in_transit']} (transit)"
    )
    deploy_table.add_row(
        "Monitoring",
        f"Metrics: {prod_spec.monitoring_config['metrics']}\n"
        f"Logs: {prod_spec.monitoring_config['logs']}\n"
        f"Alerts: {prod_spec.monitoring_config['alerting']}"
    )
    
    console.print(deploy_table)
    
    # Define SLA
    console.print("\n[yellow]Defining Service Level Agreement...[/yellow]")
    
    sla = ada.define_sla(
        uptime_target=0.999,
        latency_p95_ms=100,
        error_rate_max=0.01
    )
    
    sla_panel = Panel(
        f"""[bold]Uptime Target:[/bold] {sla['uptime_target']*100}% (99.9%)
[bold]Monthly Downtime:[/bold] <{sla['monthly_downtime_max_minutes']:.1f} minutes
[bold]Latency P95:[/bold] <{sla['latency_p95_ms']}ms
[bold]Error Rate:[/bold] <{sla['error_rate_max']*100}%
[bold]Measurement Window:[/bold] {sla['measurement_window']}""",
        title="Service Level Agreement (SLA)",
        border_style="green"
    )
    
    console.print(sla_panel)


def demo_stage_7_maintenance(ada: ADA7Assistant):
    """Demonstrate Stage 7: Maintenance"""
    console.print("\n[bold cyan]Stage 7: Maintenance & Continuous Evolution[/bold cyan]\n")
    
    # Technical debt
    console.print("[yellow]Tracking Technical Debt...[/yellow]")
    
    debt_items = [
        ada.track_technical_debt(
            "Legacy provider adapter needs refactoring",
            effort=13,
            impact="High"
        ),
        ada.track_technical_debt(
            "Improve error handling in router",
            effort=5,
            impact="Medium"
        ),
        ada.track_technical_debt(
            "Add comprehensive logging",
            effort=8,
            impact="Medium"
        )
    ]
    
    debt_table = Table(title="Technical Debt Register", box=box.ROUNDED)
    debt_table.add_column("Item", style="cyan")
    debt_table.add_column("Effort (SP)", style="yellow")
    debt_table.add_column("Impact", style="magenta")
    
    for debt in debt_items:
        debt_table.add_row(debt.description, str(debt.effort_story_points), debt.impact)
    
    console.print(debt_table)
    
    # Evolution roadmap
    console.print("\n[yellow]Creating Evolution Roadmap...[/yellow]")
    
    roadmap = ada.create_evolution_roadmap({
        "Q1 2024": [
            "Multi-region deployment",
            "Advanced caching layer",
            "GraphQL API support"
        ],
        "Q2 2024": [
            "Microservices migration (Phase 1)",
            "ML model updates",
            "Enhanced monitoring dashboard"
        ],
        "Q3 2024": [
            "Multi-modal support (images)",
            "Real-time streaming improvements",
            "Cost optimization v2"
        ]
    })
    
    roadmap_tree = Tree("📅 Evolution Roadmap", guide_style="bold cyan")
    
    for quarter, initiatives in roadmap.quarters.items():
        quarter_node = roadmap_tree.add(f"[bold]{quarter}[/bold]")
        for initiative in initiatives:
            quarter_node.add(f"• {initiative}")
    
    console.print(roadmap_tree)


def demo_decision_framework():
    """Demonstrate evidence-based decision framework"""
    console.print("\n[bold cyan]ADA-7 Decision Framework Example[/bold cyan]\n")
    console.print("[yellow]Decision: Database Selection for User Management[/yellow]\n")
    
    ada = ADA7Assistant()
    
    # Create evidence
    evidence = create_evidence(
        academic_papers=[
            create_citation("Cooper et al.", 2023, "NoSQL Performance Comparison",
                          "arXiv:2303.54321",
                          relevance="DynamoDB: 8ms P95, MongoDB: 12ms P95, PostgreSQL: 15ms P95"),
            create_citation("Lee et al.", 2024, "Database TCO Analysis",
                          "arXiv:2401.98765",
                          relevance="Cost comparison: DynamoDB $1200/mo, MongoDB $800/mo, PostgreSQL $600/mo")
        ],
        production_examples=[
            {"repo": "netflix/conductor", "stars": 12000, "choice": "DynamoDB",
             "lesson": "Scales to millions but watch costs"},
            {"repo": "strapi/strapi", "stars": 15000, "choice": "PostgreSQL",
             "lesson": "Reliable, easier operations"},
            {"repo": "parse-server/parse-server", "stars": 20000, "choice": "MongoDB",
             "lesson": "Flexible but needs tuning"}
        ],
        confidence=ConfidenceLevel.HIGH
    )
    
    # Define options
    options = [
        {
            "name": "PostgreSQL",
            "scalability_score": 7,
            "cost_score": 9,
            "team_expertise_score": 9,
            "ops_complexity_score": 8
        },
        {
            "name": "MongoDB",
            "scalability_score": 8,
            "cost_score": 7,
            "team_expertise_score": 6,
            "ops_complexity_score": 6
        },
        {
            "name": "DynamoDB",
            "scalability_score": 10,
            "cost_score": 5,
            "team_expertise_score": 5,
            "ops_complexity_score": 9
        }
    ]
    
    # Make decision
    decision = ada.make_evidence_based_decision(
        decision_name="Database for User Management",
        options=options,
        evidence=evidence,
        criteria={
            "scalability": 0.25,
            "cost": 0.30,
            "team_expertise": 0.25,
            "ops_complexity": 0.20
        }
    )
    
    # Display decision
    decision_table = Table(title="Decision Analysis", box=box.ROUNDED)
    decision_table.add_column("Option", style="cyan")
    decision_table.add_column("Score", style="yellow")
    decision_table.add_column("Rank", style="magenta")
    
    scores = decision["decision_matrix"]["scores"]
    sorted_options = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (option, score) in enumerate(sorted_options, 1):
        if rank == 1:
            decision_table.add_row(
                f"[bold green]{option}[/bold green]",
                f"[bold green]{score:.2f}[/bold green]",
                f"[bold green]#{rank}[/bold green]"
            )
        else:
            decision_table.add_row(option, f"{score:.2f}", f"#{rank}")
    
    console.print(decision_table)
    
    console.print(f"\n[bold green]✓ Recommended:[/bold green] {decision['recommended_option']}")
    console.print(f"[bold]Evidence Quality:[/bold] {decision['evidence']['academic_citations']} "
                 f"academic papers, {decision['evidence']['production_examples']} production examples")
    console.print(f"[bold]Confidence:[/bold] {decision['evidence']['confidence'].upper()}")


async def main():
    """Main demo function"""
    print_header()
    
    console.print("\n[bold yellow]Demonstrating all 7 stages of the ADA-7 framework...[/bold yellow]\n")
    
    # Run through all stages
    ada = demo_stage_1_requirements()
    demo_stage_2_architecture(ada)
    demo_stage_3_components(ada)
    demo_stage_4_implementation(ada)
    demo_stage_5_testing(ada)
    demo_stage_6_deployment(ada)
    demo_stage_7_maintenance(ada)
    
    # Demonstrate decision framework
    demo_decision_framework()
    
    # Summary
    console.print("\n" + "="*80 + "\n")
    summary_text = """
# ✨ ADA-7 Framework Summary

The Advanced Development Assistant (ADA-7) provides a comprehensive,
structured approach to software development with:

✓ **Evidence-Based Decisions** - Academic research + industry practice
✓ **7 Evolutionary Stages** - From requirements to maintenance
✓ **Quality Assurance** - Testing, validation, monitoring
✓ **Practical Focus** - Real-world constraints and trade-offs
✓ **Continuous Evolution** - Built-in improvement mechanisms

For full documentation, see: **ADA_7_FRAMEWORK.md**
    """
    console.print(Panel(Markdown(summary_text), border_style="green", box=box.DOUBLE))


if __name__ == "__main__":
    asyncio.run(main())
