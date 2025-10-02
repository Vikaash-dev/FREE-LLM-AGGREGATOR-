# Project File Structure

This document describes the organization of the FREE-LLM-AGGREGATOR repository.

## Directory Structure

```
FREE-LLM-AGGREGATOR-/
│
├── 📁 src/                          # Source code
│   ├── __init__.py
│   ├── models.py                    # Data models and schemas
│   │
│   ├── 📁 api/                      # API server
│   │   ├── __init__.py
│   │   └── server.py                # FastAPI server implementation
│   │
│   ├── 📁 config/                   # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py              # Application settings
│   │   └── logging_config.py        # Logging configuration
│   │
│   ├── 📁 core/                     # Core business logic
│   │   ├── __init__.py
│   │   ├── aggregator.py            # LLM API aggregator
│   │   ├── router.py                # Intelligent routing
│   │   ├── account_manager.py       # Credential management
│   │   ├── rate_limiter.py          # Rate limiting
│   │   ├── auto_updater.py          # Auto-updater system
│   │   ├── browser_monitor.py       # Browser automation
│   │   ├── meta_controller.py       # Meta-controller with external memory
│   │   ├── ensemble_system.py       # Multi-model response fusion
│   │   ├── planner.py               # Planning system
│   │   ├── reasoner.py              # Reasoning engine
│   │   ├── researcher.py            # Research system
│   │   ├── state_tracker.py         # State tracking
│   │   ├── code_generator.py        # Code generation
│   │   ├── planning_structures.py   # Planning data structures
│   │   ├── generation_structures.py # Generation data structures
│   │   └── research_structures.py   # Research data structures
│   │
│   └── 📁 providers/                # Provider implementations
│       ├── __init__.py
│       ├── base.py                  # Base provider interface
│       ├── openrouter.py            # OpenRouter integration
│       ├── groq.py                  # Groq integration
│       └── cerebras.py              # Cerebras integration
│
├── 📁 tests/                        # Test suite
│   ├── __init__.py
│   ├── test_aggregator.py           # Aggregator tests
│   │
│   └── 📁 core/                     # Core component tests
│       ├── __init__.py
│       ├── test_planner.py
│       ├── test_planning_structures.py
│       ├── test_reasoner.py
│       └── test_state_tracker.py
│
├── 📁 config/                       # Configuration files
│   ├── providers.yaml               # Provider configurations
│   └── auto_update.yaml             # Auto-update settings
│
├── 📁 docs/                         # Documentation
│   ├── README.md                    # Documentation index
│   ├── USAGE.md                     # Usage guide
│   ├── DEPLOYMENT_GUIDE.md          # Deployment instructions
│   ├── PROJECT_SUMMARY.md           # Project overview
│   ├── AUTO_UPDATER_DOCUMENTATION.md
│   ├── EXPERIMENTAL_FEATURES.md
│   ├── RESEARCH_ENHANCEMENTS.md
│   ├── research/                    # Research papers and analysis
│   │   └── arxiv_analysis.md
│   └── ... (other documentation files)
│
├── 📁 examples/                     # Example scripts and demos
│   ├── README.md                    # Examples index
│   ├── demo.py                      # Basic demo
│   ├── enhanced_demo.py             # Enhanced demo with research
│   ├── experimental_demo.py         # Experimental features demo
│   ├── auto_updater_demo.py         # Auto-updater demo
│   ├── ai_scientist_openhands.py    # AI-Scientist system
│   ├── experimental_optimizer.py    # Experimental optimizer
│   ├── recursive_optimizer.py       # Recursive optimizer
│   ├── openhands_improver.py        # OpenHands improver
│   └── run_planning_simulation.py   # Planning simulation
│
├── 📁 scripts/                      # Utility scripts
│   ├── README.md                    # Scripts index
│   ├── upload_to_github.py          # GitHub upload utility
│   └── test_auto_updater_integration.py
│
├── 📁 archive/                      # Archived/historical files
│   ├── README.md
│   ├── openhandsintegratorv1.py
│   ├── openhandsintegratorv2.py
│   └── openhandsintegratorv3.py
│
├── 📄 main.py                       # Main server entry point
├── 📄 cli.py                        # Command-line interface
├── 📄 web_ui.py                     # Web user interface (Streamlit)
├── 📄 setup.py                      # Interactive setup script
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # Main project README
├── 📄 Dockerfile                    # Docker configuration
├── 📄 docker-compose.yml            # Docker Compose setup
├── 📄 pytest.ini                    # Pytest configuration
├── 📄 .env.example                  # Example environment variables
└── 📄 .gitignore                    # Git ignore rules

```

## Key Entry Points

### For Users
- **main.py**: Start the API server
  ```bash
  python main.py
  ```

- **cli.py**: Command-line interface for interactive use
  ```bash
  python cli.py chat
  ```

- **web_ui.py**: Web-based user interface
  ```bash
  streamlit run web_ui.py
  ```

- **setup.py**: Interactive setup for credentials and configuration
  ```bash
  python setup.py configure
  ```

### For Developers
- **src/**: All source code organized by component
- **tests/**: Comprehensive test suite
- **examples/**: Example implementations and demos
- **docs/**: Complete documentation

### For Deployment
- **Dockerfile**: Container definition
- **docker-compose.yml**: Multi-container deployment
- **requirements.txt**: Python dependencies
- **config/**: YAML configuration files

## Architecture Principles

1. **Separation of Concerns**: Each directory has a clear purpose
2. **Modular Design**: Components are independent and reusable
3. **Easy Navigation**: Clear hierarchy and naming conventions
4. **Documentation First**: Each directory includes a README
5. **Clean Root**: Only essential entry points in root directory

## Adding New Components

### New Provider
1. Create provider class in `src/providers/`
2. Inherit from `src/providers/base.py`
3. Add configuration to `config/providers.yaml`

### New Core Feature
1. Add implementation to `src/core/`
2. Write tests in `tests/core/`
3. Update documentation in `docs/`

### New Example
1. Add script to `examples/`
2. Update `examples/README.md`

## Configuration Files

- **config/providers.yaml**: Provider-specific settings
- **config/auto_update.yaml**: Auto-updater configuration
- **.env**: Environment variables (create from .env.example)

## Database Files

- **model_memory.db**: SQLite database for model metadata (auto-created)

## Build Artifacts (Git-ignored)

- `__pycache__/`: Python bytecode
- `*.pyc`, `*.pyo`, `*.pyd`: Compiled Python files
- `.pytest_cache/`: Pytest cache
- `*.log`: Log files
- `.venv/`, `venv/`: Virtual environments
