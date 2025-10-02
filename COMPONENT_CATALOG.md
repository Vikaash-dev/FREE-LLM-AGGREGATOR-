# Component Catalog: FREE-LLM-AGGREGATOR

## Core Orchestration Components

### 1. LLMAggregator (`src/core/aggregator.py`)
**Responsibility**: Main orchestration class that coordinates provider selection, routing, and request handling  
**Key Features**:
- Provider pool management
- Circuit breaker pattern
- Retry logic with exponential backoff
- Integration with meta-controller and ensemble system
- Auto-updater integration

**Dependencies**: Router, AccountManager, RateLimiter, MetaController, EnsembleSystem, AutoUpdater  
**Evaluation Priority**: HIGH

### 2. ProviderRouter (`src/core/router.py`)
**Responsibility**: Intelligent provider routing and selection logic  
**Key Features**:
- Rule-based routing (code generation, reasoning, general text)
- Provider preference ordering
- Fallback chain management
- Request-rule matching

**Dependencies**: ProviderConfig, RoutingRule, ModelCapability  
**Evaluation Priority**: HIGH

### 3. MetaModelController (`src/core/meta_controller.py`)
**Responsibility**: Intelligent model selection based on task complexity  
**Key Features**:
- Task complexity analysis
- Model capability profiling
- Cost-aware routing (FrugalGPT pattern)
- Learning from historical performance
- Optional PyTorch-based routing model

**Dependencies**: PyTorch (optional), ModelInfo, ChatCompletionRequest  
**Evaluation Priority**: HIGH  
**Research References**: FrugalGPT, RouteLLM, LLM-Blender, Tree of Thoughts

## Resource Management Components

### 4. AccountManager (`src/core/account_manager.py`)
**Responsibility**: Manages credentials and account rotation  
**Key Features**:
- Encrypted credential storage
- Account rotation strategies
- Health tracking per account
- Provider-account association

**Dependencies**: Cryptography, SQLAlchemy  
**Evaluation Priority**: MEDIUM

### 5. RateLimiter (`src/core/rate_limiter.py`)
**Responsibility**: API rate limiting and quota management  
**Key Features**:
- Token bucket algorithm
- Per-provider rate limiting
- Request queuing
- Quota tracking

**Dependencies**: asyncio, Redis (optional)  
**Evaluation Priority**: MEDIUM

### 6. StateTracker (`src/core/state_tracker.py`)
**Responsibility**: Track system state and provider health  
**Key Features**:
- Provider availability monitoring
- Performance metrics collection
- State persistence

**Dependencies**: SQLite/Redis  
**Evaluation Priority**: MEDIUM

## Advanced Feature Components

### 7. EnsembleSystem (`src/core/ensemble_system.py`)
**Responsibility**: Multi-model response fusion and quality assessment  
**Key Features**:
- Parallel model invocation
- Response quality scoring
- Consensus-based selection
- Hybrid response generation

**Dependencies**: Multiple providers, quality metrics  
**Evaluation Priority**: HIGH  
**Research References**: LLM-Blender, Ensemble methods

### 8. AutoUpdater (`src/core/auto_updater.py`)
**Responsibility**: Automatic discovery of new providers and models  
**Key Features**:
- GitHub repository monitoring
- API-based model discovery
- Web scraping for provider info
- Browser automation (Playwright)
- Configuration auto-update

**Dependencies**: BeautifulSoup, Playwright, httpx  
**Evaluation Priority**: MEDIUM

### 9. Researcher (`src/core/researcher.py`)
**Responsibility**: Research and information gathering capabilities  
**Key Features**:
- Web search integration
- Document analysis
- Knowledge extraction

**Dependencies**: Search APIs, NLP tools  
**Evaluation Priority**: LOW

### 10. Planner (`src/core/planner.py`)
**Responsibility**: Task planning and decomposition  
**Key Features**:
- Goal decomposition
- Step generation
- Dependency analysis

**Dependencies**: LLM providers  
**Evaluation Priority**: LOW

### 11. CodeGenerator (`src/core/code_generator.py`)
**Responsibility**: Code generation and templating  
**Key Features**:
- Template-based generation
- Language-specific formatting
- Code validation

**Dependencies**: LLM providers  
**Evaluation Priority**: LOW

## Provider Implementation Components

### 12. BaseProvider (`src/providers/base.py`)
**Responsibility**: Abstract base class for all LLM providers  
**Key Features**:
- Standard interface definition
- Error types (ProviderError, RateLimitError, AuthenticationError)
- Common utilities

**Dependencies**: None (base abstraction)  
**Evaluation Priority**: HIGH

### 13. OpenRouterProvider (`src/providers/openrouter.py`)
**Responsibility**: OpenRouter API integration  
**Key Features**:
- Multi-model support (50+ free models)
- Standard OpenAI-compatible interface

**Dependencies**: BaseProvider, httpx  
**Evaluation Priority**: MEDIUM

### 14. GroqProvider (`src/providers/groq.py`)
**Responsibility**: Groq API integration  
**Key Features**:
- Ultra-fast inference
- Llama model support

**Dependencies**: BaseProvider, httpx  
**Evaluation Priority**: MEDIUM

### 15. CerebrasProvider (`src/providers/cerebras.py`)
**Responsibility**: Cerebras API integration  
**Key Features**:
- Fast inference
- Llama model support

**Dependencies**: BaseProvider, httpx  
**Evaluation Priority**: MEDIUM

## API Layer Components

### 16. FastAPI Server (`src/api/server.py`)
**Responsibility**: HTTP API endpoint implementation  
**Key Features**:
- OpenAI-compatible API
- Request validation
- Response streaming
- Error handling

**Dependencies**: FastAPI, Pydantic  
**Evaluation Priority**: MEDIUM

## Utility Components

### 17. BrowserMonitor (`src/core/browser_monitor.py`)
**Responsibility**: Web browser automation for provider monitoring  
**Key Features**:
- Playwright integration
- Dynamic content scraping
- Provider status checking

**Dependencies**: Playwright  
**Evaluation Priority**: LOW

## Data Model Components

### 18. Models (`src/models.py`)
**Responsibility**: Pydantic data models for type safety  
**Key Features**:
- Request/response models
- Configuration models
- Validation logic

**Dependencies**: Pydantic  
**Evaluation Priority**: MEDIUM

## Evaluation Order

### Phase 1 (Critical Path - HIGH Priority)
1. ProviderRouter - Foundation for provider selection
2. MetaModelController - Intelligence layer
3. LLMAggregator - Main orchestrator
4. EnsembleSystem - Advanced feature
5. BaseProvider - Provider abstraction

### Phase 2 (Core Features - MEDIUM Priority)
6. AccountManager - Resource management
7. RateLimiter - Resource management
8. StateTracker - Observability
9. AutoUpdater - Maintenance automation
10. OpenRouterProvider - Primary provider
11. FastAPI Server - API layer
12. Models - Data layer

### Phase 3 (Supporting Features - LOW Priority)
13. GroqProvider
14. CerebrasProvider
15. Researcher
16. Planner
17. CodeGenerator
18. BrowserMonitor

---
*Created: 2024-10-02*  
*Priority Assessment: Based on architectural centrality and feature importance*
