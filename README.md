# 🤖 Multi-Provider LLM API Aggregator

A production-grade solution for switching between different free LLM API providers with intelligent routing, account management, and fallback mechanisms.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## Problem Statement

The user needs a system that can:
1. **Switch between different free LLM API providers** from various sources
2. **Manage different accounts** for these providers
3. **Handle model switching dynamically** with intelligent routing
4. **Optimize costs** by using free tiers and trial credits effectively
5. **Provide fallback mechanisms** when providers fail or hit rate limits

## Key Features

- 🔄 **Multi-Provider Support**: Integrate with 25+ free LLM providers
- 🔐 **Secure Account Management**: Encrypted credential storage and rotation
- 🎯 **Intelligent Routing**: Automatic provider selection based on model, cost, and availability
- 📊 **Rate Limit Management**: Track and respect provider-specific limits
- 🔄 **Automatic Fallbacks**: Seamless switching when providers fail
- 💰 **Cost Optimization**: Prioritize free tiers and trial credits
- 📈 **Usage Analytics**: Track usage across providers and accounts
- 🛡️ **Error Handling**: Robust error handling and retry mechanisms
- 🔄 **Auto-Updater**: Automatic discovery of new providers and models
- 🧠 **Meta-Controller**: Research-based intelligent model selection
- 🎯 **Ensemble System**: Multi-model response fusion and quality assessment
- 🚀 **ADA-7 Framework**: 7-stage evolutionary development methodology with academic research integration

## Supported Providers

### Free Providers
- OpenRouter (50+ free models)
- Google AI Studio (Gemini models)
- NVIDIA NIM (Various open models)
- Mistral (La Plateforme & Codestral)
- HuggingFace Inference
- Cerebras (Fast inference)
- Groq (Ultra-fast inference)
- Together (Free tier)
- Cohere (Command models)
- GitHub Models (Premium models)
- Chutes (Decentralized)
- Cloudflare Workers AI
- Google Cloud Vertex AI

### Trial Credit Providers
- Together ($1 credit)
- Fireworks ($1 credit)
- Unify ($5 credit)
- Baseten ($30 credit)
- Nebius ($1 credit)
- Novita ($0.5-$10 credit)
- AI21 ($10 credit)
- Upstage ($10 credit)
- And many more...

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   API Gateway   │    │  Provider Pool  │
│                 │────│                 │────│                 │
│ - Web UI        │    │ - Rate Limiting │    │ - OpenRouter    │
│ - CLI Tool      │    │ - Load Balancer │    │ - Google AI     │
│ - API Client    │    │ - Auth Handler  │    │ - NVIDIA NIM    │
└─────────────────┘    └─────────────────┘    │ - Groq          │
                                              │ - Cerebras      │
                                              │ - ...           │
                                              └─────────────────┘
```

## Auto-Updater System

The system includes a comprehensive auto-updater that continuously discovers new free LLM providers and models:

### 🔍 Multi-Source Discovery
- **GitHub Integration**: Monitors community projects like `cheahjs/free-llm-api-resources`
- **API Discovery**: Real-time discovery of new models via provider APIs
- **Web Scraping**: Automated monitoring of provider websites
- **Browser Automation**: Advanced monitoring using Playwright

### 🔄 Intelligent Integration
- **Automatic Updates**: Seamlessly integrates discovered changes
- **Meta-Controller Adaptation**: Updates model capability profiles
- **Configuration Management**: Maintains provider configs and rate limits
- **Real-time Monitoring**: Live status dashboard and update history

### 📊 Community Integration
Integrates with existing GitHub projects:
- [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources)
- [zukixa/cool-ai-stuff](https://github.com/zukixa/cool-ai-stuff)
- [wdhdev/free-for-life](https://github.com/wdhdev/free-for-life)

```bash
# Run auto-updater demo
python auto_updater_demo.py

# Test integration
python test_auto_updater_integration.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure providers
python setup.py configure

# Start the service
python main.py --port 8000

# Test the API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration

The system uses a hierarchical configuration approach:

1. **Provider Configuration**: Define available providers and their capabilities
2. **Account Management**: Securely store API keys and credentials
3. **Routing Rules**: Define how to select providers for different requests
4. **Rate Limits**: Configure provider-specific limits and quotas

## Usage Examples

### Basic Chat Completion
```python
from llm_aggregator import LLMAggregator

aggregator = LLMAggregator()
response = aggregator.chat_completion(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    model="auto"  # Automatically select best available model
)
```

### Provider-Specific Request
```python
response = aggregator.chat_completion(
    messages=[{"role": "user", "content": "Write Python code"}],
    provider="openrouter",
    model="deepseek/deepseek-coder-33b-instruct"
)
```

### Streaming Response
```python
for chunk in aggregator.chat_completion_stream(
    messages=[{"role": "user", "content": "Tell me a story"}],
    model="auto"
):
    print(chunk.choices[0].delta.content, end="")
```

## Advanced Features

### Intelligent Provider Selection
The system automatically selects the best provider based on:
- Model availability and capabilities
- Current rate limits and quotas
- Provider reliability and response time
- Cost optimization (prioritizing free tiers)

### Account Rotation
Automatically rotate between multiple accounts for the same provider to maximize free tier usage.

### Fallback Chains
Configure fallback chains for high availability:
```yaml
fallback_chains:
  coding:
    - provider: openrouter
      model: deepseek/deepseek-coder-33b-instruct
    - provider: groq
      model: llama-3.1-70b-versatile
    - provider: cerebras
      model: llama3.1-8b
```

## Monitoring and Analytics

- Real-time usage dashboard
- Provider performance metrics
- Cost tracking and optimization suggestions
- Rate limit monitoring and alerts

## Security

- Encrypted credential storage using industry-standard encryption
- API key rotation and management
- Audit logging for all requests
- Rate limiting and abuse prevention

## ADA-7 Advanced Development Assistant

The repository now includes ADA-7, an Advanced Development Assistant framework that guides software development through 7 evolutionary stages:

1. **Requirements Analysis & Competitive Intelligence** - User personas, competitive analysis, feature gaps
2. **Architecture Design & Academic Validation** - Architecture variants with research backing
3. **Component Design & Technology Stack** - Component breakdown and technology selection
4. **Implementation Strategy & Development Pipeline** - MVP planning, CI/CD setup
5. **Testing Framework & Quality Assurance** - Testing pyramid, quality gates
6. **Deployment & Infrastructure Management** - IaC, security, monitoring
7. **Maintenance & Continuous Evolution** - Operational excellence, evolution roadmap

### Using ADA-7

```python
from src.core.ada7.framework import ADA7Framework

# Initialize framework
ada7 = ADA7Framework()

# Start new project
project = await ada7.start_project(
    name="My Application",
    description="AI-powered application",
    constraints={"budget": 10000, "timeline": "3 months"}
)

# Execute all stages
results = await ada7.execute_all_stages(project)
```

See [ADA7_FRAMEWORK.md](ADA7_FRAMEWORK.md) and [ADA7_USAGE_GUIDE.md](ADA7_USAGE_GUIDE.md) for complete documentation.

### ADA-7 Demos

```bash
# Run basic demo
python ada7_demo.py

# Run integration example
python ada7_integration_example.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

MIT License - see [LICENSE](LICENSE) for details.