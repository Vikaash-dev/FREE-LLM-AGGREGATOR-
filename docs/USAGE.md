# LLM API Aggregator - Usage Guide

This guide will help you get started with the LLM API Aggregator, a production-grade solution for switching between different free LLM API providers.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-api-aggregator

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Credentials

Run the interactive setup to configure your API credentials:

```bash
python setup.py configure
```

This will guide you through adding API keys for various providers like:
- **OpenRouter** (50+ free models)
- **Groq** (ultra-fast inference)
- **Cerebras** (fast inference with 8K context)
- **Together AI** (free tier + trial credits)
- **Cohere** (Command models)

### 3. Start the Server

```bash
python main.py
```

The server will start on `http://localhost:8000` by default.

### 4. Test the API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello! How are you?"}]
  }'
```

## Usage Examples

### Basic Chat Completion

```python
import httpx

async def chat_example():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "auto",  # Automatically select best model
                "messages": [
                    {"role": "user", "content": "Explain quantum computing"}
                ]
            }
        )
        return response.json()
```

### Provider-Specific Request

```python
# Force a specific provider
response = await client.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "provider": "groq",  # Use Groq for fast inference
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": "Write a Python function"}
        ]
    }
)
```

### Streaming Response

```python
async def streaming_example():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "auto",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Tell me a story"}
                ]
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    print(data)
```

## Command Line Interface

### Interactive Chat

```bash
# Start interactive chat session
python cli.py chat

# Use specific model/provider
python cli.py chat --model "deepseek/deepseek-r1:free" --provider openrouter

# Single message
python cli.py chat --message "What is machine learning?"

# Streaming response
python cli.py chat --message "Explain AI" --stream
```

### List Available Models

```bash
python cli.py models
```

### Check Provider Status

```bash
python cli.py status
```

### View Usage Statistics

```bash
python cli.py stats
```

## Web Interface

Launch the web interface for a user-friendly experience:

```bash
streamlit run web_ui.py
```

Features:
- 💬 **Chat Interface**: Interactive chat with model/provider selection
- 📊 **Dashboard**: Real-time provider status and health monitoring
- 🔧 **Settings**: Add/manage API credentials
- 📈 **Analytics**: Usage statistics and performance metrics

## API Endpoints

### Chat Completions (OpenAI Compatible)

```
POST /v1/chat/completions
```

Request body:
```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "provider": "openrouter",  // Optional: force specific provider
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

### List Models

```
GET /v1/models
```

Returns all available models across providers.

### Admin Endpoints

#### Add Credentials
```
POST /admin/credentials
```

#### List Credentials
```
GET /admin/credentials
```

#### Provider Status
```
GET /admin/providers
```

#### Usage Statistics
```
GET /admin/usage-stats
```

## Configuration

### Provider Selection Logic

The system automatically selects the best provider based on:

1. **Content Analysis**: Detects if you need code generation, reasoning, etc.
2. **Model Availability**: Checks which providers have suitable models
3. **Rate Limits**: Avoids providers that are rate-limited
4. **Performance**: Uses historical performance data
5. **Cost**: Prioritizes free tiers and trial credits

### Routing Rules

You can customize routing behavior by modifying `config/providers.yaml`:

```yaml
routing_rules:
  - name: "code_generation"
    conditions:
      content_keywords: ["code", "python", "programming"]
    provider_preferences: ["openrouter", "groq"]
    fallback_chain: ["openrouter", "groq", "cerebras"]
```

### Account Rotation

The system automatically rotates between multiple accounts for the same provider to:
- Maximize free tier usage
- Distribute load across accounts
- Avoid hitting individual account limits

## Best Practices

### 1. Multiple Accounts

Add multiple accounts per provider for better rate limits:

```bash
# Add multiple OpenRouter accounts
python setup.py configure
# Add account1@example.com with API key 1
# Add account2@example.com with API key 2
```

### 2. Model Selection

- Use `"model": "auto"` for automatic selection
- Specify models for specific use cases:
  - `deepseek/deepseek-r1:free` for reasoning tasks
  - `qwen/qwen-2.5-coder-32b-instruct:free` for code generation
  - `llama-3.3-70b-versatile` for general tasks

### 3. Provider Selection

- Use `"provider": null` for automatic routing
- Force specific providers when needed:
  - `groq` for fastest responses
  - `openrouter` for most model variety
  - `cerebras` for good balance of speed and quality

### 4. Rate Limit Management

- Monitor usage with `/admin/usage-stats`
- Add more accounts when approaching limits
- Use the web interface to track real-time status

## Troubleshooting

### Common Issues

1. **"No valid credentials" error**
   - Run `python setup.py configure` to add API keys
   - Check that your API keys are valid

2. **Rate limit errors**
   - Add more accounts for the same provider
   - Wait for rate limits to reset
   - Try a different provider

3. **Provider unavailable**
   - Check provider status with `python cli.py status`
   - Verify your internet connection
   - Check if the provider's API is down

### Logs

Check the log file for detailed error information:
```bash
tail -f llm_aggregator.log
```

### Health Check

Monitor system health:
```bash
curl http://localhost:8000/health
```

## Advanced Features

### Custom Headers

Some providers require custom headers:

```python
# For OpenRouter with app identification
await account_manager.add_credentials(
    provider="openrouter",
    account_id="my_account",
    api_key="your_api_key",
    additional_headers={
        "HTTP-Referer": "https://myapp.com",
        "X-Title": "My App"
    }
)
```

### Performance Monitoring

The system tracks performance metrics:
- Response times
- Success rates
- Error counts
- Rate limit hits

Access via `/admin/usage-stats` or the web interface.

### Fallback Chains

Configure custom fallback chains for high availability:

```yaml
routing_rules:
  - name: "high_availability"
    provider_preferences: ["groq", "cerebras", "openrouter"]
    fallback_chain: ["groq", "cerebras", "openrouter", "together"]
```

## Security

- API keys are encrypted at rest
- Use HTTPS in production
- Rotate API keys regularly
- Monitor usage for anomalies

## Support

For issues and questions:
1. Check the logs: `llm_aggregator.log`
2. Verify provider status: `python cli.py status`
3. Test individual providers manually
4. Check the provider's documentation for API changes