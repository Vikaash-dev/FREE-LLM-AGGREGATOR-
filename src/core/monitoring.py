"""
Prometheus metrics for monitoring the LLM API Aggregator.
"""

from prometheus_client import Counter, Histogram

# A counter to track the total number of LLM requests.
# Labels:
# - provider: The name of the LLM provider (e.g., 'openrouter', 'groq').
# - model: The name of the model used for the request.
# - status: The outcome of the request ('success', 'failure').
LLM_REQUESTS_TOTAL = Counter(
    'llm_requests_total',
    'Total number of LLM requests',
    ['provider', 'model', 'status']
)

# A histogram to track the latency of LLM requests in seconds.
# Labels:
# - provider: The name of the LLM provider.
# - model: The name of the model used for the request.
LLM_REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'Latency of LLM requests in seconds',
    ['provider', 'model'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]  # Buckets from 100ms to 1 minute
)

# A counter to track the number of failed requests by error type.
# Labels:
# - provider: The name of the LLM provider.
# - model: The name of the model used for the request.
# - error_type: The type of error (e.g., 'ProviderError', 'RateLimitError').
LLM_ERRORS_TOTAL = Counter(
    'llm_errors_total',
    'Total number of failed LLM requests by error type',
    ['provider', 'model', 'error_type']
)

# A counter for model selection events from the meta-controller.
# Labels:
# - model: The model that was selected.
# - provider: The provider of the selected model.
META_CONTROLLER_MODEL_SELECTIONS = Counter(
    'meta_controller_model_selections_total',
    'Total number of times a model was selected by the meta-controller',
    ['provider', 'model']
)
