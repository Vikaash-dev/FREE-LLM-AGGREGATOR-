"""
Main LLM Aggregator class that orchestrates provider selection and routing.
"""

import asyncio
import structlog # Changed from logging
import random
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import httpx # Added for specific exception handling

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ProviderConfig,
    AccountCredentials,
    RoutingRule,
    ModelCapability,
)
from ..providers.base import BaseProvider, ProviderError, RateLimitError, AuthenticationError
from .account_manager import AccountManager
from .router import ProviderRouter
from .rate_limiter import RateLimiter
from .meta_controller import MetaModelController, ModelCapabilityProfile # Removed TaskComplexityAnalyzer, not directly used here
from .ensemble_system import EnsembleSystem
from .auto_updater import AutoUpdater, integrate_auto_updater
from src.config import settings # Centralized settings


logger = structlog.get_logger(__name__) # Changed from logging


class LLMAggregator:
    """
    Orchestrates interactions with multiple Language Model (LLM) providers.

    This class manages a pool of LLM providers, selects appropriate providers
    for incoming requests, handles retries, rate limiting, and advanced features
    like meta-controller based model selection, ensemble responses, and auto-updates
    for provider configurations. It aims to provide a unified and resilient interface
    for accessing various LLMs.
    """
    
    def __init__(
        self,
        providers: List[BaseProvider],
        account_manager: AccountManager,
        router: ProviderRouter,
        rate_limiter: RateLimiter,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        enable_meta_controller: bool = True,
        enable_ensemble: bool = False,
        enable_auto_updater: bool = True,
        auto_update_interval: Optional[int] = None
    ):
        """
        Initializes the LLMAggregator.

        Args:
            providers: A list of initialized provider instances (e.g., OpenAIProvider).
            account_manager: Manages credentials for different providers.
            router: Determines the order/selection of providers for a request.
            rate_limiter: Handles rate limiting for API requests.
            max_retries: Maximum number of retries for a request to a provider.
                         Sources from settings if None.
            retry_delay: Initial delay in seconds between retries.
                         Sources from settings if None.
            enable_meta_controller: Flag to enable the meta-controller for intelligent
                                    model selection.
            enable_ensemble: Flag to enable ensemble methods for responses.
            enable_auto_updater: Flag to enable automatic updates of provider info.
            auto_update_interval: Interval in minutes for auto-updater.
                                  Sources from settings if None.
        """
        self.providers: Dict[str, BaseProvider] = {provider.name: provider for provider in providers}
        self.account_manager: AccountManager = account_manager
        self.router: ProviderRouter = router
        self.rate_limiter = rate_limiter
        
        # Enhanced features
        self.enable_meta_controller = enable_meta_controller
        self.enable_ensemble = enable_ensemble
        self.enable_auto_updater = enable_auto_updater
        
        # Circuit Breaker Attributes
        self.provider_failure_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.provider_circuit_open_until: Dict[str, Dict[str, datetime]] = defaultdict(lambda: defaultdict(datetime))
        self.CIRCUIT_BREAKER_THRESHOLD = 3  # Max failures before opening circuit
        self.CIRCUIT_BREAKER_COOLDOWN_SECONDS = 300  # 5 minutes cooldown

        # Configuration from settings or defaults
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
        self.retry_delay = retry_delay if retry_delay is not None else settings.RETRY_DELAY
        self.auto_update_interval = auto_update_interval if auto_update_interval is not None else settings.AUTO_UPDATE_INTERVAL_MINUTES

        # Initialize meta-controller if enabled
        if self.enable_meta_controller:
            self.meta_controller = self._initialize_meta_controller()
        else:
            self.meta_controller = None
            
        # Initialize ensemble system if enabled
        if self.enable_ensemble:
            self.ensemble_system = EnsembleSystem()
        else:
            self.ensemble_system = None
            
        # Initialize auto-updater if enabled
        if self.enable_auto_updater:
            self.auto_updater = AutoUpdater(account_manager=self.account_manager)
            # Start auto-update task
            asyncio.create_task(self._start_auto_updater())
        else:
            self.auto_updater = None
        
        logger.info("LLM Aggregator initialized",
                    num_providers=len(self.providers),
                    meta_controller_enabled=self.enable_meta_controller,
                    ensemble_system_enabled=self.enable_ensemble,
                    auto_updater_enabled=self.enable_auto_updater)
    
    def _initialize_meta_controller(self) -> MetaModelController:
        """Initialize the meta-controller with model capability profiles."""
        
        model_profiles = {}
        
        for provider_name, provider in self.providers.items():
            models = provider.list_models()
            
            for model in models:
                # Create capability profile for each model
                profile = ModelCapabilityProfile(
                    model_name=model.name,
                    provider=provider_name,
                    size_category=self._categorize_model_size(model),
                    
                    # Capability scores (estimated based on model characteristics)
                    reasoning_ability=self._estimate_reasoning_ability(model),
                    code_generation=self._estimate_code_generation(model),
                    mathematical_reasoning=self._estimate_math_reasoning(model),
                    creative_writing=self._estimate_creative_writing(model),
                    factual_knowledge=self._estimate_factual_knowledge(model),
                    instruction_following=self._estimate_instruction_following(model),
                    context_handling=self._estimate_context_handling(model),
                    
                    # Performance metrics (initial estimates)
                    avg_response_time=2.0,  # Will be updated with real data
                    reliability_score=0.8,  # Will be updated with real data
                    cost_per_token=0.0 if model.is_free else 0.001,
                    max_context_length=model.context_length or 4096,
                    
                    # Specializations
                    domain_expertise=self._identify_domain_expertise(model),
                    preferred_task_types=self._identify_preferred_tasks(model)
                )
                
                model_profiles[model.name] = profile
        
        return MetaModelController(model_profiles)
    
    def _categorize_model_size(self, model: ModelInfo) -> str:
        """Categorize model size based on name and characteristics."""
        
        model_name_lower = model.name.lower()
        
        # Check for size indicators in model name
        if any(indicator in model_name_lower for indicator in ['7b', '8b', 'small', 'mini']):
            return "small"
        elif any(indicator in model_name_lower for indicator in ['13b', '14b', '20b', '27b', 'medium']):
            return "medium"
        elif any(indicator in model_name_lower for indicator in ['70b', '72b', '405b', 'large', 'xl']):
            return "large"
        else:
            # Default categorization based on context length
            if model.context_length and model.context_length > 100000:
                return "large"
            elif model.context_length and model.context_length > 32000:
                return "medium"
            else:
                return "small"
    
    def _estimate_reasoning_ability(self, model: ModelInfo) -> float:
        """Estimate reasoning ability based on model characteristics."""
        
        model_name_lower = model.name.lower()
        
        # Models known for reasoning
        if any(keyword in model_name_lower for keyword in ['r1', 'reasoning', 'think', 'o1']):
            return 0.9
        elif any(keyword in model_name_lower for keyword in ['llama', 'qwen', 'deepseek']):
            return 0.8
        elif any(keyword in model_name_lower for keyword in ['gemma', 'mistral']):
            return 0.7
        else:
            return 0.6
    
    def _estimate_code_generation(self, model: ModelInfo) -> float:
        """Estimate code generation ability."""
        
        model_name_lower = model.name.lower()
        
        if any(keyword in model_name_lower for keyword in ['code', 'coder', 'codestral']):
            return 0.9
        elif any(keyword in model_name_lower for keyword in ['deepseek', 'qwen']):
            return 0.8
        elif any(keyword in model_name_lower for keyword in ['llama', 'mistral']):
            return 0.7
        else:
            return 0.5
    
    def _estimate_math_reasoning(self, model: ModelInfo) -> float:
        """Estimate mathematical reasoning ability."""
        
        model_name_lower = model.name.lower()
        
        if any(keyword in model_name_lower for keyword in ['math', 'deepseek', 'qwen']):
            return 0.8
        elif any(keyword in model_name_lower for keyword in ['llama', 'r1']):
            return 0.7
        else:
            return 0.6
    
    def _estimate_creative_writing(self, model: ModelInfo) -> float:
        """Estimate creative writing ability."""
        
        model_name_lower = model.name.lower()
        
        if any(keyword in model_name_lower for keyword in ['creative', 'story', 'writer']):
            return 0.9
        elif any(keyword in model_name_lower for keyword in ['llama', 'mistral', 'gemma']):
            return 0.7
        else:
            return 0.6
    
    def _estimate_factual_knowledge(self, model: ModelInfo) -> float:
        """Estimate factual knowledge capability."""
        
        # Larger models generally have more factual knowledge
        if model.context_length and model.context_length > 100000:
            return 0.8
        elif model.context_length and model.context_length > 32000:
            return 0.7
        else:
            return 0.6
    
    def _estimate_instruction_following(self, model: ModelInfo) -> float:
        """Estimate instruction following ability."""
        
        model_name_lower = model.name.lower()
        
        if any(keyword in model_name_lower for keyword in ['instruct', 'chat', 'assistant']):
            return 0.8
        else:
            return 0.7
    
    def _estimate_context_handling(self, model: ModelInfo) -> float:
        """Estimate context handling ability."""
        
        if model.context_length:
            if model.context_length >= 128000:
                return 0.9
            elif model.context_length >= 32000:
                return 0.8
            elif model.context_length >= 8000:
                return 0.7
            else:
                return 0.6
        else:
            return 0.6
    
    def _identify_domain_expertise(self, model: ModelInfo) -> List[str]:
        """Identify domain expertise based on model characteristics."""
        
        model_name_lower = model.name.lower()
        expertise = []
        
        if any(keyword in model_name_lower for keyword in ['code', 'coder']):
            expertise.append('programming')
        if any(keyword in model_name_lower for keyword in ['math']):
            expertise.append('mathematics')
        if any(keyword in model_name_lower for keyword in ['reasoning', 'r1']):
            expertise.append('reasoning')
        if any(keyword in model_name_lower for keyword in ['creative', 'story']):
            expertise.append('creative_writing')
        
        return expertise if expertise else ['general']
    
    def _identify_preferred_tasks(self, model: ModelInfo) -> List[str]:
        """Identify preferred task types for the model."""
        
        model_name_lower = model.name.lower()
        tasks = []
        
        if any(keyword in model_name_lower for keyword in ['code', 'coder']):
            tasks.extend(['code_generation', 'debugging', 'code_review'])
        if any(keyword in model_name_lower for keyword in ['math']):
            tasks.extend(['mathematical_reasoning', 'problem_solving'])
        if any(keyword in model_name_lower for keyword in ['reasoning', 'r1']):
            tasks.extend(['logical_reasoning', 'analysis', 'problem_solving'])
        if any(keyword in model_name_lower for keyword in ['chat', 'assistant']):
            tasks.extend(['conversation', 'question_answering'])
        
        return tasks if tasks else ['general_text_generation']
    
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """
        Processes a chat completion request.

        This method selects an appropriate LLM provider (potentially using a meta-controller
        or ensemble methods if enabled), sends the request, and returns the response.
        It handles rate limiting and ensures that streaming requests are directed to
        `chat_completion_stream`.

        Args:
            request: The chat completion request object.
            user_id: Optional identifier for the user making the request, used for
                     rate limiting and potentially for personalized routing.

        Returns:
            A ChatCompletionResponse object from the selected provider.

        Raises:
            ValueError: If a streaming request is made to this non-streaming endpoint.
            ProviderError: If all attempts to get a response from providers fail.
            RateLimitExceeded: If the user exceeds their rate limits.
        """
        
        if request.stream:
            logger.error("Streaming request made to non-streaming endpoint", request_id=getattr(request, 'id', 'N/A'))
            raise ValueError("Use chat_completion_stream for streaming requests")
        
        # Apply rate limiting
        await self.rate_limiter.acquire(user_id)
        
        try:
            # Use meta-controller for intelligent model selection if enabled
            if self.enable_meta_controller and self.meta_controller:
                return await self._chat_completion_with_meta_controller(request, user_id)
            
            # Use ensemble system if enabled
            elif self.enable_ensemble and self.ensemble_system:
                return await self._chat_completion_with_ensemble(request, user_id)
            
            # Fallback to traditional routing
            else:
                return await self._chat_completion_traditional(request, user_id)
            
        finally:
            self.rate_limiter.release(user_id)
    
    async def _chat_completion_with_meta_controller(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Chat completion using meta-controller for intelligent model selection."""
        
        start_time = time.time()
        
        # Get optimal model from meta-controller
        optimal_model, confidence = await self.meta_controller.select_optimal_model(request, user_id) # user_id might be useful here
        
        logger.info("Meta-controller model selection",
                    selected_model=optimal_model,
                    confidence=round(confidence, 2),
                    user_id=user_id,
                    request_model=request.model) # Log initial request model
        
        # If confidence is low, get cascade chain
        if confidence < 0.7:
            cascade_chain = await self.meta_controller.get_cascade_chain(request)
            logger.info("Low confidence, using cascade chain",
                        cascade_chain=cascade_chain,
                        selected_model=optimal_model,
                        confidence=round(confidence,2),
                        user_id=user_id)
        else:
            cascade_chain = [optimal_model]
        
        # Try models in cascade order
        last_error = None
        for model_name in cascade_chain:
            # Find provider for this model
            provider_name = None
            for prov_name, provider in self.providers.items():
                if any(model.name == model_name for model in provider.list_models()):
                    provider_name = prov_name
                    break
            
            if not provider_name:
                continue
            
            try:
                # Create request with specific model
                model_request = request.model_copy()
                model_request.model = model_name
                
                response = await self._try_provider(provider_name, model_request)
                if response:
                    # Update performance feedback
                    response_time = time.time() - start_time
                    await self.meta_controller.update_performance_feedback(
                        model_name, request, True, response_time, None # Error details are None for success
                    )
                    
                    logger.info("Meta-controller: Successfully completed request",
                                model_used=model_name,
                                provider=provider_name,
                                response_time=response_time,
                                user_id=user_id)
                    return response
                    
            except Exception as e: # This is a broad catch, specific errors are handled in _try_provider
                logger.warning("Meta-controller: Model failed in cascade",
                               model_name=model_name,
                               provider=provider_name, # provider_name might not be set if model not found in any provider
                               error=str(e),
                               user_id=user_id,
                               exc_info=True) # Log stack trace for unexpected errors
                # Update performance feedback for failure
                response_time = time.time() - start_time # Recalculate response_time for accuracy
                await self.meta_controller.update_performance_feedback(
                    model_name, request, False, response_time, None
                )
                if provider_name: # Ensure provider_name was found
                    self.router.update_provider_score(provider_name, success=False, model_name=model_name)
                last_error = e # last_error should ideally be a ProviderError or similar structured error.
                continue
        
        # If all models in cascade failed, fallback to traditional routing
        logger.warning("Meta-controller cascade failed, falling back to traditional routing",
                       last_error=str(last_error) if last_error else "No specific error", # Log last error if available
                       user_id=user_id,
                       request_model=request.model,
                       exc_info=True if last_error else False) # Log stack trace if an error occurred
        return await self._chat_completion_traditional(request, user_id)
    
    async def _chat_completion_with_ensemble(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Chat completion using ensemble system for improved accuracy."""
        
        # Get top 3 models for ensemble
        if self.meta_controller:
            cascade_chain = await self.meta_controller.get_cascade_chain(request)
            ensemble_models = cascade_chain[:3]  # Top 3 models
        else:
            # Fallback to provider chain
            provider_chain = await self._get_provider_chain(request)
            ensemble_models = []
            for provider_name in provider_chain[:3]:
                provider = self.providers[provider_name]
                models = provider.list_models()
                if models:
                    ensemble_models.append(models[0].name)
        
        # Generate responses from multiple models
        model_responses = {}
        model_metadata = {}
        
        tasks = []
        for model_name in ensemble_models:
            # Find provider for this model
            provider_name = None
            for prov_name, provider in self.providers.items():
                if any(model.name == model_name for model in provider.list_models()):
                    provider_name = prov_name
                    break
            
            if provider_name:
                task = self._get_model_response(provider_name, model_name, request)
                tasks.append((model_name, provider_name, task))
        
        # Execute all requests concurrently
        for model_name, provider_name, task in tasks:
            try:
                start_time = time.time()
                response = await task
                response_time = time.time() - start_time
                
                if response:
                    model_responses[model_name] = response
                    model_metadata[model_name] = {
                        'provider': provider_name,
                        'response_time': response_time,
                        'confidence': 0.8,  # Default confidence
                        'cost_estimate': 0.0
                    }
            except Exception as e: # Broad catch, specific errors are in _try_provider
                logger.warning("Ensemble model failed",
                               model_name=model_name,
                               provider=provider_name,
                               error=str(e),
                               user_id=user_id, # Assuming user_id is accessible here
                               exc_info=True)
                continue
        
        # If we have multiple responses, use ensemble system
        if len(model_responses) > 1:
            logger.info("Generating ensemble response",
                        num_models_participating=len(model_responses),
                        models=list(model_responses.keys()),
                        user_id=user_id)
            return await self.ensemble_system.generate_ensemble_response(
                request, model_responses, model_metadata
            )
        
        # If only one response, return it
        elif model_responses:
            logger.info("Ensemble: Only one model succeeded, returning its response",
                        model_returned=list(model_responses.keys())[0],
                        user_id=user_id)
            return list(model_responses.values())[0]
        
        # If no responses, fallback to traditional routing
        else:
            logger.warning("Ensemble failed: No models returned a response, falling back to traditional routing",
                           user_id=user_id,
                           request_model=request.model)
            return await self._chat_completion_traditional(request, user_id)
    
    async def _get_model_response(self, provider_name: str, model_name: str, 
                                request: ChatCompletionRequest) -> Optional[ChatCompletionResponse]:
        """Get response from a specific model."""
        
        model_request = request.model_copy()
        model_request.model = model_name
        
        return await self._try_provider(provider_name, model_request)
    
    async def _chat_completion_traditional(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Traditional chat completion with provider chain fallback."""
        
        # Get provider selection strategy
        provider_chain = await self._get_provider_chain(request)
        
        # Try providers in order
        last_error = None
        for provider_name in provider_chain:
            try:
                response = await self._try_provider(provider_name, request) # _try_provider logs its own specifics
                if response:
                    logger.info("Traditional routing: Successfully completed request",
                                provider_used=provider_name,
                                model_requested=request.model, # Actual model used is logged in _try_provider
                                user_id=user_id)
                    return response
            # These specific errors are already handled and logged in _try_provider and re-raised.
            # The circuit breaker logic in _try_provider handles repeated failures.
            # Logging them again here would be redundant unless we want a summary log after trying a provider.
            except RateLimitError as e:
                logger.debug("Traditional routing: Rate limit hit, trying next provider", provider=provider_name, error=str(e), user_id=user_id)
                last_error = e
                continue
            except AuthenticationError as e:
                logger.debug("Traditional routing: Authentication failed, trying next provider", provider=provider_name, error=str(e), user_id=user_id)
                # mark_credentials_invalid is called in _try_provider
                last_error = e
                continue
            except ProviderError as e: # Includes errors from circuit breaker like "Circuit Open"
                logger.debug("Traditional routing: Provider error, trying next provider", provider=provider_name, error=str(e), user_id=user_id)
                last_error = e
                continue
            except Exception as e: # Catch any other unexpected error from _try_provider if it wasn't wrapped
                logger.warning("Traditional routing: Unexpected error from provider, trying next", provider=provider_name, error=str(e), user_id=user_id, exc_info=True)
                last_error = e
                continue

        # If we get here, all providers failed
        logger.error("Traditional routing: All providers failed",
                     last_error=str(last_error) if last_error else "No specific error",
                     provider_chain_tried=provider_chain,
                     user_id=user_id,
                     request_model=request.model,
                     exc_info=True if last_error else False)
        raise ProviderError(f"All providers failed. Last error: {last_error}")
    
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """
        Processes a streaming chat completion request.

        This method selects an appropriate LLM provider and streams the response back
        as an asynchronous generator. It handles rate limiting.

        Args:
            request: The chat completion request object, with `stream` set to True.
            user_id: Optional identifier for the user making the request.

        Yields:
            ChatCompletionResponse: Chunks of the streaming response from the provider.

        Raises:
            ProviderError: If all attempts to get a response from providers fail.
            RateLimitExceeded: If the user exceeds their rate limits.
        """
        
        # Apply rate limiting
        await self.rate_limiter.acquire(user_id)
        
        try:
            # Get provider selection strategy
            provider_chain = await self._get_provider_chain(request)
            
            # Try providers in order
            for provider_name in provider_chain:
                try:
                    async for chunk in self._try_provider_stream(provider_name, request): # _try_provider_stream logs its own specifics
                        yield chunk
                    logger.info("Streaming: Successfully completed stream",
                                provider_used=provider_name,
                                model_requested=request.model, # Actual model used is logged in _try_provider_stream
                                user_id=user_id)
                    return  # Successfully completed
                    
                except RateLimitError as e:
                    logger.debug("Streaming: Rate limit hit, trying next provider", provider=provider_name, error=str(e), user_id=user_id)
                    continue
                except AuthenticationError as e:
                    logger.debug("Streaming: Authentication failed, trying next provider", provider=provider_name, error=str(e), user_id=user_id)
                    # mark_credentials_invalid is called in _try_provider_stream
                    continue
                except ProviderError as e:
                    logger.debug("Streaming: Provider error, trying next provider", provider=provider_name, error=str(e), user_id=user_id)
                    continue
                except Exception as e: # Catch any other unexpected error
                    logger.warning("Streaming: Unexpected error from provider, trying next", provider=provider_name, error=str(e), user_id=user_id, exc_info=True)
                    continue
            
            # If we get here, all providers failed
            logger.error("Streaming: All providers failed",
                         provider_chain_tried=provider_chain,
                         user_id=user_id,
                         request_model=request.model)
            raise ProviderError("All providers failed for streaming request") # Consider more specific error or structured response
            
        finally:
            self.rate_limiter.release(user_id)
    
    async def _get_provider_chain(self, request: ChatCompletionRequest) -> List[str]:
        """Get ordered list of providers to try for this request."""
        
        # If specific provider requested, use it
        if request.provider and request.provider in self.providers:
            return [request.provider]
        
        # Use router to determine provider chain
        return await self.router.get_provider_chain(request)
    
    async def _try_provider(
        self,
        provider_name: str,
        request: ChatCompletionRequest
    ) -> Optional[ChatCompletionResponse]:
        """Try to complete request with specific provider, including circuit breaker and refined error handling."""
        
        # Provider check should happen before model resolution for circuit breaker key
        provider = self.providers.get(provider_name)
        if not provider or not provider.is_available:
            logger.debug("_try_provider: Provider not found or unavailable", provider_name=provider_name)
            return None
        
        # Get credentials for this provider
        credentials = await self.account_manager.get_credentials(provider_name)
        if not credentials:
            logger.warning("No valid credentials for provider", provider_name=provider_name, model_requested=request.model)
            return None
        
        # Resolve model name if "auto" is specified
        model_name = request.model # This is the model_name used for circuit breaker key initially
        if model_name == "auto":
            # We need to select the model first to get the actual model_name for circuit breaker
            # This means the circuit check for "auto" might be less specific or happen after model selection
            # For simplicity, the circuit for "auto" could be generic for the provider,
            # or we check *after* model_name is resolved.
            # Let's refine: check circuit AFTER model_name is resolved.
            pass # Will resolve model_name later

        # If model_name is not "auto", check circuit now.
        if model_name != "auto":
            open_until = self.provider_circuit_open_until[provider_name].get(model_name)
            if open_until and datetime.utcnow() < open_until:
                logger.warning("Circuit open for model", provider_name=provider_name, model_name=model_name, open_until=open_until.isoformat())
                return None # Circuit is open for this specific model

        # Resolve actual model name if "auto"
        if request.model == "auto": # Use request.model here, as model_name might have been changed above
            resolved_model_name = await self._select_model(provider, request) # _select_model could log its choice/failure
            if not resolved_model_name:
                logger.warning("No suitable model found by _select_model", provider_name=provider_name, original_request_model=request.model)
                # Consider this a failure for "auto" on this provider
                self.provider_failure_counts[provider_name]["auto"] += 1 # Use "auto" as key
                if self.provider_failure_counts[provider_name]["auto"] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name]["auto"] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name]["auto"] = 0
                    logger.error("Circuit for 'auto' model selection opened", provider_name=provider_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                return None
            model_name = resolved_model_name # This is the actual model_name to be used
            logger.info("Model 'auto' resolved", provider_name=provider_name, resolved_model_name=model_name)

            # Now check circuit for the resolved model_name
            open_until = self.provider_circuit_open_until[provider_name].get(model_name)
            if open_until and datetime.utcnow() < open_until:
                logger.warning("Circuit open for resolved 'auto' model", provider_name=provider_name, model_name=model_name, open_until=open_until.isoformat())
                return None


        # Create request with resolved model
        resolved_request = request.copy()
        resolved_request.model = model_name # model_name is now the resolved one
        
        # Attempt request with retries
        for attempt in range(self.max_retries):
            try:
                # model_name here is the one used for the actual API call and for circuit breaker logic on failure/success
                response = await provider.chat_completion(resolved_request, credentials)
                
                # Update credentials usage
                await self.account_manager.update_usage(credentials)
                
                # Success: reset failure count for this provider/model
                self.provider_failure_counts[provider_name][model_name] = 0
                # Optionally log if circuit was closed:
                if self.provider_circuit_open_until[provider_name].get(model_name) and \
                   self.provider_circuit_open_until[provider_name][model_name] < datetime.utcnow(): # Check if it was open and now it's past cooldown
                    logger.info("Circuit closed due to successful call", provider_name=provider_name, model_name=model_name)
                
                return response

            # Specific httpx errors
            except httpx.ReadTimeout as e:
                logger.warning("ReadTimeout from provider", provider_name=provider_name, model_name=model_name, attempt=attempt + 1, max_retries=self.max_retries, error=str(e))
                self.provider_failure_counts[provider_name][model_name] += 1
                if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name][model_name] = 0
                    logger.error("Circuit opened due to ReadTimeouts", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                if attempt == self.max_retries - 1:
                    raise ProviderError(f"ReadTimeout with {provider_name}/{model_name} after {self.max_retries} attempts: {e}") from e
            except httpx.ConnectError as e:
                logger.warning("ConnectError with provider", provider_name=provider_name, model_name=model_name, attempt=attempt + 1, max_retries=self.max_retries, error=str(e))
                self.provider_failure_counts[provider_name][model_name] += 1
                if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name][model_name] = 0
                    logger.error("Circuit opened due to ConnectErrors", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                if attempt == self.max_retries - 1:
                    raise ProviderError(f"ConnectError with {provider_name}/{model_name} after {self.max_retries} attempts: {e}") from e
            except httpx.HTTPStatusError as e:
                logger.warning("HTTPStatusError from provider",
                               provider_name=provider_name, model_name=model_name, attempt=attempt + 1, max_retries=self.max_retries,
                               status_code=e.response.status_code, response_text=e.response.text[:200], error=str(e))
                self.provider_failure_counts[provider_name][model_name] += 1
                if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name][model_name] = 0
                    logger.error("Circuit opened due to HTTPStatusErrors", provider_name=provider_name, model_name=model_name, status_code=e.response.status_code, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                if attempt == self.max_retries - 1:
                    raise ProviderError(f"HTTPStatusError {e.response.status_code} with {provider_name}/{model_name} after {self.max_retries} attempts: {e}") from e
            except httpx.RequestError as e:
                logger.warning("RequestError with provider", provider_name=provider_name, model_name=model_name, attempt=attempt + 1, max_retries=self.max_retries, error=str(e))
                self.provider_failure_counts[provider_name][model_name] += 1
                if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name][model_name] = 0
                    logger.error("Circuit opened due to RequestErrors", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                if attempt == self.max_retries - 1:
                    raise ProviderError(f"RequestError with {provider_name}/{model_name} after {self.max_retries} attempts: {e}") from e

            except RateLimitError:
                logger.warning("RateLimitError from provider, not retrying or opening circuit", provider_name=provider_name, model_name=model_name)
                raise
            except AuthenticationError as e:
                logger.error("AuthenticationError from provider, marking credentials invalid", provider_name=provider_name, model_name=model_name, error=str(e), account_id=credentials.account_id)
                await self.account_manager.mark_credentials_invalid(provider_name, credentials.account_id)
                raise

            except ProviderError as e: # General provider error from the provider's SDK
                logger.warning("ProviderError from provider", provider_name=provider_name, model_name=model_name, attempt=attempt + 1, max_retries=self.max_retries, error=str(e), exc_info=True) # exc_info for better context
                self.provider_failure_counts[provider_name][model_name] += 1
                if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name][model_name] = 0
                    logger.error("Circuit opened due to ProviderErrors", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                if attempt == self.max_retries - 1:
                    raise # Re-raise the original ProviderError
            except asyncio.TimeoutError as e:
                logger.warning("Asyncio TimeoutError", provider_name=provider_name, model_name=model_name, attempt=attempt + 1, max_retries=self.max_retries, error=str(e))
                self.provider_failure_counts[provider_name][model_name] += 1
                if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name][model_name] = 0
                    logger.error("Circuit opened due to asyncio.TimeoutError", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                if attempt == self.max_retries - 1:
                     raise ProviderError(f"Asyncio TimeoutError with {provider_name}/{model_name} after {self.max_retries} attempts: {e}") from e
            except Exception as e: # Catch-all for other unexpected errors
                logger.error("Unexpected error during provider call",
                               provider_name=provider_name, model_name=model_name, attempt=attempt + 1, max_retries=self.max_retries,
                               error=str(e), exc_info=True)
                self.provider_failure_counts[provider_name][model_name] += 1
                if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name][model_name] = 0
                    logger.error("Circuit opened due to unexpected error", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                if attempt == self.max_retries - 1: # If it's the last attempt, raise it as a ProviderError
                    raise ProviderError(f"Unexpected error with {provider_name}/{model_name} after {self.max_retries} attempts: {e}") from e

            # If an error occurred and it's not the last attempt, sleep and retry
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info("Retrying provider call after error", provider_name=provider_name, model_name=model_name, attempt=attempt + 2, max_retries=self.max_retries, delay_seconds=round(delay,2))
                await asyncio.sleep(delay)
        
        return None # Should be unreachable if max_retries > 0, as errors are raised or re-raised.
    
    async def _try_provider_stream(
        self,
        provider_name: str,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Try streaming request with specific provider, including circuit breaker and refined error handling."""

        provider = self.providers.get(provider_name)
        if not provider or not provider.is_available:
            logger.debug("_try_provider_stream: Provider not found or unavailable", provider_name=provider_name)
            return

        # Get credentials for this provider
        credentials = await self.account_manager.get_credentials(provider_name)
        if not credentials:
            logger.warning("No valid credentials for provider for streaming", provider_name=provider_name, model_requested=request.model)
            return

        # Resolve model name if "auto" is specified
        model_name = request.model # Initial model name for circuit key if not "auto"
        if model_name == "auto":
            # Check circuit for "auto" before resolving
            open_until_auto = self.provider_circuit_open_until[provider_name].get("auto")
            if open_until_auto and datetime.utcnow() < open_until_auto:
                logger.warning("Circuit for 'auto' model selection is open for streaming", provider_name=provider_name, open_until=open_until_auto.isoformat())
                return

            resolved_model_name = await self._select_model(provider, request)
            if not resolved_model_name:
                logger.warning("No suitable model found by _select_model for streaming", provider_name=provider_name, original_request_model=request.model)
                self.provider_failure_counts[provider_name]["auto"] += 1
                if self.provider_failure_counts[provider_name]["auto"] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self.provider_circuit_open_until[provider_name]["auto"] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                    self.provider_failure_counts[provider_name]["auto"] = 0
                    logger.error("Circuit for 'auto' model selection opened for streaming", provider_name=provider_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                return
            model_name = resolved_model_name # Actual model name to be used
            logger.info("Model 'auto' resolved for streaming", provider_name=provider_name, resolved_model_name=model_name)
        
        # Check circuit for the specific model_name (either provided or resolved from "auto")
        open_until = self.provider_circuit_open_until[provider_name].get(model_name)
        if open_until and datetime.utcnow() < open_until:
            logger.warning("Circuit open for model for streaming", provider_name=provider_name, model_name=model_name, open_until=open_until.isoformat())
            return

        # Create request with resolved model
        resolved_request = request.copy()
        resolved_request.model = model_name
        
        # Stream response
        try:
            async for chunk in provider.chat_completion_stream(resolved_request, credentials):
                yield chunk

            # Success: Update credentials usage and reset failure count
            await self.account_manager.update_usage(credentials)
            self.provider_failure_counts[provider_name][model_name] = 0
            if self.provider_circuit_open_until[provider_name].get(model_name) and \
               self.provider_circuit_open_until[provider_name][model_name] < datetime.utcnow():
                logger.info("Circuit closed due to successful stream", provider_name=provider_name, model_name=model_name)

        except httpx.ReadTimeout as e:
            logger.warning("ReadTimeout during stream", provider_name=provider_name, model_name=model_name, error=str(e))
            self.provider_failure_counts[provider_name][model_name] += 1
            if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                self.provider_failure_counts[provider_name][model_name] = 0
                logger.error("Circuit opened for stream due to ReadTimeouts", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            raise ProviderError(f"ReadTimeout during stream with {provider_name}/{model_name}: {e}") from e
        except httpx.ConnectError as e:
            logger.warning("ConnectError during stream", provider_name=provider_name, model_name=model_name, error=str(e))
            self.provider_failure_counts[provider_name][model_name] += 1
            if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                self.provider_failure_counts[provider_name][model_name] = 0
                logger.error("Circuit opened for stream due to ConnectErrors", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            raise ProviderError(f"ConnectError during stream with {provider_name}/{model_name}: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.warning("HTTPStatusError during stream",
                           provider_name=provider_name, model_name=model_name, status_code=e.response.status_code,
                           response_text=e.response.text[:200], error=str(e))
            self.provider_failure_counts[provider_name][model_name] += 1
            if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                self.provider_failure_counts[provider_name][model_name] = 0
                logger.error("Circuit opened for stream due to HTTPStatusErrors", provider_name=provider_name, model_name=model_name, status_code=e.response.status_code, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            raise ProviderError(f"HTTPStatusError {e.response.status_code} during stream with {provider_name}/{model_name}: {e}") from e
        except httpx.RequestError as e:
            logger.warning("RequestError during stream", provider_name=provider_name, model_name=model_name, error=str(e))
            self.provider_failure_counts[provider_name][model_name] += 1
            if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                self.provider_failure_counts[provider_name][model_name] = 0
                logger.error("Circuit opened for stream due to RequestErrors", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            raise ProviderError(f"RequestError during stream with {provider_name}/{model_name}: {e}") from e

        except RateLimitError as e:
            logger.warning("RateLimitError during stream, not opening circuit", provider_name=provider_name, model_name=model_name, error=str(e))
            raise
        except AuthenticationError as e:
            logger.error("AuthenticationError during stream, marking credentials invalid", provider_name=provider_name, model_name=model_name, error=str(e), account_id=credentials.account_id)
            await self.account_manager.mark_credentials_invalid(provider_name, credentials.account_id)
            raise
        except ProviderError as e:
            logger.warning("ProviderError during stream", provider_name=provider_name, model_name=model_name, error=str(e), exc_info=True)
            self.provider_failure_counts[provider_name][model_name] += 1
            if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                self.provider_failure_counts[provider_name][model_name] = 0
                logger.error("Circuit opened for stream due to ProviderErrors", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            raise
        except asyncio.TimeoutError as e:
            logger.warning("Asyncio TimeoutError during stream", provider_name=provider_name, model_name=model_name, error=str(e))
            self.provider_failure_counts[provider_name][model_name] += 1
            if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                self.provider_failure_counts[provider_name][model_name] = 0
                logger.error("Circuit opened for stream due to asyncio.TimeoutError", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            raise ProviderError(f"Asyncio TimeoutError during stream with {provider_name}/{model_name}: {e}") from e
        except Exception as e: # Catch-all for other unexpected errors
            logger.error("Unexpected error during stream",
                           provider_name=provider_name, model_name=model_name, error=str(e), exc_info=True)
            self.provider_failure_counts[provider_name][model_name] += 1
            if self.provider_failure_counts[provider_name][model_name] >= self.CIRCUIT_BREAKER_THRESHOLD:
                self.provider_circuit_open_until[provider_name][model_name] = datetime.utcnow() + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
                self.provider_failure_counts[provider_name][model_name] = 0
                logger.error("Circuit opened for stream due to unexpected error", provider_name=provider_name, model_name=model_name, cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            raise ProviderError(f"Unexpected error during stream with {provider_name} for model {model_name}: {e}") from e
    
    async def _select_model(
        self,
        provider: BaseProvider,
        request: ChatCompletionRequest
    ) -> Optional[str]:
        """Select best model for request from provider's available models."""
        
        available_models = provider.list_models()
        if not available_models:
            return None
        
        # Simple model selection logic - can be enhanced
        # Prefer free models, then by capability match
        
        # Determine required capabilities from request
        required_capabilities = self._infer_capabilities(request)
        
        # Score models based on capability match and other factors
        scored_models = []
        for model in available_models:
            score = 0
            
            # Prefer free models
            if model.is_free:
                score += 100
            
            # Score based on capability match
            matching_caps = set(model.capabilities) & set(required_capabilities)
            score += len(matching_caps) * 10
            
            # Prefer larger context windows
            if model.context_length:
                score += min(model.context_length / 1000, 50)  # Cap at 50 points
            
            scored_models.append((score, model))
        
        # Sort by score and return best model
        scored_models.sort(key=lambda x: x[0], reverse=True)
        return scored_models[0][1].name if scored_models else None
    
    def _infer_capabilities(self, request: ChatCompletionRequest) -> List[ModelCapability]:
        """Infer required capabilities from request."""
        capabilities = [ModelCapability.TEXT_GENERATION]
        
        # Analyze message content for capability hints
        content = " ".join(msg.content.lower() for msg in request.messages)
        
        if any(keyword in content for keyword in ["code", "python", "javascript", "programming"]):
            capabilities.append(ModelCapability.CODE_GENERATION)
        
        if any(keyword in content for keyword in ["think", "reason", "solve", "analyze"]):
            capabilities.append(ModelCapability.REASONING)
        
        return capabilities
    
    async def list_available_models(self) -> Dict[str, List[ModelInfo]]:
        """
        Lists all available models grouped by their provider.

        Only models from providers that are currently marked as available
        are included.

        Returns:
            A dictionary where keys are provider names and values are lists
            of ModelInfo objects for that provider.
        """
        models_by_provider: Dict[str, List[ModelInfo]] = {}
        
        for provider_name, provider in self.providers.items():
            if provider.is_available:
                models_by_provider[provider_name] = provider.list_models()
        
        return models_by_provider
    
    async def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves the current status of all configured providers.

        This includes availability, configuration status, metrics, model counts,
        and credential counts.

        Returns:
            A dictionary where keys are provider names and values are dictionaries
            containing status information for that provider.
        """
        status: Dict[str, Dict[str, Any]] = {}
        
        for provider_name, provider in self.providers.items():
            status[provider_name] = {
                "available": provider.is_available,
                "status": provider.config.status.value,
                "metrics": provider.metrics.dict(),
                "models_count": len(provider.config.models),
                "credentials_count": len(provider.credentials)
            }
        
        return status
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Performs a health check on all available providers.

        This typically involves making a lightweight test call to each provider's API
        to ensure it's responsive.

        Returns:
            A dictionary where keys are provider names and values are booleans
            indicating whether the provider is healthy (True) or not (False).
        """
        results: Dict[str, bool] = {}
        
        tasks: List[asyncio.Task] = []
        for provider_name, provider in self.providers.items():
            if provider.is_available:
                tasks.append(self._provider_health_check(provider_name, provider))
        
        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(health_results):
                provider_name = list(self.providers.keys())[i]
                results[provider_name] = isinstance(result, bool) and result
        
        return results
    
    async def _provider_health_check(self, provider_name: str, provider: BaseProvider) -> bool: # provider object passed for direct call
        """Perform health check on single provider."""
        try:
            is_healthy = await provider.health_check() # This method within provider should ideally use structlog too
            logger.info("Provider health check status", provider_name=provider_name, is_healthy=is_healthy)
            return is_healthy
        except Exception as e:
            logger.error("Health check failed for provider", provider_name=provider_name, error=str(e), exc_info=True)
            return False
    
    async def get_meta_controller_insights(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves insights from the meta-controller, if enabled.

        This can include information about model performance, usage patterns,
        and recommendations for model selection.

        Returns:
            A dictionary containing insights from the meta-controller, or None
            if the meta-controller is not enabled or has no insights.
        """
        
        if not self.meta_controller:
            logger.debug("Meta-controller not enabled, cannot get insights.")
            return None
        
        return self.meta_controller.get_model_insights()
    
    async def get_ensemble_insights(self, request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
        """
        Retrieves insights about the ensemble decision process for a given request, if enabled.

        Args:
            request: The ChatCompletionRequest for which to get ensemble insights.

        Returns:
            A dictionary containing insights about the ensemble process, or None
            if the ensemble system or meta-controller is not enabled.
        """
        
        if not self.ensemble_system or not self.meta_controller:
            logger.debug("Ensemble system or meta-controller not enabled, cannot get ensemble insights.")
            return None
        
        # Get candidate models
        cascade_chain = await self.meta_controller.get_cascade_chain(request)
        
        # Create mock candidates for analysis
        from .ensemble_system import ResponseCandidate
        candidates = []
        
        for model_name in cascade_chain[:3]:
            # Find provider for this model
            provider_name = None
            for prov_name, provider in self.providers.items():
                if any(model.name == model_name for model in provider.list_models()):
                    provider_name = prov_name
                    break
            
            if provider_name and model_name in self.meta_controller.model_profiles:
                profile = self.meta_controller.model_profiles[model_name]
                
                # Create mock candidate
                candidate = ResponseCandidate(
                    model_name=model_name,
                    provider=provider_name,
                    response=None,  # Mock response
                    confidence_score=profile.reliability_score,
                    response_time=profile.avg_response_time,
                    cost_estimate=profile.cost_per_token,
                    coherence_score=profile.instruction_following,
                    relevance_score=profile.factual_knowledge,
                    factual_accuracy_score=profile.factual_knowledge,
                    creativity_score=profile.creative_writing,
                    safety_score=0.8  # Default safety score
                )
                candidates.append(candidate)
        
        return self.ensemble_system.get_ensemble_insights(candidates)
    
    async def analyze_task_complexity(self, request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
        """
        Analyzes the complexity of a given task request using the meta-controller.

        Args:
            request: The ChatCompletionRequest to analyze.

        Returns:
            A dictionary containing various complexity scores and metrics, or None
            if the meta-controller is not enabled.
        """
        
        if not self.meta_controller or not self.meta_controller.complexity_analyzer: # Added check for complexity_analyzer
            logger.debug("Meta-controller or complexity analyzer not enabled, cannot analyze task complexity.")
            return None
        
        complexity = self.meta_controller.complexity_analyzer.analyze_task_complexity(request) # Assuming this is not async
        
        return {
            'reasoning_depth': complexity.reasoning_depth,
            'domain_specificity': complexity.domain_specificity,
            'context_length': complexity.context_length,
            'computational_intensity': complexity.computational_intensity,
            'creativity_required': complexity.creativity_required,
            'factual_accuracy_importance': complexity.factual_accuracy_importance,
            'overall_complexity': complexity.overall_complexity
        }
    
    async def get_model_recommendations(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Provides model recommendations for a given request.

        This includes recommendations from traditional routing, meta-controller insights,
        ensemble insights, and task complexity analysis, if available.

        Args:
            request: The ChatCompletionRequest for which to get recommendations.

        Returns:
            A dictionary containing various model recommendations and insights.
        """
        
        recommendations: Dict[str, Any] = {
            'traditional_routing_chain': await self._get_provider_chain(request), # Renamed for clarity
            'meta_controller_insights': None,
            'ensemble_insights': None,
            'task_complexity': None
        }
        
        if self.meta_controller:
            # Get meta-controller recommendation
            optimal_model, confidence = await self.meta_controller.select_optimal_model(request)
            cascade_chain = await self.meta_controller.get_cascade_chain(request)
            
            recommendations['meta_controller_insights'] = {
                'optimal_model': optimal_model,
                'confidence': confidence,
                'cascade_chain': cascade_chain,
                'model_insights': await self.get_meta_controller_insights()
            }
            
            # Get task complexity analysis
            recommendations['task_complexity'] = await self.analyze_task_complexity(request)
        
        if self.ensemble_system:
            # Get ensemble insights
            recommendations['ensemble_insights'] = await self.get_ensemble_insights(request)
        
        return recommendations
    
    async def _start_auto_updater(self):
        """Start the auto-updater background task."""
        if not self.auto_updater:
            logger.debug("Auto-updater is not enabled, skipping start.")
            return
        
        try:
            logger.info("Integrating and starting auto-updater...")
            # Integrate auto-updater with this aggregator
            await integrate_auto_updater(self, self.auto_updater) # This function should also be updated if it logs
            
            # Start the auto-update loop
            await self.auto_updater.start_auto_update(self.auto_update_interval) # This method should also use structlog
            logger.info("Auto-updater started.", update_interval_minutes=self.auto_update_interval)
            
        except Exception as e:
            logger.error("Error starting auto-updater", error=str(e), exc_info=True)
    
    async def force_update_providers(self) -> Dict[str, Any]:
        """
        Forces an immediate update of all provider information via the auto-updater.

        This is useful for manually triggering an update outside of the scheduled interval.

        Returns:
            A dictionary containing the status of the update operation, including
            any updates found, or an error message if the auto-updater is not enabled
            or an error occurs.
        """
        if not self.auto_updater:
            logger.warning("Auto-updater not enabled, cannot force update.")
            return {"error": "Auto-updater not enabled"}
        
        logger.info("Forcing update of all provider information.")
        try:
            updates = await self.auto_updater.force_update_all()
            
            return {
                "status": "success",
                "updates_found": len(updates),
                "updates": [
                    {
                        "provider": update.provider_name,
                        "models_added": len(update.models_added),
                        "models_removed": len(update.models_removed),
                        "models_updated": len(update.models_updated),
                        "rate_limits_updated": bool(update.rate_limits_updated),
                        "timestamp": update.timestamp.isoformat() if update.timestamp else None
                    }
                    for update in updates
                ]
            }
            
        except Exception as e:
            logger.error("Error forcing provider updates", error=str(e), exc_info=True)
            return {"error": str(e)}
    
    async def get_auto_update_status(self) -> Dict[str, Any]:
        """
        Retrieves the current status of the auto-updater.

        Returns:
            A dictionary containing the auto-updater's status (e.g., enabled,
            last update time, next update time), or an error message if an error occurs.
            If auto-updater is not enabled, returns `{"enabled": False}`.
        """
        if not self.auto_updater:
            logger.debug("Auto-updater not enabled, status check returning disabled.")
            return {"enabled": False}
        
        logger.debug("Getting auto-updater status.")
        try:
            status = await self.auto_updater.get_update_status()
            status["enabled"] = True
            return status
            
        except Exception as e:
            logger.error("Error getting auto-update status", error=str(e), exc_info=True)
            return {"enabled": True, "error": str(e)}
    
    async def configure_auto_updater(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configures settings for the auto-updater.

        Allows updating parameters like the update interval or enabling/disabling
        specific update sources.

        Args:
            config: A dictionary containing configuration options to apply.
                    Expected keys might include 'update_interval' (int, minutes)
                    or 'sources' (list of source configurations).

        Returns:
            A dictionary indicating the success or failure of the configuration update.
        """
        if not self.auto_updater:
            logger.warning("Auto-updater not enabled, cannot configure.")
            return {"error": "Auto-updater not enabled"}
        
        logger.info("Configuring auto-updater.", new_config=config)
        try:
            # Update interval
            if "update_interval" in config:
                self.auto_update_interval = config["update_interval"]
            
            # Enable/disable specific sources
            if "sources" in config:
                for source_config in config["sources"]:
                    source_name = source_config.get("name")
                    if source_name:
                        # Find and update source
                        for source in self.auto_updater.sources:
                            if source.name == source_name:
                                for key, value in source_config.items():
                                    if hasattr(source, key):
                                        setattr(source, key, value)
                                break
            
            # Save updated configuration
            self.auto_updater._save_update_sources(self.auto_updater.sources)
            
            return {"status": "success", "message": "Auto-updater configuration updated"}
            
        except Exception as e:
            logger.error("Error configuring auto-updater", config_options=config, error=str(e), exc_info=True)
            return {"error": str(e)}
    
    async def get_provider_updates_history(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves the history of provider updates from the auto-updater.

        This typically shows cached information about models and when they were last updated.
        This method's utility depends on the auto-updater's caching implementation.

        Args:
            provider_name: Optional. If provided, filters history for a specific provider.

        Returns:
            A dictionary containing update history information, or an error message
            if the auto-updater is not enabled or an error occurs.
        """
        if not self.auto_updater:
            logger.warning("Auto-updater not enabled, cannot get update history.")
            return {"error": "Auto-updater not enabled"}
        
        logger.debug("Getting provider updates history.", provider_filter=provider_name)
        try:
            # This would require storing update history
            # For now, return cached data
            if provider_name:
                cached_data = self.auto_updater.cache.get(f"api_{provider_name}_models")
                if cached_data:
                    return {
                        "provider": provider_name,
                        "cached_models": len(cached_data),
                        "last_update": "Available in cache"
                    }
                else:
                    return {"provider": provider_name, "error": "No cached data"}
            else:
                # Return summary for all providers
                summary = {}
                for key, value in self.auto_updater.cache.items():
                    if key.startswith("api_") and key.endswith("_models"):
                        provider = key.replace("api_", "").replace("_models", "")
                        summary[provider] = {
                            "cached_models": len(value) if isinstance(value, list) else 0
                        }
                
                return {"providers": summary}
                
        except Exception as e:
            logger.error("Error getting provider updates history", provider_name_filter=provider_name, error=str(e), exc_info=True)
            return {"error": str(e)}
    
    async def close(self) -> None:
        """
        Closes all provider connections and cleans up resources.

        This method should be called during application shutdown to ensure graceful
        termination of provider clients and any background tasks like the auto-updater.
        """
        logger.info("Closing LLM Aggregator and its components...")
        tasks: List[asyncio.Task] = []
        for provider_name, provider in self.providers.items(): # Iterate with name for logging
            logger.debug("Adding provider to close queue", provider_name=provider_name)
            tasks.append(provider.close())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close auto-updater
        if self.auto_updater:
            await self.auto_updater.close() # This method should also use structlog
        
        logger.info("LLM Aggregator closed successfully.")