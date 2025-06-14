"""
FastAPI server for the LLM API Aggregator.
"""

# Initialize logging early
from src.config.logging_config import setup_logging
setup_logging()

import asyncio
import structlog # Changed from logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from src.config import settings # Centralized settings
# Ensure setup_logging is called before any loggers are instantiated by other modules if they also use structlog.
# For stdlib logging, structlog's configuration will take over.

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    AccountCredentials,
    ProviderConfig,
    SystemConfig
)
from ..core.aggregator import LLMAggregator
from ..core.account_manager import AccountManager
from ..core.router import ProviderRouter
from ..core.rate_limiter import RateLimiter, RateLimitExceeded
from ..providers.openrouter import create_openrouter_provider
from ..providers.groq import create_groq_provider
from ..providers.cerebras import create_cerebras_provider


logger = structlog.get_logger(__name__) # Changed from logging
security = HTTPBearer(auto_error=False)

# Security configuration using centralized settings
# ADMIN_TOKEN and ALLOWED_ORIGINS are now accessed via settings object

if not settings.ADMIN_TOKEN:
    logger.warning("ADMIN_TOKEN not set. Admin endpoints may be disabled if called.", admin_token_present=False)

# Global aggregator instance
aggregator: Optional[LLMAggregator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global aggregator
    
    # Startup
    logger.info("Starting LLM API Aggregator server...") # Changed message slightly for clarity
    
    # Initialize components
    # AccountManager will now use OPENHANDS_ENCRYPTION_KEY from settings
    account_manager = AccountManager()
    rate_limiter = RateLimiter()
    
    # Create providers
    providers = []
    
    # Initialize providers with empty credentials (will be added via API)
    openrouter = create_openrouter_provider([])
    groq = create_groq_provider([])
    cerebras = create_cerebras_provider([])
    
    providers.extend([openrouter, groq, cerebras])
    
    # Create provider configs dict
    provider_configs = {provider.name: provider.config for provider in providers}
    
    # Initialize router
    router = ProviderRouter(provider_configs)
    
    # Initialize aggregator
    aggregator = LLMAggregator(
        providers=providers,
        account_manager=account_manager,
        router=router,
        rate_limiter=rate_limiter
    )
    
    logger.info("LLM API Aggregator started successfully.") # Added a period
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM API Aggregator server...") # Changed message slightly for clarity
    if aggregator:
        await aggregator.close() # This close method should also use structlog if it logs
    logger.info("LLM API Aggregator shut down successfully.") # Added a period


# Create FastAPI app
app = FastAPI(
    title="LLM API Aggregator",
    description="Multi-provider LLM API with intelligent routing and account management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS, # Use settings
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


def get_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Extract user ID from authorization header."""
    if credentials:
        return credentials.credentials  # Use token as user ID for simplicity
    return None


async def verify_admin_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify admin token for admin endpoints."""
    if not settings.ADMIN_TOKEN:
        logger.error("Admin token not configured", endpoint=str(request.url), admin_token_present=False)
        raise HTTPException(status_code=503, detail="Admin functionality is not configured or disabled.")
    
    if not credentials or credentials.credentials != settings.ADMIN_TOKEN:
        logger.warning("Invalid admin token received", endpoint=str(request.url), client_host=request.client.host if request.client else "unknown")
        raise HTTPException(status_code=401, detail="Invalid admin token")
    
    return credentials.credentials


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM API Aggregator",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    provider_health = await aggregator.health_check()
    
    return {
        "status": "healthy",
        "providers": provider_health,
        "timestamp": time.time()
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(get_user_id)
):
    """OpenAI-compatible chat completions endpoint."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in aggregator.chat_completion_stream(request, user_id):
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Return regular response
            response = await aggregator.chat_completion(request, user_id)
            return response
            
    except RateLimitExceeded as e:
        logger.warning("Rate limit exceeded for chat completion", user_id=user_id, error=str(e), request_model=request.model)
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error("Chat completion error", error=str(e), user_id=user_id, request_model=request.model, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models(request_obj: Request): # Added Request for logging context
    """List available models across all providers."""
    
    if not aggregator:
        logger.warning("List models endpoint called but aggregator not ready", endpoint=str(request_obj.url))
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        models_by_provider = await aggregator.list_available_models()
        
        # Flatten into OpenAI-compatible format
        models = []
        for provider_name, provider_models in models_by_provider.items():
            for model in provider_models:
                models.append({
                    "id": f"{provider_name}/{model.name}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider_name,
                    "permission": [],
                    "root": model.name,
                    "parent": None,
                    "display_name": model.display_name,
                    "capabilities": [cap.value for cap in model.capabilities],
                    "context_length": model.context_length,
                    "is_free": model.is_free
                })
        
        return {"object": "list", "data": models}
        
    except Exception as e:
        logger.error("List models error", error=str(e), endpoint=str(request_obj.url), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/credentials")
async def add_credentials(
    request_obj: Request, # Added Request for logging context
    provider: str,
    account_id: str,
    api_key: str, # Consider not logging API key directly, even its presence
    additional_headers: Optional[Dict[str, str]] = None,
    _admin_token: str = Depends(verify_admin_token)
):
    """Add API credentials for a provider."""
    
    if not aggregator:
        logger.warning("Add credentials endpoint called but aggregator not ready", endpoint=str(request_obj.url))
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Note: API key itself is not logged for security, but its presence could be if needed.
        logger.info("Attempting to add credentials", provider=provider, account_id=account_id, has_additional_headers=bool(additional_headers), admin_user=_admin_token) # Assuming admin_token is username or similar
        created_credentials = await aggregator.account_manager.add_credentials(
            provider=provider,
            account_id=account_id,
            api_key=api_key,
            additional_headers=additional_headers
        )
        
        return {
            "message": f"Credentials successfully added for {provider}:{account_id}", # Added "successfully"
            "provider": provider,
            "account_id": account_id
        }
        
    except Exception as e:
        logger.error("Add credentials error", error=str(e), provider=provider, account_id=account_id, admin_user=_admin_token, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/credentials")
async def list_credentials(request_obj: Request, _admin_token: str = Depends(verify_admin_token)): # Added Request
    """List all credentials (without sensitive data)."""
    
    if not aggregator:
        logger.warning("List credentials endpoint called but aggregator not ready", endpoint=str(request_obj.url), admin_user=_admin_token)
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info("Listing credentials", admin_user=_admin_token)
        credentials_list = await aggregator.account_manager.list_credentials()
        return credentials_list
        
    except Exception as e:
        logger.error("List credentials error", error=str(e), admin_user=_admin_token, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/credentials/{provider}/{account_id}")
async def remove_credentials(request_obj: Request, provider: str, account_id: str, _admin_token: str = Depends(verify_admin_token)): # Added Request
    """Remove credentials for a specific account."""
    
    if not aggregator:
        logger.warning("Remove credentials endpoint called but aggregator not ready", endpoint=str(request_obj.url), admin_user=_admin_token)
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info("Attempting to remove credentials", provider=provider, account_id=account_id, admin_user=_admin_token)
        removed = await aggregator.account_manager.remove_credentials(provider, account_id)
        
        if removed:
            logger.info("Successfully removed credentials", provider=provider, account_id=account_id, admin_user=_admin_token)
            return {"message": f"Credentials removed for {provider}:{account_id}"}
        else:
            logger.warning("Credentials not found for removal", provider=provider, account_id=account_id, admin_user=_admin_token)
            raise HTTPException(status_code=404, detail="Credentials not found")
            
    except HTTPException: # Re-raise HTTPException directly to preserve its details
        raise
    except Exception as e:
        logger.error("Remove credentials error", error=str(e), provider=provider, account_id=account_id, admin_user=_admin_token, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/providers")
async def get_provider_status(request_obj: Request, _admin_token: str = Depends(verify_admin_token)): # Added Request
    """Get status of all providers."""
    
    if not aggregator:
        logger.warning("Get provider status endpoint called but aggregator not ready", endpoint=str(request_obj.url), admin_user=_admin_token)
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info("Getting provider status", admin_user=_admin_token)
        status_info = await aggregator.get_provider_status()
        return status_info
        
    except Exception as e:
        logger.error("Get provider status error", error=str(e), admin_user=_admin_token, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/rate-limits")
async def get_rate_limit_status(request_obj: Request, user_id: Optional[str] = Depends(get_user_id), _admin_token: str = Depends(verify_admin_token)): # Added Request
    """Get rate limit status."""
    
    if not aggregator:
        logger.warning("Get rate limit status endpoint called but aggregator not ready", endpoint=str(request_obj.url), admin_user=_admin_token)
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info("Getting rate limit status", user_id_param=user_id, admin_user=_admin_token) # Renamed user_id to avoid conflict
        status_info = aggregator.rate_limiter.get_rate_limit_status(user_id)
        return status_info
        
    except Exception as e:
        logger.error("Get rate limit status error", error=str(e), user_id_param=user_id, admin_user=_admin_token, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/usage-stats")
async def get_usage_stats(request_obj: Request, _admin_token: str = Depends(verify_admin_token)): # Added Request
    """Get usage statistics."""
    
    if not aggregator:
        logger.warning("Get usage stats endpoint called but aggregator not ready", endpoint=str(request_obj.url), admin_user=_admin_token)
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info("Getting usage statistics", admin_user=_admin_token)
        # Get account usage stats
        account_stats = await aggregator.account_manager.get_usage_stats()
        
        # Get rate limiter stats
        rate_limit_stats = aggregator.rate_limiter.get_user_stats()
        
        # Get provider scores
        provider_scores = aggregator.router.get_provider_scores()
        
        return {
            "account_usage": account_stats,
            "rate_limits": rate_limit_stats,
            "provider_scores": provider_scores,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Get usage stats error", error=str(e), admin_user=_admin_token, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/rotate-credentials/{provider}")
async def rotate_credentials(request_obj: Request, provider: str, _admin_token: str = Depends(verify_admin_token)): # Added Request
    """Rotate credentials for a provider."""
    
    if not aggregator:
        logger.warning("Rotate credentials endpoint called but aggregator not ready", endpoint=str(request_obj.url), admin_user=_admin_token)
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        logger.info("Attempting to rotate credentials", provider=provider, admin_user=_admin_token)
        await aggregator.account_manager.rotate_credentials(provider)
        logger.info("Successfully rotated credentials", provider=provider, admin_user=_admin_token)
        return {"message": f"Credentials rotated for {provider}"}
        
    except Exception as e:
        logger.error("Rotate credentials error", error=str(e), provider=provider, admin_user=_admin_token, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the server."""
    # Uvicorn will use the structured logging if setup_logging() configures the root logger effectively.
    # The log_level here for uvicorn itself might be separate from application log level set by structlog.
    # structlog's setup should handle the application's log level.
    uvicorn.run(
        "src.api.server:app", # app object should be recognized
        host=host,
        port=port,
        reload=reload,
        log_level=settings.LOG_LEVEL.lower()
        # Consider uvicorn's own logging configuration if more control over server logs vs app logs is needed.
        # For now, structlog setup is global.
    )


if __name__ == "__main__":
    # setup_logging() is called at the top of the file, so basicConfig is no longer needed here.
    # logging.basicConfig(level=settings.LOG_LEVEL.upper()) # Removed
    logger.info("Starting server directly via __main__.")
    run_server()