"""
FastAPI server for the LLM API Aggregator.
"""

import asyncio
import logging
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

# Imports for CodeMasterAgent and its models
import uuid # For generating request_id
from pathlib import Path
import sys
# Adjust sys.path to allow importing from project root for new_code_models
# and openhands_2_0 sibling directory.
# This assumes server.py is in src/api/server.py
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from openhands_2_0.core.agent_swarm.agents.codemaster_agent import CodeMasterAgent
from new_code_models import CodeExecutionRequest, CodeExecutionResponse, CodeTaskType


logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Security configuration using centralized settings
# ADMIN_TOKEN and ALLOWED_ORIGINS are now accessed via settings object

if not settings.ADMIN_TOKEN:
    logger.warning("ADMIN_TOKEN not set in environment or .env file. Admin endpoints will be disabled if called.")

# Global aggregator instance
aggregator: Optional[LLMAggregator] = None
code_master_agent: Optional[CodeMasterAgent] = None # Added for CodeMasterAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global aggregator, code_master_agent # Added code_master_agent
    
    # Startup
    logger.info("Starting LLM API Aggregator and OpenHands Agents...")
    
    # Initialize components for LLM Aggregator
    account_manager = AccountManager()
    rate_limiter = RateLimiter()
    providers = []
    openrouter = create_openrouter_provider([])
    groq = create_groq_provider([])
    cerebras = create_cerebras_provider([])
    providers.extend([openrouter, groq, cerebras])
    provider_configs = {provider.name: provider.config for provider in providers}
    router = ProviderRouter(provider_configs)
    aggregator = LLMAggregator(
        providers=providers,
        account_manager=account_manager,
        router=router,
        rate_limiter=rate_limiter
    )
    
    # Initialize CodeMasterAgent
    code_master_agent = CodeMasterAgent()

    # Gather async initializations
    # Note: LLMAggregator itself doesn't have an async init in this snippet
    # If it did, it would be: await aggregator.initialize() or similar
    init_tasks = [code_master_agent.initialize()]
    # Add other agent initializations here if needed:
    # tasks_to_gather.append(other_agent.initialize())

    await asyncio.gather(*init_tasks)

    logger.info("LLM API Aggregator and OpenHands Agents started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM API Aggregator and OpenHands Agents...")
    if aggregator:
        await aggregator.close()
    # Add shutdown for code_master_agent if it has one:
    # if code_master_agent and hasattr(code_master_agent, 'shutdown'):
    #     await code_master_agent.shutdown()
    logger.info("LLM API Aggregator and OpenHands Agents shut down")


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


async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify admin token for admin endpoints."""
    if not settings.ADMIN_TOKEN:
        logger.error("Attempt to access admin endpoint, but ADMIN_TOKEN is not configured.")
        raise HTTPException(status_code=503, detail="Admin functionality is not configured or disabled.")
    
    if not credentials or credentials.credentials != settings.ADMIN_TOKEN:
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
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models across all providers."""
    
    if not aggregator:
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
        logger.error(f"List models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/credentials")
async def add_credentials(
    provider: str,
    account_id: str,
    api_key: str,
    additional_headers: Optional[Dict[str, str]] = None,
    _admin_token: str = Depends(verify_admin_token)
):
    """Add API credentials for a provider."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        credentials = await aggregator.account_manager.add_credentials(
            provider=provider,
            account_id=account_id,
            api_key=api_key,
            additional_headers=additional_headers
        )
        
        return {
            "message": f"Credentials added for {provider}:{account_id}",
            "provider": provider,
            "account_id": account_id
        }
        
    except Exception as e:
        logger.error(f"Add credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/credentials")
async def list_credentials(_admin_token: str = Depends(verify_admin_token)):
    """List all credentials (without sensitive data)."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        credentials = await aggregator.account_manager.list_credentials()
        return credentials
        
    except Exception as e:
        logger.error(f"List credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/credentials/{provider}/{account_id}")
async def remove_credentials(provider: str, account_id: str, _admin_token: str = Depends(verify_admin_token)):
    """Remove credentials for a specific account."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        removed = await aggregator.account_manager.remove_credentials(provider, account_id)
        
        if removed:
            return {"message": f"Credentials removed for {provider}:{account_id}"}
        else:
            raise HTTPException(status_code=404, detail="Credentials not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/providers")
async def get_provider_status(_admin_token: str = Depends(verify_admin_token)):
    """Get status of all providers."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        status = await aggregator.get_provider_status()
        return status
        
    except Exception as e:
        logger.error(f"Get provider status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/rate-limits")
async def get_rate_limit_status(user_id: Optional[str] = Depends(get_user_id), _admin_token: str = Depends(verify_admin_token)):
    """Get rate limit status."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        status = aggregator.rate_limiter.get_rate_limit_status(user_id)
        return status
        
    except Exception as e:
        logger.error(f"Get rate limit status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/usage-stats")
async def get_usage_stats(_admin_token: str = Depends(verify_admin_token)):
    """Get usage statistics."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
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
        logger.error(f"Get usage stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/rotate-credentials/{provider}")
async def rotate_credentials(provider: str, _admin_token: str = Depends(verify_admin_token)):
    """Rotate credentials for a provider."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        await aggregator.account_manager.rotate_credentials(provider)
        return {"message": f"Credentials rotated for {provider}"}
        
    except Exception as e:
        logger.error(f"Rotate credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the server."""
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.LOG_LEVEL.lower() # Use settings for log level
    )


if __name__ == "__main__":
    # Basic logging configuration for startup messages
    logging.basicConfig(level=settings.LOG_LEVEL.upper())
    run_server()

# New endpoint for CodeMasterAgent
@app.post("/v1/code/execute", response_model=CodeExecutionResponse, tags=["Code Execution"])
async def execute_code_task(
    request: CodeExecutionRequest,
    # user_id: Optional[str] = Depends(get_user_id) # User context can be added if needed
):
    """
    Executes a code-related task (generate, analyze, refactor, etc.) using the CodeMasterAgent.
    """
    global code_master_agent
    if not code_master_agent:
        logger.error("CodeMasterAgent not initialized during server startup.")
        raise HTTPException(status_code=503, detail="CodeMasterAgent not available. Service not ready.")

    # Prepare input for the CodeMasterAgent
    # CodeMasterAgent's execute expects 'input_data' (a dict) and 'context'.
    # The 'text' field within input_data is often the primary prompt.
    input_data_for_agent = {
        "text": request.user_prompt, # Main instruction
        "code": request.code_snippet,
        "language": request.language,
        "task_type_hint": request.task_type.value, # Pass the enum value to help agent determine task
        "dependencies": request.dependencies # For generation tasks primarily
    }

    request_context = request.context or {}

    try:
        agent_response = await code_master_agent.execute(
            input_data=input_data_for_agent,
            context=request_context
        )
    except Exception as e:
        logger.exception(f"CodeMasterAgent execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Error during code task execution: {str(e)}")

    req_id = uuid.uuid4().hex

    if agent_response.get("success"):
        agent_output = agent_response.get("output", {})
        # Map agent_output fields to CodeExecutionResponse fields
        # Handle cases where some fields might be None or under different keys in agent_output
        return CodeExecutionResponse(
            request_id=req_id,
            task_type=request.task_type,
            status="completed",
            output_code=agent_output.get("code"),
            explanation=agent_output.get("explanation") or agent_output.get("summary") or agent_output.get("analysis_summary"),
            # Consolidate various possible analysis keys from CodeMasterAgent general tasks
            analysis=agent_output.get("analysis") or agent_output.get("structure_analysis") or agent_output.get("review_summary") or agent_output.get("details"),
            self_reflection=agent_output.get("self_reflection"),
            reasoning_log=agent_output.get("reasoning_log"),
            error_message=None
        )
    else:
        # Attempt to get reflection and reasoning even from failed agent responses if available
        agent_output_on_failure = agent_response.get("output", {})
        return CodeExecutionResponse(
            request_id=req_id,
            task_type=request.task_type,
            status="failed",
            error_message=agent_response.get("error", "Unknown error from CodeMasterAgent"),
            self_reflection=agent_output_on_failure.get("self_reflection"),
            reasoning_log=agent_output_on_failure.get("reasoning_log")
        )