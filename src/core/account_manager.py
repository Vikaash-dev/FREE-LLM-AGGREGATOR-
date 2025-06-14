"""
Account and credentials management system.
"""

import asyncio
import structlog # Changed from logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from cryptography.fernet import Fernet
import json
import os
from src.config import settings

from ..models import AccountCredentials


logger = structlog.get_logger(__name__) # Changed from logging


class AccountManager:
    """
    Manages API credentials and accounts for multiple LLM providers.

    This class handles the storage, encryption, loading, and retrieval of
    API credentials. It supports loading credentials from a local encrypted
    JSON file and/or from environment variables. It also tracks basic usage
    of credentials to allow for round-robin selection.
    """
    
    def __init__(self, storage_path: str = "credentials.json"):
        """
        Initializes the AccountManager.

        Args:
            storage_path: Path to the JSON file used for storing encrypted
                          credentials. Defaults to "credentials.json".
        """
        self.storage_path: str = storage_path
        self.credentials: Dict[str, List[AccountCredentials]] = {}
        self.usage_tracking: Dict[str, Dict[str, int]] = {}
        
        # Initialize encryption
        if settings.OPENHANDS_ENCRYPTION_KEY:
            self.cipher = Fernet(settings.OPENHANDS_ENCRYPTION_KEY.encode())
        else:
            # Generate a new key if none provided
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            logger.critical("OPENHANDS_ENCRYPTION_KEY not set, new key generated.",
                            generated_key_part=key.decode()[:8] + "...", # Log a part of the key for identification if needed, carefully
                            message="For persistent encrypted credentials, please set OPENHANDS_ENCRYPTION_KEY environment variable. "
                                    "Without it, credentials in credentials.json may not be reloadable across sessions.")
        
        # Load existing credentials
        asyncio.create_task(self.load_credentials())
    
    async def add_credentials(
        self,
        provider: str,
        account_id: str,
        api_key: str,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> AccountCredentials:
        """
        Adds new credentials for a specific provider and account ID.

        The credentials (api_key and additional_headers) are stored both in-memory
        and persisted to the encrypted storage file.

        Args:
            provider: The name of the LLM provider (e.g., "openai").
            account_id: A user-defined identifier for this specific account/credential set.
            api_key: The API key for the provider.
            additional_headers: Optional dictionary of additional HTTP headers
                                required by the provider.

        Returns:
            The created AccountCredentials object.
        """
        
        credentials = AccountCredentials(
            provider=provider,
            account_id=account_id,
            api_key=api_key,
            additional_headers=additional_headers or {},
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        if provider not in self.credentials:
            self.credentials[provider] = []
        
        self.credentials[provider].append(credentials)
        
        # Initialize usage tracking
        if provider not in self.usage_tracking:
            self.usage_tracking[provider] = {}
        self.usage_tracking[provider][account_id] = 0
        
        await self.save_credentials()
        logger.info("Added credentials", provider=provider, account_id=account_id)
        
        return credentials
    
    async def get_credentials(self, provider: str) -> Optional[AccountCredentials]:
        """Get active credentials for a provider using round-robin selection."""
        
        if provider not in self.credentials:
            return None
        
        active_creds = [
            cred for cred in self.credentials[provider] 
            if cred.is_active and not self._is_rate_limited(cred)
        ]
        
        if not active_creds:
            return None
        
        # Simple round-robin selection based on usage count
        selected_cred = min(active_creds, key=lambda c: self.usage_tracking[provider].get(c.account_id, 0))
        
        return selected_cred
    
    async def update_usage(self, credentials: AccountCredentials) -> None:
        """
        Updates usage statistics for the given credentials.

        This typically involves incrementing a usage counter and updating the
        last used timestamp. The changes are persisted.

        Args:
            credentials: The AccountCredentials object that was used.
        """
        
        provider: str = credentials.provider
        account_id: str = credentials.account_id
        
        if provider in self.usage_tracking and account_id in self.usage_tracking[provider]:
            self.usage_tracking[provider][account_id] += 1
        
        # Update last used timestamp
        credentials.last_used = datetime.utcnow()
        credentials.usage_count += 1
        
        await self.save_credentials()
    
    async def mark_credentials_invalid(self, provider: str, account_id: Optional[str] = None) -> None:
        """
        Marks specific credentials (or all for a provider) as inactive.

        This is typically called after an authentication failure. The changes are persisted.

        Args:
            provider: The name of the provider.
            account_id: Optional. If provided, only marks credentials for this
                        specific account ID as invalid. If None, all credentials
                        for the provider are marked invalid.
        """
        
        if provider not in self.credentials:
            logger.debug("Provider not found, cannot mark credentials invalid", provider_name=provider)
            return
        
        found_to_mark = False
        for cred in self.credentials[provider]:
            if account_id is None or cred.account_id == account_id:
                if cred.is_active: # Only log if state actually changes
                    cred.is_active = False
                    logger.warning("Marked credentials as invalid", provider=provider, account_id=cred.account_id)
                    found_to_mark = True

        if found_to_mark:
        
        await self.save_credentials()
    
    async def set_rate_limit_reset(self, provider: str, account_id: str, reset_time: datetime) -> None:
        """
        Sets the rate limit reset time for a specific credential.

        This information is used by `get_credentials` to avoid selecting
        credentials that are currently rate-limited. Changes are persisted.

        Args:
            provider: The name of the provider.
            account_id: The account ID for the specific credential.
            reset_time: The datetime object indicating when the rate limit will reset.
        """
        
        if provider not in self.credentials:
            logger.debug("Provider not found, cannot set rate limit reset", provider_name=provider, account_id=account_id)
            return
        
        found_credential = False
        for cred in self.credentials[provider]:
            if cred.account_id == account_id:
                cred.rate_limit_reset = reset_time
                logger.info("Set rate limit reset time for credential", provider=provider, account_id=account_id, reset_time=reset_time.isoformat())
                found_credential = True
                break
        
        if found_credential:
            await self.save_credentials()
        else:
            logger.warning("Credential not found to set rate limit reset", provider=provider, account_id=account_id)
    
    def _is_rate_limited(self, credentials: AccountCredentials) -> bool:
        """Check if credentials are currently rate limited."""
        
        if not credentials.rate_limit_reset:
            return False
        
        return datetime.utcnow() < credentials.rate_limit_reset
    
    async def list_credentials(self) -> Dict[str, List[Dict[str, Any]]]: # Changed 'any' to 'Any'
        """
        Lists all configured credentials, excluding sensitive data like API keys.

        Returns:
            A dictionary where keys are provider names and values are lists of
            dictionaries, each representing a credential's non-sensitive details.
        """
        
        result: Dict[str, List[Dict[str, Any]]] = {}
        for provider, creds_list in self.credentials.items():
            result[provider] = []
            for cred in creds_list: # Changed variable name from creds to cred_list then cred
                result[provider].append({
                    "account_id": cred.account_id,
                    "is_active": cred.is_active,
                    "created_at": cred.created_at.isoformat(),
                    "last_used": cred.last_used.isoformat() if cred.last_used else None,
                    "usage_count": cred.usage_count,
                    "rate_limit_reset": cred.rate_limit_reset.isoformat() if cred.rate_limit_reset else None
                })
        
        return result
    
    async def remove_credentials(self, provider: str, account_id: str) -> bool:
        """
        Removes a specific credential set for a provider and account ID.

        Changes are persisted.

        Args:
            provider: The name of the provider.
            account_id: The account ID of the credentials to remove.

        Returns:
            True if credentials were found and removed, False otherwise.
        """
        
        if provider not in self.credentials:
            logger.warning("Provider not found, cannot remove credentials", provider_name=provider, account_id=account_id)
            return False
        
        original_count = len(self.credentials[provider])
        self.credentials[provider] = [
            cred for cred in self.credentials[provider] 
            if cred.account_id != account_id
        ]
        
        removed = len(self.credentials[provider]) < original_count
        
        if removed:
            # Clean up usage tracking
            if provider in self.usage_tracking and account_id in self.usage_tracking[provider]:
                del self.usage_tracking[provider][account_id]
            
            await self.save_credentials() # save_credentials logs its own success/failure
            logger.info("Removed credentials", provider=provider, account_id=account_id)
        else:
            logger.warning("Credentials not found for removal", provider=provider, account_id=account_id)
        
        return removed
    
    async def rotate_credentials(self, provider: str) -> None:
        """
        Rotates credentials for a provider by resetting usage counts.

        This encourages the round-robin selection in `get_credentials` to pick
        a different credential if multiple are available and active.
        This method does not change the order or deactivate credentials.

        Args:
            provider: The name of the provider for which to rotate credentials.
        """
        
        if provider not in self.credentials or not self.credentials[provider]:
            logger.debug("No credentials for provider or provider not found, cannot rotate", provider_name=provider)
            return
        
        active_creds = [cred for cred in self.credentials[provider] if cred.is_active]
        if len(active_creds) <= 1:
            logger.debug("Not enough active credentials to rotate", provider_name=provider, active_creds_count=len(active_creds))
            return
        
        # Reset usage counts to force rotation
        for cred in active_creds:
            if provider in self.usage_tracking and cred.account_id in self.usage_tracking[provider]:
                 self.usage_tracking[provider][cred.account_id] = 0
            # If not in usage_tracking, it's implicitly 0, so no action needed.
        
        logger.info("Rotated credentials for provider by resetting usage counts", provider=provider)
        # Persisting changes as usage_tracking is part of saved data.
        await self.save_credentials()
    
    async def save_credentials(self) -> None:
        """
        Saves the current state of all credentials and usage tracking to the
        encrypted storage file.

        This method is called internally after modifications like adding, removing,
        or updating credentials.
        """

        if not settings.OPENHANDS_ENCRYPTION_KEY:
            logger.warning("Saving credentials without a stable OPENHANDS_ENCRYPTION_KEY.",
                           storage_path=self.storage_path,
                           risk="Credentials may not be recoverable if application restarts with a new auto-generated key.")

        try:
            # Prepare data for serialization
            data_to_save: Dict[str, Any] = { # More specific type for data_to_save
                "credentials": {},
                "usage_tracking": self.usage_tracking
            }
            
            for provider_key, creds_list in self.credentials.items():
                data_to_save["credentials"][provider_key] = []
                for cred in creds_list:
                    cred_data = {
                        "account_id": cred.account_id,
                        "api_key": cred.api_key, # This will be encrypted
                        "additional_headers": cred.additional_headers,
                        "is_active": cred.is_active,
                        "created_at": cred.created_at.isoformat(),
                        "last_used": cred.last_used.isoformat() if cred.last_used else None,
                        "usage_count": cred.usage_count,
                        "rate_limit_reset": cred.rate_limit_reset.isoformat() if cred.rate_limit_reset else None
                    }
                    data_to_save["credentials"][provider_key].append(cred_data)
            
            # Encrypt and save
            json_data = json.dumps(data_to_save)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            
            with open(self.storage_path, "wb") as f:
                f.write(encrypted_data)
            
            logger.debug("Credentials saved successfully", storage_path=self.storage_path)
            
        except Exception as e:
            logger.error("Failed to save credentials", storage_path=self.storage_path, error=str(e), exc_info=True)
    
    async def load_credentials(self) -> None:
        """
        Loads credentials from the encrypted storage file and environment variables.

        Credentials from environment variables take precedence over those in the file
        if there are conflicts for the same provider and account ID.
        This method is typically called once during initialization.
        """

        # Attempt to load from storage_path (credentials.json)
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as f:
                    encrypted_data = f.read()

                json_data = self.cipher.decrypt(encrypted_data).decode()
                data: Dict[str, Any] = json.loads(json_data) # Type hint for data

                # Load credentials from file
                loaded_from_file_count = 0
                for provider_key, creds_data_list in data.get("credentials", {}).items():
                    if provider_key not in self.credentials:
                        self.credentials[provider_key] = []
                    for cred_data in creds_data_list:
                        # Check if already loaded from env var, skip if so
                        if any(c.provider == provider_key and c.account_id == cred_data["account_id"] for c in self.credentials.get(provider_key, [])):
                            logger.info("Credential already loaded from environment, skipping from file.",
                                        provider=provider_key, account_id=cred_data["account_id"])
                            continue

                        cred = AccountCredentials(
                            provider=provider_key,
                            account_id=cred_data["account_id"],
                            api_key=cred_data["api_key"],
                            additional_headers=cred_data.get("additional_headers", {}),
                            is_active=cred_data.get("is_active", True),
                            created_at=datetime.fromisoformat(cred_data["created_at"]),
                            last_used=datetime.fromisoformat(cred_data["last_used"]) if cred_data.get("last_used") else None,
                            usage_count=cred_data.get("usage_count", 0),
                            rate_limit_reset=datetime.fromisoformat(cred_data["rate_limit_reset"]) if cred_data.get("rate_limit_reset") else None
                        )
                        self.credentials[provider_key].append(cred)
                        loaded_from_file_count +=1

                # Load usage tracking from file
                self.usage_tracking = data.get("usage_tracking", {})

                if not settings.OPENHANDS_ENCRYPTION_KEY:
                    logger.warning(
                        "Credentials loaded from file, but OPENHANDS_ENCRYPTION_KEY was not set.",
                        warning="These credentials might not be recoverable if the application restarts with a new auto-generated key.",
                        storage_path=self.storage_path)

                if loaded_from_file_count > 0: # Log only if actual credentials were processed from file
                    logger.info("Loaded credentials from file", storage_path=self.storage_path, count=loaded_from_file_count)

            except Exception as e:
                logger.error("Failed to load credentials from file", storage_path=self.storage_path, error=str(e), exc_info=True)
                # Continue to load from environment variables even if file loading fails
        else:
            logger.info("No credentials file found, attempting to load from environment variables.", storage_path=self.storage_path)

        # Load credentials from environment variables
        # Environment variable naming convention:
        # OPENHANDS_PROVIDER_{PROVIDER_NAME}_ACCOUNT_{ACCOUNT_ID}_API_KEY
        # OPENHANDS_PROVIDER_{PROVIDER_NAME}_ACCOUNT_{ACCOUNT_ID}_ADDITIONAL_HEADERS (JSON string)
        loaded_from_env = 0
        for env_var, value in os.environ.items():
            if env_var.startswith("OPENHANDS_PROVIDER_") and env_var.endswith("_API_KEY"):
                parts = env_var.split("_")
                if len(parts) < 6: # OPENHANDS, PROVIDER, provider_name, ACCOUNT, account_id, API, KEY
                    logger.warning("Could not parse environment variable for API key", env_var_name=env_var)
                    continue

                provider_name = parts[2] # Example: OPENHANDS_PROVIDER_OPENAI_ACCOUNT_USER1_API_KEY -> OPENAI
                # account_id can contain underscores, so we join the relevant parts
                account_id_parts = []
                for i in range(4, len(parts) - 2): # Iterate between ACCOUNT and API_KEY parts
                    account_id_parts.append(parts[i])
                account_id = "_".join(account_id_parts)

                api_key = value
                additional_headers: Dict[str, str] = {} # Ensure type
                headers_env_var = f"OPENHANDS_PROVIDER_{provider_name}_ACCOUNT_{account_id}_ADDITIONAL_HEADERS"

                if headers_env_var in os.environ:
                    try:
                        parsed_headers = json.loads(os.environ[headers_env_var])
                        if isinstance(parsed_headers, dict):
                            additional_headers = parsed_headers
                        else:
                            logger.error("Parsed additional headers from environment is not a dictionary",
                                         env_var_name=headers_env_var, parsed_type=type(parsed_headers).__name__)
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse additional headers JSON from environment variable",
                                     env_var_name=headers_env_var, error=str(e), exc_info=True)

                # Check if this credential (provider + account_id) already exists from file
                # If it exists, the environment variable takes precedence and overwrites it.
                existing_cred_index = -1
                if provider_name in self.credentials: # self.credentials might be populated from file
                    for idx, cred_from_file in enumerate(self.credentials[provider_name]):
                        if cred_from_file.account_id == account_id:
                            existing_cred_index = idx
                            logger.info("Overwriting credential from file with environment variable.",
                                        provider=provider_name, account_id=account_id)
                            break

                new_cred = AccountCredentials(
                    provider=provider_name,
                    account_id=account_id,
                    api_key=api_key,
                    additional_headers=additional_headers,
                    is_active=True, # Assume active if loaded from env
                    created_at=datetime.utcnow(), # Set new creation time
                    # last_used, usage_count, rate_limit_reset will be default/re-initialized
                )

                if provider_name not in self.credentials:
                    self.credentials[provider_name] = []

                if existing_cred_index != -1:
                    self.credentials[provider_name][existing_cred_index] = new_cred
                else:
                    self.credentials[provider_name].append(new_cred)

                # Initialize usage tracking if not present from file
                if provider_name not in self.usage_tracking:
                    self.usage_tracking[provider_name] = {}
                if account_id not in self.usage_tracking[provider_name]:
                     self.usage_tracking[provider_name][account_id] = 0

                loaded_from_env +=1
                logger.debug("Loaded credential from environment variable", provider=provider_name, account_id=account_id)


        if loaded_from_env > 0:
            logger.info("Finished loading credentials from environment variables.", count=loaded_from_env)

        if not self.credentials: # Check after both file and env var attempts
             if not os.path.exists(self.storage_path): # File never existed
                 logger.info("No credentials loaded: File not found and no relevant environment variables set.")
             else: # File existed but might have been empty, unreadable, or all creds overwritten by env
                 logger.info("No credentials currently loaded (file might have been empty/unreadable or all creds came from env).")


    async def get_usage_stats(self) -> Dict[str, Dict[str, Any]]: # Changed 'any' to 'Any'
        """
        Retrieves usage statistics for all providers and their accounts.

        Returns:
            A dictionary where keys are provider names. Each provider's value
            is a dictionary containing 'total_usage', 'active_accounts', and
            'account_usage' (a sub-dictionary of account_id to usage count).
        """

        stats: Dict[str, Dict[str, Any]] = {}
        for provider, accounts_usage in self.usage_tracking.items(): # Changed variable name
            total_usage = sum(accounts_usage.values())
            active_accounts_count = 0
            if provider in self.credentials: # Check if provider key exists in credentials
                active_accounts_count = len([
                    cred for cred in self.credentials[provider]
                    if cred.is_active
                ])
            
            stats[provider] = {
                "total_usage": total_usage,
                "active_accounts": active_accounts_count,
                "account_usage": accounts_usage # Direct use of iterated item
            }
        
        return stats