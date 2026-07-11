"""
Mistral Client Module

Handles communication with Mistral AI API.
"""

import os
import json
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import requests
import httpx
from pydantic import BaseModel, Field


@dataclass
class MistralConfig:
    """Configuration for Mistral API."""
    api_key: str = Field(default="", description="Mistral API key")
    base_url: str = Field(default="https://api.mistral.ai/v1", description="Mistral API base URL")
    model: str = Field(default="mistral-small", description="Model to use")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens to generate")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    retries: int = Field(default=3, description="Number of retries for failed requests")


class MistralClient:
    """Client for interacting with Mistral AI API."""
    
    def __init__(self, config: Optional[MistralConfig] = None):
        """
        Initialize the Mistral client.
        
        Args:
            config: Mistral configuration
        """
        if config is None:
            # Try to get API key from environment
            api_key = os.getenv('MISTRAL_API_KEY', '')
            config = MistralConfig(api_key=api_key)
        
        self.config = config
        self.client = httpx.Client(timeout=self.config.timeout)
        
        # Validate API key
        if not self.config.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable.")
            
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict] = None,
        retries: Optional[int] = None
    ) -> Dict:
        """
        Make a request to the Mistral API with retry logic.
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint
            payload: Request payload
            retries: Number of retries
            
        Returns:
            Response as dictionary
        """
        retries = retries or self.config.retries
        url = f"{self.config.base_url}{endpoint}"
        headers = self._get_headers()
        
        for attempt in range(retries):
            try:
                if method.upper() == 'GET':
                    response = self.client.get(url, headers=headers, params=payload)
                elif method.upper() == 'POST':
                    response = self.client.post(url, headers=headers, json=payload)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif e.response.status_code == 401:
                    raise ValueError("Invalid API key. Please check your MISTRAL_API_KEY.")
                else:
                    raise Exception(f"API request failed: {e}")
                    
            except Exception as e:
                if attempt == retries - 1:
                    raise Exception(f"API request failed after {retries} attempts: {e}")
                wait_time = 2 ** attempt
                print(f"Request failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
        return {}
    
    def chat_completion(
        self,
        messages: List[Dict[str, Union[str, List[Dict]]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Completion response
        """
        endpoint = "/chat/completions"
        
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": stream
        }
        
        return self._make_request('POST', endpoint, payload)
    
    def completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """
        Generate a text completion (legacy endpoint).
        
        Args:
            prompt: Text prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Completion response
        """
        endpoint = "/completions"
        
        payload = {
            "model": model or self.config.model,
            "prompt": prompt,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens
        }
        
        return self._make_request('POST', endpoint, payload)
    
    def get_models(self) -> List[Dict]:
        """Get list of available models."""
        endpoint = "/models"
        response = self._make_request('GET', endpoint)
        return response.get('data', [])
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a specific model."""
        endpoint = f"/models/{model_id}"
        return self._make_request('GET', endpoint)
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MistralResponse(BaseModel):
    """Pydantic model for Mistral API response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Dict


class Message(BaseModel):
    """Pydantic model for chat message."""
    role: str  # 'system', 'user', 'assistant'
    content: str


class FunctionCall(BaseModel):
    """Pydantic model for function call."""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Pydantic model for tool call."""
    id: str
    type: str  # 'function'
    function: FunctionCall


class AssistantMessage(BaseModel):
    """Pydantic model for assistant message."""
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


# Module-level functions
def get_mistral_client(api_key: Optional[str] = None, **kwargs) -> MistralClient:
    """
    Create a Mistral client with optional configuration.
    
    Args:
        api_key: Mistral API key (overrides environment variable)
        **kwargs: Additional configuration parameters
        
    Returns:
        MistralClient instance
    """
    if api_key:
        os.environ['MISTRAL_API_KEY'] = api_key
        
    config = MistralConfig(**kwargs)
    return MistralClient(config)


def chat_completion(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = "mistral-small",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Dict:
    """
    Generate a chat completion with default client.
    
    Args:
        messages: List of message dictionaries
        api_key: Mistral API key
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Completion response
    """
    client = get_mistral_client(api_key)
    try:
        return client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    finally:
        client.close()
