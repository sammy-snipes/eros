import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Union

import anthropic
from anthropic import APIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, Field, field_validator


class ClaudeModel(str, Enum):
    """Claude model configurations with pricing (per 1M tokens)."""

    # Current models
    OPUS_4_1 = "claude-opus-4-1-20250805"
    OPUS_4 = "claude-opus-4-20250514"
    SONNET_4 = "claude-sonnet-4-20250514"
    SONNET_3_7 = "claude-3-7-sonnet-20250219"
    SONNET_3_7_LATEST = "claude-3-7-sonnet-latest"
    HAIKU_3_5 = "claude-3-5-haiku-20241022"
    HAIKU_3_5_LATEST = "claude-3-5-haiku-latest"

    # Deprecated but available
    HAIKU_3 = "claude-3-haiku-20240307"

    @property
    def input_cost(self) -> float:
        """Cost per 1M input tokens in USD."""
        costs = {
            self.OPUS_4_1: 15.0,
            self.OPUS_4: 15.0,
            self.SONNET_4: 3.0,
            self.SONNET_3_7: 3.0,
            self.SONNET_3_7_LATEST: 3.0,
            self.HAIKU_3_5: 0.8,
            self.HAIKU_3_5_LATEST: 0.8,
            self.HAIKU_3: 0.25,
        }
        return costs[self]

    @property
    def output_cost(self) -> float:
        """Cost per 1M output tokens in USD."""
        costs = {
            self.OPUS_4_1: 75.0,
            self.OPUS_4: 75.0,
            self.SONNET_4: 15.0,
            self.SONNET_3_7: 15.0,
            self.SONNET_3_7_LATEST: 15.0,
            self.HAIKU_3_5: 4.0,
            self.HAIKU_3_5_LATEST: 4.0,
            self.HAIKU_3: 1.25,
        }
        return costs[self]

    @property
    def display_name(self) -> str:
        """Human-readable model name."""
        names = {
            self.OPUS_4_1: "Claude Opus 4.1",
            self.OPUS_4: "Claude Opus 4",
            self.SONNET_4: "Claude Sonnet 4",
            self.SONNET_3_7: "Claude Sonnet 3.7",
            self.SONNET_3_7_LATEST: "Claude Sonnet 3.7 (Latest)",
            self.HAIKU_3_5: "Claude Haiku 3.5",
            self.HAIKU_3_5_LATEST: "Claude Haiku 3.5 (Latest)",
            self.HAIKU_3: "Claude Haiku 3 (Deprecated)",
        }
        return names[self]


class ClaudeConfig(BaseModel):
    model: ClaudeModel = Field(default=ClaudeModel.SONNET_4)
    max_tokens: int = Field(default=8192, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is worth retrying."""
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIError) and error.status_code >= 500:
        return True
    return False


def exponential_backoff_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
):
    """Exponential backoff retry decorator for API calls."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries or not is_retryable_error(e):
                        raise e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay *= 0.5 + 0.5 * (time.time() % 1)

                    logging.warning(
                        f"Attempt {attempt + 1} failed: {str(e)[:100]}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

            if last_exception is not None:
                raise last_exception
            else:
                raise RuntimeError("All retry attempts failed")

        return wrapper

    return decorator


class ChatMessage(BaseModel):
    """Represents a single message in the conversation."""

    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()

    def __str__(self) -> str:
        content_preview = (
            self.content[:80] + "..." if len(self.content) > 80 else self.content
        )
        return (
            f"[{self.role.upper()}] {self.timestamp.strftime('%H:%M:%S')}\n{content_preview}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "role": self.role,
            "content": self.content
        }


class ClaudeClient:
    """Anthropic API client with proper error handling and retry logic."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 120):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=timeout
        )

        self.logger = logging.getLogger(__name__)

    @exponential_backoff_retry(max_retries=3, base_delay=1.0)
    def send_message(
        self,
        messages: list[ChatMessage],
        system_prompt: Optional[str] = None,
        config: Optional[ClaudeConfig] = None
    ) -> ChatMessage:
        """Send message to Claude API with retry logic."""

        if not messages:
            raise ValueError("Messages list cannot be empty")

        config = config or ClaudeConfig()

        # Convert messages to API format
        api_messages = [msg.to_dict() for msg in messages]

        try:
            # Build request parameters
            request_params = {
                "model": config.model.value,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "messages": api_messages
            }

            if system_prompt:
                request_params["system"] = system_prompt

            self.logger.debug(f"Sending request to {config.model.display_name}")

            response = self.client.messages.create(**request_params)

            # Extract response content
            content = response.content[0].text if response.content else ""

            return ChatMessage(
                role="assistant",
                content=content,
                timestamp=datetime.now()
            )

        except APIError as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Claude API error: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise RuntimeError(f"Unexpected error calling Claude: {e}")


class ChatSession:
    """Manages a conversation session with Claude."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        config: Optional[ClaudeConfig] = None
    ):
        self.system_prompt = system_prompt
        self.config = config or ClaudeConfig()
        self.messages: list[ChatMessage] = []
        self.client = ClaudeClient()
        self.logger = logging.getLogger(__name__)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.logger.debug(f"Added {role} message: {content[:50]}...")

    def send_user_message(self, content: str) -> ChatMessage:
        """Send user message and get Claude's response."""
        # Add user message
        self.add_message("user", content)

        # Get response from Claude
        response = self.client.send_message(
            messages=self.messages,
            system_prompt=self.system_prompt,
            config=self.config
        )

        # Add response to history
        self.messages.append(response)

        return response

    def get_last_response(self) -> Optional[ChatMessage]:
        """Get the last assistant message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.logger.info("Conversation history cleared")

    def save_to_file(self, filepath: str) -> None:
        """Save conversation to JSON file."""
        data = {
            "system_prompt": self.system_prompt,
            "config": self.config.model_dump(),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in self.messages
            ],
            "created_at": datetime.now().isoformat()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Conversation saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> "ChatSession":
        """Load conversation from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        config_data = data.get("config")
        config = None
        if config_data:
            # Handle both old string format and new enum format
            if isinstance(config_data.get("model"), str):
                # Convert old model_name to new model enum
                model_name = config_data.pop("model", config_data.pop("model_name", None))
                if model_name:
                    # Find matching enum value
                    for model in ClaudeModel:
                        if model.value == model_name:
                            config_data["model"] = model
                            break
            config = ClaudeConfig(**config_data)

        session = cls(
            system_prompt=data.get("system_prompt"),
            config=config
        )

        # Restore messages
        for msg_data in data.get("messages", []):
            message = ChatMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"])
            )
            session.messages.append(message)

        return session

    def __len__(self) -> int:
        """Return number of messages in conversation."""
        return len(self.messages)

    def __str__(self) -> str:
        """String representation of the conversation."""
        if not self.messages:
            return "Empty conversation"

        return f"Conversation with {len(self.messages)} messages"