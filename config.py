"""
Centralized Configuration Management
Loads environment variables and provides configuration access
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Application configuration"""

    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # User Settings
    USER_EMAIL: str = os.getenv("USER_EMAIL", "user@example.com")

    # Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-5-20250929")
    FAST_MODEL: str = os.getenv("FAST_MODEL", "claude-haiku-4-5")

    # Analysis Parameters
    MARKET_LOOKBACK_DAYS: int = int(os.getenv("MARKET_LOOKBACK_DAYS", "90"))

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
    DATA_DIR: Path = PROJECT_ROOT
    HOLDINGS_FILE: Path = DATA_DIR / "holdings.csv"

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.ANTHROPIC_API_KEY:
            print("Warning: ANTHROPIC_API_KEY not set in environment")
            return False
        return True

    @classmethod
    def get_anthropic_api_key(cls) -> Optional[str]:
        """Get Anthropic API key, return None if not set"""
        return cls.ANTHROPIC_API_KEY if cls.ANTHROPIC_API_KEY else None


# Create a singleton instance
config = Config()


# Validate configuration on import
if __name__ != "__main__":
    config.validate()
