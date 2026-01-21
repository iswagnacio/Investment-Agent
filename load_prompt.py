"""
Prompt Loader - Loads and formats prompts from the prompts directory.

Prompts are stored as plain text files with {placeholder} variables
that get filled in at runtime.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from config import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


class PromptLoader:
    """
    Loads prompts from text files.

    Usage:
        loader = PromptLoader()

        # Load and format a prompt
        system = loader.get("metric_selection_system")
        user = loader.get("metric_selection_user",
            company_name="Apple Inc.",
            sector="technology",
            ...
        )
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            prompts_dir: Path to prompts directory.
                         Defaults to config.PROMPTS_DIR or ./prompts
        """
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        elif HAS_CONFIG:
            self.prompts_dir = config.PROMPTS_DIR
        else:
            # Fallback: prompts/ directory relative to this file
            self.prompts_dir = Path(__file__).parent / "prompts"

        self._cache: dict[str, str] = {}
    
    def _load_file(self, name: str) -> str:
        """Load a prompt file by name (without extension)."""
        if name in self._cache:
            return self._cache[name]
        
        filepath = self.prompts_dir / f"{name}.txt"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        self._cache[name] = content
        return content
    
    def get(self, name: str, **kwargs) -> str:
        """
        Load a prompt and optionally format it with variables.
        
        Args:
            name: Prompt file name (without .txt extension)
            **kwargs: Variables to substitute in the prompt
            
        Returns:
            Formatted prompt string
        """
        template = self._load_file(name)
        
        if kwargs:
            return template.format(**kwargs)
        
        return template
    
    def list_prompts(self) -> list[str]:
        """List all available prompt names."""
        if not self.prompts_dir.exists():
            return []
        
        return [
            f.stem for f in self.prompts_dir.glob("*.txt")
        ]
    
    def reload(self, name: Optional[str] = None):
        """
        Clear cache to reload prompts from disk.
        
        Args:
            name: Specific prompt to reload, or None to clear all
        """
        if name:
            self._cache.pop(name, None)
        else:
            self._cache.clear()


# Singleton instance for convenience
_default_loader: Optional[PromptLoader] = None


def get_prompt(name: str, **kwargs) -> str:
    """
    Convenience function to load a prompt.
    
    Args:
        name: Prompt file name (without .txt extension)
        **kwargs: Variables to substitute
        
    Returns:
        Formatted prompt string
    """
    global _default_loader
    
    if _default_loader is None:
        _default_loader = PromptLoader()
    
    return _default_loader.get(name, **kwargs)


def list_prompts() -> list[str]:
    """List all available prompt names."""
    global _default_loader
    
    if _default_loader is None:
        _default_loader = PromptLoader()
    
    return _default_loader.list_prompts()


# --- Demo ---

if __name__ == "__main__":
    loader = PromptLoader()
    
    print("Available prompts:")
    for name in loader.list_prompts():
        print(f"  - {name}")
    
    print("\n--- metric_selection_system ---")
    print(loader.get("metric_selection_system"))
    
    print("\n--- metric_selection_user (formatted) ---")
    print(loader.get("metric_selection_user",
        company_name="Apple Inc.",
        sector="technology",
        industry_desc="Electronic Computers",
        available_data='["revenue", "net_income", "eps"]',
        focus_areas="None specified",
        metrics_catalog='{"gross_margin": {...}, ...}'
    ))