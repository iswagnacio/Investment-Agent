"""Shared chat model configuration for the stock analysis pipeline."""

import json
import os
from pathlib import Path


def _pi_zai_key() -> str:
    agent_dir = Path(os.getenv("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent")
    auth_path = agent_dir.expanduser() / "auth.json"
    try:
        credentials = json.loads(auth_path.read_text(encoding="utf-8"))
        record = credentials["zai"]
        if record["type"] == "api_key":
            key = record["key"]
            if isinstance(key, str) and key.strip():
                return key
    except (OSError, ValueError, KeyError, TypeError):
        pass
    raise ValueError(f"No usable Z.ai API key found in Pi auth file {auth_path}.")


def create_chat_model(model: str, *, max_tokens: int = 4096):
    if model.startswith("glm-"):
        if os.getenv("ZAI_USE_PI_AUTH") == "1":
            api_key = _pi_zai_key()
            base_url = "https://api.z.ai/api/coding/paas/v4"
        else:
            api_key = os.getenv("ZAI_API_KEY")
            if not api_key:
                raise ValueError("Set ZAI_API_KEY or use ZAI_USE_PI_AUTH=1 to use GLM.")
            base_url = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=1,
            top_p=0.95,
            # Z.ai expects max_tokens, and Flash counts reasoning in that budget.
            extra_body={"max_tokens": max(max_tokens, 8192)},
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model_name=model, temperature=0, max_tokens=max_tokens)
