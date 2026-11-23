"""Offline startup checks using temporary data and no paid API requests."""

import socket
from pathlib import Path
from tempfile import gettempdir

import pytest
from langchain_community.llms.fake import FakeListLLM
from langchain_core.callbacks import BaseCallbackHandler

from investment.agents.fundamental import FundamentalAnalystAgent
from investment.agents.lead import GeneralAnalystAgent
from investment.agents.perspective import PerspectiveAnalystAgent
from investment.agents.portfolio import PortfolioTrackerAgent
from investment.agents.sentiment import SentimentAnalystAgent
from investment.agents.technical import TechnicalAnalyzerAgent
from investment.calculators import fundamental, sentiment
from yfinance_cache import yfc_cache_manager


AGENT_TYPES = (
    GeneralAnalystAgent,
    PortfolioTrackerAgent,
    TechnicalAnalyzerAgent,
    FundamentalAnalystAgent,
    SentimentAnalystAgent,
    PerspectiveAnalystAgent,
)


def test_yahoo_cache_is_temporary_before_agent_imports():
    # YFC chooses its cache (and may migrate it) during module import/collection.
    cache = Path(yfc_cache_manager.GetCacheDirpath()).resolve()
    assert cache.is_relative_to(Path(gettempdir()).resolve())


@pytest.fixture(autouse=True)
def isolated_startup(monkeypatch, tmp_path):
    """Use real constructors/clients, but prevent network and real data writes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-placeholder-not-a-credential")
    for key in ("FINNHUB_API_KEY", "NEWS_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.setattr(sentiment.Config, key, None)
    monkeypatch.chdir(tmp_path)

    def forbid_network(*args, **kwargs):
        raise AssertionError("Startup tests must not make network requests")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    fundamental_cache = fundamental.FundamentalCache
    news_cache = sentiment.CacheManager
    monkeypatch.setattr(
        fundamental, "FundamentalCache",
        lambda: fundamental_cache(cache_dir=str(tmp_path / "fundamental_cache")),
    )
    monkeypatch.setattr(
        sentiment, "CacheManager",
        lambda: news_cache(cache_dir=str(tmp_path / "news_cache")),
    )


def make_agent(agent_type, tmp_path):
    if agent_type is PortfolioTrackerAgent:
        return agent_type(csv_path=str(tmp_path / "synthetic.csv"))
    if agent_type is GeneralAnalystAgent:
        return agent_type(use_perspective=True)
    return agent_type()


@pytest.mark.parametrize("agent_type", AGENT_TYPES, ids=lambda cls: cls.__name__)
def test_agent_initializes_without_api_requests(agent_type, tmp_path):
    agent = make_agent(agent_type, tmp_path)
    assert agent.llm is not None
    if agent_type is GeneralAnalystAgent:
        # The lead catches specialist startup failures, so check for degradation.
        assert isinstance(agent.technical_agent, TechnicalAnalyzerAgent)
        assert isinstance(agent.fundamental_agent, FundamentalAnalystAgent)
        assert isinstance(agent.sentiment_agent, SentimentAnalystAgent)
        assert isinstance(agent.perspective_agent, PerspectiveAnalystAgent)
    elif agent_type is PortfolioTrackerAgent:
        assert agent.agent_executor is not None
        assert agent.portfolio_mgr.read_portfolio().empty


@pytest.mark.parametrize("agent_type", AGENT_TYPES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("credential", [None, ""], ids=["absent", "empty"])
def test_missing_key_has_clear_error_before_creating_data(
    agent_type, credential, monkeypatch, tmp_path,
):
    if credential is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", credential)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable not set"):
        make_agent(agent_type, tmp_path)
    assert not (tmp_path / "synthetic.csv").exists()
    assert not (tmp_path / "fundamental_cache").exists()
    assert not (tmp_path / "news_cache").exists()


def test_portfolio_query_executes_read_tool_without_modifying_csv(monkeypatch, tmp_path):
    import investment.agents.portfolio as portfolio

    csv = tmp_path / "synthetic.csv"
    contents = (
        "status,ticker,shares,cost_basis,date_acquired,notes\n"
        "hold,DEMO,10,1,2026-01-01,Synthetic fixture only\n"
    )
    csv.write_text(contents, encoding="utf-8")
    llm = FakeListLLM(responses=[
        "Thought: Inspect the holdings.\nAction: get_holdings\nAction Input: unused",
        "Thought: The tool returned the holdings.\nFinal Answer: The portfolio contains DEMO.",
    ])
    monkeypatch.setattr(portfolio, "ChatAnthropic", lambda **kwargs: llm)
    agent = PortfolioTrackerAgent(csv_path=str(csv))
    prompts = []
    # Capture actual prompts to verify that the real tool result reaches the model.
    class CapturePrompts(BaseCallbackHandler):
        def on_llm_start(self, serialized, batch, **kwargs):
            prompts.extend(batch)

    llm.callbacks = [CapturePrompts()]
    assert agent.query("What's in my portfolio?") == "The portfolio contains DEMO."
    assert len(prompts) == 2
    assert "Current Holdings:" in prompts[1]
    assert "Synthetic fixture only" in prompts[1]
    assert csv.read_text(encoding="utf-8") == contents
