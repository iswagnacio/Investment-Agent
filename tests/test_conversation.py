"""Bounded history and real ReAct prompt integration, without paid model calls."""

import json
from copy import deepcopy

import pytest
from langchain_community.llms.fake import FakeListLLM
from langchain_core.callbacks import BaseCallbackHandler

from investment import conversation
from investment.universe import PortfolioManager


def test_history_retains_newest_complete_exchanges_without_mutating_input():
    history = [{"user": f"question {i}", "assistant": f"answer {i}"} for i in range(9)]
    original = deepcopy(history)
    bounded = conversation.bounded_history(history)
    assert len(bounded) == 6
    assert bounded[0]["user"] == "question 3"
    assert bounded[-1]["assistant"] == "answer 8"
    bounded[0]["assistant"] = "changed copy"
    assert history == original


def test_memory_bounds_message_size_and_serialized_prompt_size():
    history = [{"user": '"\\' * 3000, "assistant": "答" * 6000} for _ in range(8)]
    bounded = conversation.bounded_history(history)
    assert 1 <= len(bounded) <= 6
    assert all(len(turn[field]) <= 2000 for turn in bounded for field in ("user", "assistant"))
    serialized = conversation.format_history(history)
    assert len(serialized) <= 12000
    assert json.loads(serialized) == bounded
    assert "[truncated]" in serialized


def test_add_exchange_is_bounded_and_independent():
    previous = [{"user": "Which stock?", "assistant": "AAPL"}]
    result = conversation.add_exchange(previous, "How many shares?", "10 shares")
    assert result[-1] == {"user": "How many shares?", "assistant": "10 shares"}
    assert len(previous) == 1
    result[0]["user"] = "edited"
    assert previous[0]["user"] == "Which stock?"
    assert conversation.format_history([]) == ""
    assert conversation.bounded_history(None) == []


def test_json_escaping_cannot_evict_the_newest_exchange():
    history = conversation.add_exchange(
        [{"user": "Earlier question", "assistant": "Earlier answer"}],
        "Newest question", "\x01" * 2000,
    )
    assert history and history[-1]["user"] == "Newest question"
    assert "[truncated]" in history[-1]["assistant"]
    assert len(conversation.format_history(history)) <= 12000


@pytest.fixture
def synthetic_csv(tmp_path):
    csv = tmp_path / "synthetic.csv"
    csv.write_text(
        "status,ticker,shares,cost_basis,date_acquired,notes\n"
        "hold,AAPL,10,100,2026-01-15,Synthetic first holding\n"
        "hold,MSFT,5,200,2026-02-01,Synthetic second holding\n"
        "watch,NVDA,0,0,,Synthetic watchlist\n",
        encoding="utf-8",
    )
    return csv


def test_existing_read_tools_expose_stored_shares_without_value_calculations(synthetic_csv):
    manager = PortfolioManager(csv_path=str(synthetic_csv))
    before = synthetic_csv.read_bytes()
    holdings = manager.get_holdings()
    assert "AAPL" in holdings and "Shares: 10" in holdings
    assert "MSFT" in holdings and "Shares: 5" in holdings
    assert "Shares: 10" in manager.get_position_details("AAPL")
    assert "Shares:" not in manager.get_position_details("NVDA")
    assert "cost_basis" not in holdings and "$" not in holdings
    assert synthetic_csv.read_bytes() == before


def test_real_react_prompt_receives_followups_and_supports_history_free_queries(
    monkeypatch, synthetic_csv,
):
    import investment.agents.portfolio as portfolio

    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-placeholder-not-a-credential")
    prompts = []

    class CapturePrompts(BaseCallbackHandler):
        def on_llm_start(self, serialized, batch, **kwargs):
            prompts.extend(batch)

    llm = FakeListLLM(responses=[
        "Thought: Read holdings.\nAction: get_holdings\nAction Input: unused",
        "Thought: Ready.\nFinal Answer: AAPL has 10 shares and MSFT has 5 shares.",
        "Thought: Check the position.\nAction: get_position_details\nAction Input: AAPL",
        "Thought: Ready.\nFinal Answer: AAPL is largest by share count with 10 shares.",
        "Thought: Ready.\nFinal Answer: You have 10 shares of AAPL.",
        "Thought: Ready.\nFinal Answer: This is a new conversation.",
    ], callbacks=[CapturePrompts()])
    monkeypatch.setattr(portfolio, "ChatAnthropic", lambda **kwargs: llm)
    agent = portfolio.PortfolioTrackerAgent(csv_path=str(synthetic_csv))
    first = agent.query("What's in my portfolio?")
    history = conversation.add_exchange([], "What's in my portfolio?", first)
    second = agent.query("Tell me more about my largest holding.", conversation.format_history(history))
    history = conversation.add_exchange(history, "Tell me more about my largest holding.", second)
    third = agent.query("How many shares do I have of that stock?", conversation.format_history(history))
    assert third == "You have 10 shares of AAPL."
    assert "AAPL has 10 shares" in prompts[2]
    assert "Shares: 10" in prompts[3]
    assert "Tell me more about my largest holding." in prompts[4]
    assert "AAPL is largest by share count" in prompts[4]
    assert "by share count" in prompts[0]
    # No persistent memory inside the agent: omitted context starts clean.
    result = agent.agent_executor.invoke({"input": "Start again"})
    assert result["output"] == "This is a new conversation."
    assert "AAPL has 10 shares" not in prompts[5]
