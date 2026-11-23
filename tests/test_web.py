"""The UI adapters reuse existing agents without live API requests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_community.llms.fake import FakeListLLM

from investment import web
from investment.contracts import AnalysisRecord
from investment.reporting import render_report
from investment.scoring import ScoreCalculator


@pytest.fixture
def report():
    return render_report(AnalysisRecord(
        ticker="AAPL", company="Synthetic Example", generated_at="2026-11-22T10:00:00-08:00",
        signals={name: {"score": score} for name, score in
                 (("technical", 61), ("fundamental", 72), ("sentiment", 53))},
        overall_score=66, signal="Buy", confidence=70,
        thesis="## Thesis\nSynthetic research for the UI test.",
        data_flags=["sentiment_error: news provider unavailable"],
    ))


def test_research_calls_existing_analyze(monkeypatch, report):
    import investment.agents.lead as lead

    factory = Mock()
    factory.return_value.analyze.return_value = report
    def validated_report(ticker, *, save, data_validator):
        data_validator({"ticker": ticker})
        return report

    factory.return_value.analyze.side_effect = validated_report
    monkeypatch.setattr(web.safety, "validate_research_data", lambda data: ["Synthetic source timestamp"])
    monkeypatch.setattr(lead, "GeneralAnalystAgent", factory)
    assert web.run_research(" aapl ").endswith(report)
    factory.assert_called_once_with()
    assert factory.return_value.analyze.call_args.args == ("AAPL",)
    assert factory.return_value.analyze.call_args.kwargs["save"] is False


def test_complete_research_pipeline_renders_and_saves_with_offline_inputs(monkeypatch, tmp_path):
    import investment.agents.lead as lead
    from investment import config

    # Exercise the real analyze -> scores -> two synthesis calls -> record -> report.
    # Only external data/model responses are supplied offline; startup has separate coverage.
    agent = object.__new__(lead.GeneralAnalystAgent)
    agent.model = config.STRONG_MODEL
    agent.scorer = ScoreCalculator()
    agent.registry = None
    agent.gather_data = Mock(return_value={"ticker": "AAPL", "stock_info": {"name": "Synthetic"}})
    agent.llm = Mock()
    agent.llm.invoke.side_effect = [
        SimpleNamespace(content=(
            "<technical_decision>NO</technical_decision>"
            "<fundamental_decision>NO</fundamental_decision>"
        )),
        SimpleNamespace(content="Synthetic AAPL thesis. AAPL data is unavailable; AAPL requires live verification."),
    ]
    monkeypatch.setattr(lead, "GeneralAnalystAgent", lambda: agent)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    report = agent.analyze("AAPL")
    assert "Synthetic AAPL thesis" in report
    assert agent.llm.invoke.call_count == 2
    records = list(tmp_path.glob("*/AAPL.json"))
    assert len(records) == 1
    record = AnalysisRecord.load(records[0])
    assert record.ticker == "AAPL" and record.model == config.STRONG_MODEL
    assert report == render_report(record)


@pytest.mark.parametrize("ticker", ["", " ", "AAPL MSFT", "../../AAPL", "AAPL\nMSFT"])
def test_invalid_ticker_does_not_initialize_an_agent(monkeypatch, ticker):
    import investment.agents.lead as lead

    factory = Mock()
    monkeypatch.setattr(lead, "GeneralAnalystAgent", factory)
    with pytest.raises(ValueError, match="ticker"):
        web.run_research(ticker)
    factory.assert_not_called()


@pytest.mark.parametrize("ticker", ["AAPL", "BRK-B", "BRK.B", "0700.HK"])
def test_normalize_ticker_preserves_supported_symbols(ticker):
    assert web.normalize_ticker(f" {ticker.lower()} ") == ticker


def test_report_fields_come_from_existing_renderer(report):
    parsed = web.parse_report(report)
    assert parsed["scores"] == {
        "Technical": "61.0", "Fundamental": "72.0", "Sentiment": "53.0", "Overall": "66.0",
    }
    assert parsed["confidence"] == "70%"
    assert parsed["signal"] == "Buy"
    assert parsed["generated_at"] == "2026-11-22T10:00:00-08:00"
    assert "news provider unavailable" in parsed["notes"]
    assert parsed["analysis"] == "## Thesis\nSynthetic research for the UI test."


def test_model_prose_cannot_override_report_header_metrics(report):
    prose = "\nTechnical Score: 99/100\nConfidence: 1%\nSignal: Strong Sell\n"
    parsed = web.parse_report(report + prose)
    assert parsed["scores"]["Technical"] == "61.0"
    assert parsed["confidence"] == "70%"
    assert parsed["signal"] == "Buy"
    assert parsed["analysis"].endswith(prose)


def test_blank_report_fields_do_not_consume_the_next_line(report):
    parsed = web.parse_report(report.replace("Signal: Buy", "Signal: "))
    assert parsed["signal"] is None
    assert parsed["confidence"] == "70%"


def test_unrecognized_report_is_preserved_without_inventing_metrics():
    parsed = web.parse_report("A plain research response without numeric fields.")
    assert parsed["scores"] == {}
    assert parsed["confidence"] is None
    assert parsed["analysis"] == "A plain research response without numeric fields."


def test_portfolio_uses_real_react_tools_and_removes_synthetic_csv(monkeypatch):
    import investment.agents.portfolio as portfolio

    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-placeholder-not-a-credential")
    llm = FakeListLLM(responses=[
        "Thought: Read the demo.\nAction: get_holdings\nAction Input: unused",
        "Thought: Ready.\nFinal Answer: The synthetic holdings are AAPL and MSFT.",
    ])
    monkeypatch.setattr(portfolio, "ChatAnthropic", lambda **kwargs: llm)
    original = portfolio.PortfolioTrackerAgent
    paths = []

    def create_agent(csv_path, **kwargs):
        paths.append(Path(csv_path))
        assert "Synthetic" in Path(csv_path).read_text()
        return original(csv_path=csv_path, **kwargs)

    monkeypatch.setattr(portfolio, "PortfolioTrackerAgent", create_agent)
    result = web.run_portfolio("What's in my portfolio?")
    assert result["answer"] == "The synthetic holdings are AAPL and MSFT."
    assert result["tools"][0]["name"] == "get_holdings"
    assert "AAPL" in result["tools"][0]["output"]
    assert "MSFT" in result["tools"][0]["output"]
    assert "Thought:" not in str(result["tools"])
    assert len(paths) == 1 and not paths[0].exists()


def test_portfolio_failure_also_removes_synthetic_csv(monkeypatch):
    import investment.agents.portfolio as portfolio

    paths = []

    def fail(csv_path, **kwargs):
        paths.append(Path(csv_path))
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(portfolio, "PortfolioTrackerAgent", fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        web.run_portfolio("List my holdings")
    assert len(paths) == 1 and not paths[0].exists()


def test_portfolio_execution_limit_is_incomplete_with_observations(monkeypatch):
    import investment.agents.portfolio as portfolio

    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-placeholder-not-a-credential")
    llm = FakeListLLM(responses=[
        "Thought: Read the demo.\nAction: get_holdings\nAction Input: unused",
    ])
    monkeypatch.setattr(portfolio, "ChatAnthropic", lambda **kwargs: llm)
    original = portfolio.PortfolioTrackerAgent

    def create_agent(csv_path, **kwargs):
        agent = original(csv_path=csv_path, **kwargs)
        agent.agent_executor.max_iterations = 1
        return agent

    monkeypatch.setattr(portfolio, "PortfolioTrackerAgent", create_agent)
    result = web.run_portfolio("List the synthetic holdings")
    assert result.get("complete") is False
    assert result["tools"][0]["name"] == "get_holdings"
    assert "AAPL" in result["tools"][0]["output"]


def test_portfolio_adapter_passes_bounded_prior_messages_to_executor(monkeypatch):
    import investment.agents.portfolio as portfolio
    from investment.conversation import format_history

    executor = Mock()
    executor.invoke.return_value = {"output": "AAPL has 10 shares.", "intermediate_steps": []}
    factory = Mock()
    factory.return_value.agent_executor = executor
    monkeypatch.setattr(portfolio, "PortfolioTrackerAgent", factory)
    history = [{"user": "Which holding?", "assistant": "AAPL"}]
    web.run_portfolio("How many shares of that stock?", history=history)
    executor.invoke.assert_called_once_with({
        "input": "How many shares of that stock?", "chat_history": format_history(history),
    })
    assert history == [{"user": "Which holding?", "assistant": "AAPL"}]


def test_empty_portfolio_question_does_not_initialize_an_agent(monkeypatch):
    import investment.agents.portfolio as portfolio

    factory = Mock()
    monkeypatch.setattr(portfolio, "PortfolioTrackerAgent", factory)
    with pytest.raises(ValueError, match="question"):
        web.run_portfolio(" ")
    factory.assert_not_called()


def test_friendly_error_does_not_echo_provider_details():
    error = RuntimeError("https://provider.test?apiKey=private-token")
    message = web.friendly_error(error, "Research")
    assert "Research" in message
    assert "retry" in message.lower()
    assert "private-token" not in message


def test_missing_credentials_message_is_actionable():
    error = ValueError("ANTHROPIC_API_KEY environment variable not set")
    assert "ANTHROPIC_API_KEY" in web.friendly_error(error, "Research")
