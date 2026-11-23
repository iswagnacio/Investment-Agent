"""Exercise both workflows through Streamlit's actual widget test runtime."""

from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("streamlit", reason="Install the web extra for UI tests")
from streamlit.testing.v1 import AppTest

from investment import web
from investment import conversation
from investment.contracts import AnalysisRecord
from investment.reporting import render_report


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def backends(monkeypatch):
    report = render_report(AnalysisRecord(
        ticker="AAPL", company="Synthetic Example",
        generated_at="2026-11-22T10:00:00-08:00",
        signals={name: {"score": score} for name, score in
                 (("technical", 61), ("fundamental", 72), ("sentiment", 53))},
        overall_score=66, confidence=70, signal="Buy",
        thesis="## Thesis\nSynthetic consolidated research.",
        data_flags=["sentiment_error: news provider unavailable"],
    ))
    research = Mock(return_value=report)
    portfolio = Mock(return_value={
        "answer": "Your synthetic holdings are AAPL and MSFT.",
        "tools": [{"name": "get_holdings", "output": "Current Holdings: AAPL, MSFT"}],
    })
    monkeypatch.setattr(web, "run_research", research)
    monkeypatch.setattr(web, "run_portfolio", portfolio)
    return research, portfolio


def app():
    return AppTest.from_file(str(APP), default_timeout=15).run()


def test_startup_and_workflow_switch_do_not_call_agents(backends):
    research, portfolio = backends
    at = app()
    assert not at.exception
    assert at.title[0].value == "Investment Agent"
    at.radio(key="workflow").set_value("Portfolio Assistant").run()
    assert not at.exception
    assert any("synthetic" in item.value.lower() for item in at.info)
    research.assert_not_called()
    portfolio.assert_not_called()


def test_research_displays_report_fields_and_preserves_result_on_rerun(backends):
    research, portfolio = backends
    at = app()
    at.text_input(key="ticker").set_value(" aapl ")
    at.button(key="research_submit").click().run()
    assert not at.exception
    research.assert_called_once_with("AAPL")
    portfolio.assert_not_called()
    metrics = {item.label: item.value for item in at.metric}
    assert metrics == {
        "Technical": "61.0 / 100", "Fundamental": "72.0 / 100",
        "Sentiment": "53.0 / 100", "Overall": "66.0 / 100",
        "Confidence": "70%", "Signal": "Buy",
    }
    assert any("Synthetic consolidated research" in item.value for item in at.text)
    assert any("news provider unavailable" in item.value for item in at.text)
    assert any("2026-11-22T10:00:00-08:00" in item.value for item in at.caption)
    assert any(item.value == "AAPL research report" for item in at.subheader)
    at.run()
    research.assert_called_once()
    assert len(at.metric) == 6


def test_failed_research_clears_old_result_and_can_recover(backends):
    research, _ = backends
    at = app()
    at.text_input(key="ticker").set_value("AAPL")
    at.button(key="research_submit").click().run()
    report = research.return_value
    research.side_effect = RuntimeError("private-provider-token")
    at.text_input(key="ticker").set_value("MSFT")
    at.button(key="research_submit").click().run()
    assert not at.exception
    assert not at.metric
    assert at.error and "retry" in at.error[0].value.lower()
    assert "private-provider-token" not in at.error[0].value
    research.side_effect = None
    research.return_value = report
    at.button(key="research_submit").click().run()
    assert not at.error and len(at.metric) == 6


def test_invalid_ticker_is_rejected_before_research(backends):
    research, _ = backends
    at = app()
    at.text_input(key="ticker").set_value("../../holdings")
    at.button(key="research_submit").click().run()
    assert not at.exception
    assert "ticker" in at.error[0].value
    research.assert_not_called()


def test_missing_credentials_has_actionable_ui_error(backends):
    research, _ = backends
    research.side_effect = ValueError("ANTHROPIC_API_KEY environment variable not set")
    at = app()
    at.text_input(key="ticker").set_value("AAPL")
    at.button(key="research_submit").click().run()
    assert not at.exception
    assert "ANTHROPIC_API_KEY" in at.error[0].value


def test_plain_report_is_still_visible_without_invented_scores(backends):
    research, _ = backends
    research.return_value = "Research response with unavailable score fields."
    at = app()
    at.text_input(key="ticker").set_value("AAPL")
    at.button(key="research_submit").click().run()
    assert not at.exception
    assert not at.metric
    assert any(research.return_value in item.value for item in at.text)


def test_portfolio_displays_answer_and_tool_observations_independently(backends):
    research, portfolio = backends
    at = app()
    at.radio(key="workflow").set_value("Portfolio Assistant").run()
    at.text_area(key="portfolio_question").set_value("What's in my portfolio?")
    at.button(key="portfolio_submit").click().run()
    assert not at.exception
    portfolio.assert_called_once_with("What's in my portfolio?")
    research.assert_not_called()
    assert any("Your synthetic holdings" in item.value for item in at.text)
    assert any("Current Holdings: AAPL, MSFT" in item.value for item in at.text)
    assert any("get_holdings" in item.value for item in at.text)
    at.run()
    portfolio.assert_called_once()


def test_empty_portfolio_question_and_api_failure_are_handled(backends):
    _, portfolio = backends
    at = app()
    at.radio(key="workflow").set_value("Portfolio Assistant").run()
    at.button(key="portfolio_submit").click().run()
    assert not at.exception
    assert "question" in at.error[0].value
    portfolio.assert_not_called()
    portfolio.side_effect = TimeoutError("private connection details")
    at.text_area(key="portfolio_question").set_value("Show my watchlist")
    at.button(key="portfolio_submit").click().run()
    assert not at.exception
    assert "connection" in at.error[0].value.lower()
    assert "private connection details" not in at.error[0].value


def test_portfolio_execution_limit_is_not_reported_as_success(backends):
    _, portfolio = backends
    portfolio.return_value = {
        "complete": False,
        "answer": "Agent stopped due to iteration limit or time limit.",
        "tools": [{"name": "get_holdings", "output": "Current Holdings: AAPL, MSFT"}],
    }
    at = app()
    at.radio(key="workflow").set_value("Portfolio Assistant").run()
    at.text_area(key="portfolio_question").set_value("Inspect my holdings")
    at.button(key="portfolio_submit").click().run()
    assert not at.exception
    assert not at.success
    assert any("limit" in item.value for item in at.warning)
    assert any("Current Holdings: AAPL, MSFT" in item.value for item in at.text)


def portfolio_app():
    at = app()
    at.radio(key="workflow").set_value("Portfolio Assistant").run()
    return at


def ask(at, question):
    at.text_area(key="portfolio_question").set_value(question)
    return at.button(key="portfolio_submit").click().run()


def test_followup_receives_earlier_messages_and_reset_removes_context(backends):
    _, portfolio = backends
    at = portfolio_app()
    ask(at, "What's in my portfolio?")
    first_answer = portfolio.return_value["answer"]
    ask(at, "Tell me more about that stock")
    portfolio.assert_called_with("Tell me more about that stock", history=[{
        "user": "What's in my portfolio?", "assistant": first_answer,
    }])
    assert len(at.chat_message) == 4
    count = portfolio.call_count
    at.button(key="reset_portfolio").click().run()
    assert not at.exception
    assert not at.chat_message
    assert not at.text_area(key="portfolio_question").value
    assert at.session_state["portfolio_history"] == []
    assert portfolio.call_count == count
    ask(at, "Start a new conversation")
    portfolio.assert_called_with("Start a new conversation")


def test_two_browser_sessions_do_not_share_history(backends):
    _, portfolio = backends
    first, second = portfolio_app(), portfolio_app()
    ask(first, "Remember my AAPL question")
    assert len(first.session_state["portfolio_history"]) == 1
    assert second.session_state["portfolio_history"] == []
    assert not second.chat_message
    ask(second, "What's in my portfolio?")
    portfolio.assert_called_with("What's in my portfolio?")
    assert first.session_state["portfolio_history"][0]["user"] == "Remember my AAPL question"


@pytest.mark.parametrize("outcome", ["error", "incomplete", "empty"])
def test_failed_or_incomplete_turn_does_not_enter_history(backends, outcome):
    _, portfolio = backends
    at = portfolio_app()
    ask(at, "What's in my portfolio?")
    original = [dict(turn) for turn in at.session_state["portfolio_history"]]
    if outcome == "error":
        portfolio.side_effect = TimeoutError("unavailable")
    elif outcome == "incomplete":
        portfolio.return_value = {"answer": "Iteration limit", "complete": False, "tools": []}
    else:
        portfolio.return_value = {"answer": "", "tools": []}
    ask(at, "Try another question")
    assert not at.exception
    assert at.session_state["portfolio_history"] == original


def test_ui_bounds_history_and_preserves_it_across_workflow_switches(backends):
    _, portfolio = backends
    at = portfolio_app()
    for i in range(8):
        ask(at, f"question {i}")
    history = at.session_state["portfolio_history"]
    assert len(history) == conversation.MAX_EXCHANGES
    assert history[0]["user"] == "question 2"
    assert len(at.chat_message) == 12
    count = portfolio.call_count
    at.radio(key="workflow").set_value("Stock Research").run()
    at.radio(key="workflow").set_value("Portfolio Assistant").run()
    assert at.session_state["portfolio_history"] == history
    assert portfolio.call_count == count


def test_full_latest_answer_remains_visible_while_retained_context_is_bounded(backends):
    _, portfolio = backends
    portfolio.return_value = {"answer": "A" * 2300 + "LATEST_ANSWER_END", "tools": []}
    at = portfolio_app()
    ask(at, "Give a detailed answer")
    assert not at.exception
    assert any("LATEST_ANSWER_END" in item.value for item in at.text)
    assert "LATEST_ANSWER_END" not in at.session_state["portfolio_history"][-1]["assistant"]
    assert len(at.session_state["portfolio_history"][-1]["assistant"]) <= 2000


def test_public_outputs_and_retained_history_redact_credentials(backends, monkeypatch):
    _, portfolio = backends
    monkeypatch.setenv("NEWS_API_KEY", "ui-secret-credential")
    portfolio.return_value = {
        "answer": "The provider returned ui-secret-credential",
        "tools": [{"name": "get_holdings", "output": "ui-secret-credential"}],
    }
    at = portfolio_app()
    ask(at, "Show my holdings")
    visible = [item.value for group in (at.text, at.markdown, at.code, at.caption) for item in group]
    assert all("ui-secret-credential" not in item for item in visible)
    assert "ui-secret-credential" not in str(at.session_state["portfolio_history"])


def test_model_markdown_images_are_displayed_as_inert_text(backends):
    _, portfolio = backends
    portfolio.return_value = {"answer": "![external](https://untrusted.test/track)", "tools": []}
    at = portfolio_app()
    ask(at, "Show my holdings")
    assert any("untrusted.test" in item.value for item in at.text)
    assert not any("untrusted.test" in item.value for item in at.markdown)


def test_pipeline_notes_and_tool_names_are_also_inert_text(backends):
    research, portfolio = backends
    image = "![external](https://untrusted.test/metadata)"
    research.return_value = research.return_value.replace(
        "sentiment_error: news provider unavailable", image,
    )
    at = app()
    at.text_input(key="ticker").set_value("AAPL")
    at.button(key="research_submit").click().run()
    assert not any("untrusted.test" in item.value for item in at.warning)
    portfolio.return_value = {"answer": "No write performed.", "tools": [{"name": image, "output": "Denied"}]}
    at.radio(key="workflow").set_value("Portfolio Assistant").run()
    ask(at, "Show holdings")
    assert not any("untrusted.test" in item.value for item in at.caption)


def test_failed_source_validation_shows_no_research_numbers(backends):
    from investment.safety import ResearchDataError

    research, _ = backends
    research.side_effect = ResearchDataError("Research sources are unavailable. Retry later.")
    at = app()
    at.text_input(key="ticker").set_value("BAD")
    at.button(key="research_submit").click().run()
    assert not at.exception
    assert at.error and not at.metric and not at.code
