"""Local Streamlit interface for the two existing, independent agent workflows."""

import streamlit as st

from investment import web
from investment.safety import MAX_QUESTION_CHARS, redact_text
from investment.conversation import add_exchange, bounded_history


def show_research():
    st.subheader("Stock Research")
    st.write("Get a consolidated report from the technical, fundamental, and sentiment analysts.")
    with st.form("research_form"):
        ticker = st.text_input("Stock ticker", placeholder="AAPL", key="ticker", max_chars=15)
        submitted = st.form_submit_button("Run research", key="research_submit", type="primary")

    if submitted:
        st.session_state.pop("research_report", None)
        st.session_state.pop("research_ticker", None)
        st.session_state.pop("research_error", None)
        try:
            normalized = web.normalize_ticker(ticker)
        except ValueError as error:
            st.session_state["research_error"] = str(error)
        else:
            try:
                with st.spinner(f"Researching {normalized}. Data retrieval and analysis may take a few minutes…"):
                    report = web.run_research(normalized)
                    if not isinstance(report, str) or not report.strip():
                        raise ValueError("Empty research response")
                    st.session_state["research_report"] = redact_text(report)
                    st.session_state["research_ticker"] = normalized
            except Exception as error:
                st.session_state["research_error"] = web.friendly_error(error, "Research")

    if st.session_state.get("research_error"):
        st.error(st.session_state["research_error"])
    report = redact_text(st.session_state.get("research_report") or "")
    if not report:
        return

    result = web.parse_report(report)
    st.success("Research complete")
    st.subheader(f"{st.session_state['research_ticker']} research report")
    if result["generated_at"]:
        st.caption(f"Generated: {result['generated_at']}")
    st.caption("Model-generated assessment. Report generation time is not the time of a verified live quote.")
    if result["sources"]:
        st.text(result["sources"])
    else:
        st.warning("Source timestamps are unavailable for this report; freshness is unverified.")
    columns = st.columns(4)
    for column, label in zip(columns, ("Technical", "Fundamental", "Sentiment", "Overall")):
        if label in result["scores"]:
            column.metric(label, f"{result['scores'][label]} / 100")
    columns = st.columns(2)
    for column, label in zip(columns, ("Confidence", "Signal")):
        if result[label.lower()] is not None:
            column.metric(label, result[label.lower()])
    if result["notes"]:
        st.warning("Pipeline notes — check data availability before using these results.")
        st.text(result["notes"])
    st.subheader("Final analysis")
    st.text(redact_text(result["analysis"]))
    with st.expander("Full consolidated report"):
        st.code(report, language=None, wrap_lines=True)


def show_portfolio():
    st.subheader("Portfolio Assistant")
    st.info("Read-only synthetic demo: AAPL and MSFT holdings; NVDA watchlist. Portfolio data resets for each request.")
    st.write("Ask about the demo portfolio, then ask a follow-up about the same holding.")
    st.caption("Remembers up to six exchanges in this session. Long messages are shortened in retained history.")
    st.session_state["portfolio_history"] = bounded_history(st.session_state.get("portfolio_history", []))
    if st.button("Reset conversation", key="reset_portfolio"):
        st.session_state["portfolio_history"] = []
        st.session_state["portfolio_question"] = ""
        st.session_state.pop("portfolio_result", None)
        st.session_state.pop("portfolio_error", None)
    with st.form("portfolio_form"):
        question = st.text_area(
            "Portfolio question", placeholder="What's in my portfolio?", key="portfolio_question",
            max_chars=MAX_QUESTION_CHARS,
        )
        submitted = st.form_submit_button("Ask assistant", key="portfolio_submit", type="primary")

    if submitted:
        st.session_state.pop("portfolio_result", None)
        st.session_state.pop("portfolio_error", None)
        if not question.strip():
            st.session_state["portfolio_error"] = "Enter a portfolio question."
        else:
            try:
                with st.spinner("Checking the synthetic portfolio…"):
                    history = st.session_state["portfolio_history"]
                    if history:
                        result = web.run_portfolio(question.strip(), history=history)
                    else:
                        result = web.run_portfolio(question.strip())
                    if not isinstance(result.get("answer"), str) or not result["answer"].strip():
                        raise ValueError("Empty portfolio response")
                    result = {
                        **result, "answer": redact_text(result["answer"]),
                        "tools": [{"name": redact_text(tool["name"]), "output": redact_text(tool["output"])}
                                  for tool in result["tools"]],
                    }
                    st.session_state["portfolio_result"] = result
                    if result.get("complete", True):
                        st.session_state["portfolio_history"] = add_exchange(
                            history, redact_text(question.strip()), result["answer"],
                        )
            except Exception as error:
                st.session_state["portfolio_error"] = web.friendly_error(error, "Portfolio assistant")

    if st.session_state.get("portfolio_error"):
        st.error(st.session_state["portfolio_error"])
    history = st.session_state["portfolio_history"]
    result = st.session_state.get("portfolio_result")
    for index, turn in enumerate(history):
        with st.chat_message("user"):
            st.text(redact_text(turn["user"]))
        with st.chat_message("assistant"):
            if result and result.get("complete", True) and index == len(history) - 1:
                st.text(redact_text(result["answer"]))
            else:
                st.text(redact_text(turn["assistant"]))
    if not result:
        return

    if result.get("complete", True):
        st.success("Portfolio request complete")
        if not history:
            st.subheader("Answer")
            st.text(redact_text(result["answer"]))
    else:
        st.warning("The assistant reached its execution limit before answering. Try a more focused question.")
    st.subheader("Tool results")
    if not result["tools"]:
        st.info("The assistant answered without calling a portfolio tool.")
    for tool in result["tools"]:
        st.text(redact_text(tool["name"]))
        st.text(redact_text(tool["output"]))


def main():
    st.set_page_config(page_title="Investment Agent", page_icon="📊", layout="wide")
    st.title("Investment Agent")
    st.caption("Specialist stock research · Synthetic portfolio assistant")
    st.sidebar.subheader("Local demonstration")
    st.sidebar.write("Submit a request to run an agent. Model requests use your configured API account.")
    st.sidebar.caption("Research scores are analytical assessments. Review source availability and report dates.")
    workflow = st.radio(
        "Workflow", ("Stock Research", "Portfolio Assistant"), horizontal=True, key="workflow",
    )
    if workflow == "Stock Research":
        show_research()
    else:
        show_portfolio()


if __name__ == "__main__":
    main()
