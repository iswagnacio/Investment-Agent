"""Small adapters for the local UI; financial analysis stays in existing agents."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from anthropic import APIConnectionError, AuthenticationError, RateLimitError
from investment.conversation import bounded_history, format_history
from investment import safety


DEMO_PORTFOLIO_CSV = (
    "status,ticker,shares,cost_basis,date_acquired,notes\n"
    "hold,AAPL,10,100,2026-01-15,Synthetic demonstration holding\n"
    "hold,MSFT,5,200,2026-02-01,Synthetic demonstration holding\n"
    "watch,NVDA,0,0,,Synthetic watchlist item\n"
)


normalize_ticker = safety.normalize_ticker


def run_research(ticker: str) -> str:
    ticker = normalize_ticker(ticker)
    # Lazy imports keep the initial screen independent of agent initialization.
    from investment.agents.lead import GeneralAnalystAgent

    source_notes = []

    def validate(data):
        source_notes.extend(safety.validate_research_data(data))

    report = GeneralAnalystAgent().analyze(ticker, save=False, data_validator=validate)
    if not source_notes:
        raise safety.ResearchDataError("Source validation is unavailable. No report can be displayed. Retry later.")
    return safety.redact_text("SOURCE DATES\n" + "\n".join(source_notes) + "\n\n" + report)


def parse_report(report: str) -> dict:
    """Read display fields from the existing report, without computing any scores."""
    header, marker, analysis = report.partition("\nDETAILED ANALYSIS:\n==================\n\n")
    # Model-authored prose can repeat field names; only the renderer's header is metadata.
    metadata = header if marker else ""
    quantitative = metadata.partition("\nQUANTITATIVE SUMMARY:\n---------------------\n")[2]
    quantitative = quantitative.partition("\nPIPELINE NOTES:")[0]
    scores = dict(re.findall(
        r"^(Technical|Fundamental|Sentiment|Overall) Score:[ \t]+([0-9]+(?:\.[0-9]+)?)/100",
        quantitative, re.MULTILINE,
    ))

    def field(pattern, source=quantitative):
        match = re.search(pattern, source, re.MULTILINE)
        return (match.group(1).strip() or None) if match else None

    notes = re.search(r"\nPIPELINE NOTES:\n(.*?)\n={3,}", metadata, re.DOTALL)
    return {
        "scores": scores,
        "confidence": field(r"^Confidence:[ \t]+([0-9]+(?:\.[0-9]+)?%)"),
        "signal": field(r"^Signal:[ \t]+(.+)$"),
        "generated_at": field(r"^Analysis Date:[ \t]+(.+?)[ \t]+\(run ", metadata),
        "sources": metadata.partition("SOURCE DATES\n")[2].partition("\n\n")[0] if metadata.startswith("SOURCE DATES\n") else "",
        "notes": notes.group(1).strip() if notes else "",
        "analysis": analysis if marker else report,
    }


def run_portfolio(question: str, history: list[dict[str, str]] | None = None) -> dict:
    if not isinstance(question, str) or len(question) > safety.MAX_QUESTION_CHARS:
        raise ValueError("Enter a portfolio question of at most 2000 characters.")
    question = question.strip()
    if not question:
        raise ValueError("Enter a portfolio question.")
    from investment.agents.portfolio import PortfolioTrackerAgent

    # Each request gets its own disposable data, including when tools write.
    with TemporaryDirectory(prefix="investment-demo-") as directory:
        csv = Path(directory) / "synthetic.csv"
        csv.write_text(DEMO_PORTFOLIO_CSV, encoding="utf-8")
        agent = PortfolioTrackerAgent(csv_path=str(csv), read_only=True)
        agent.agent_executor.verbose = False
        agent.agent_executor.return_intermediate_steps = True
        inputs = {"input": safety.redact_text(question)}
        if history:
            clean_history = [
                {role: safety.redact_text(turn[role]) for role in ("user", "assistant")}
                for turn in bounded_history(history)
            ]
            inputs["chat_history"] = format_history(clean_history)
        result = agent.agent_executor.invoke(inputs)
        if not isinstance(result.get("output"), str) or not result["output"].strip():
            raise ValueError("Empty portfolio response")
        return {
            "answer": safety.redact_text(result["output"]),
            "complete": result["output"] != "Agent stopped due to iteration limit or time limit.",
            "tools": [
                {"name": safety.redact_text(action.tool), "output": safety.redact_text(str(observation))}
                for action, observation in result.get("intermediate_steps", [])
            ],
        }


def friendly_error(error: Exception, workflow: str) -> str:
    """Do not put provider exception text, URLs, or credentials on the screen."""
    if isinstance(error, safety.ResearchDataError):
        return safety.redact_text(str(error))
    if (isinstance(error, ValueError)
            and str(error) == "ANTHROPIC_API_KEY environment variable not set"):
        return "Set ANTHROPIC_API_KEY in .env or the environment, then restart Streamlit and retry."
    if isinstance(error, AuthenticationError):
        return "Anthropic rejected the credentials. Check ANTHROPIC_API_KEY, restart Streamlit, and retry."
    if isinstance(error, RateLimitError):
        return "The model service is rate limited. Check your quota and retry later."
    if isinstance(error, (APIConnectionError, TimeoutError)):
        return f"{workflow} could not reach a required service. Check your connection and retry."
    return (
        f"{workflow} could not complete. Check your API credentials, data-provider "
        "access, network, and service quotas, then retry."
    )
