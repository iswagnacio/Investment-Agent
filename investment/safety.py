"""Public-demo boundaries: input validation, credential redaction and source checks."""

import math
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, quote_plus


MAX_QUESTION_CHARS = 2000


class ResearchDataError(ValueError):
    """A safe, source-independent message for unavailable public research."""


def normalize_ticker(value: str) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("Enter one stock ticker, for example AAPL or BRK-B.")
    ticker = value.strip().upper()
    if len(ticker) > 15 or not re.fullmatch(r"[A-Z0-9]+(?:[.-][A-Z0-9]+)?", ticker):
        raise ValueError("Enter one stock ticker, for example AAPL or BRK-B.")
    return ticker


def redact_text(value: str) -> str:
    """Redact configured credentials and common credential-bearing error syntax."""
    text = str(value)
    secrets = {
        secret for name, secret in os.environ.items()
        if secret and re.search(r"API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.IGNORECASE)
    }
    for secret in sorted(secrets, key=len, reverse=True):
        for variant in {secret, quote(secret, safe=""), quote_plus(secret)}:
            text = text.replace(variant, "[redacted]")
    text = re.sub(r"sk-(?:ant-)?[A-Za-z0-9_-]{12,}", "[redacted]", text)
    text = re.sub(r"(?i)(\bBearer\s+)[^\s\"'<>]+", r"\1[redacted]", text)
    return re.sub(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|token|secret|password)\s*[=:]\s*)[^\s&\"'<>]+",
        r"\1[redacted]", text,
    )


def _timestamp(value):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def _recent(value, now, max_age):
    parsed = _timestamp(value)
    return parsed is not None and -timedelta(minutes=5) <= now - parsed <= max_age


def _finite(value, positive=False):
    try:
        number = float(value)
        return not isinstance(value, bool) and math.isfinite(number) and (not positive or number > 0)
    except (ValueError, TypeError, OverflowError):
        return False


def _statement_date(statements, now):
    if not isinstance(statements, list) or not statements:
        return None
    row = statements[0]  # The calculator consumes row[0] as the current statement.
    if (isinstance(row, dict) and isinstance(row.get("data"), dict)
            and any(_finite(value) and float(value) != 0 for value in row["data"].values())
            and _recent(row.get("date"), now, timedelta(days=550))):
        return _timestamp(row["date"])
    return None


def _usable_specialist(domain, structured, ticker):
    if not isinstance(structured, dict) or structured.get("ticker") != ticker or structured.get("error"):
        return False
    analysis = structured.get("llm_analysis")
    fields = {
        "technical": {
            "trend_verdict": ("Bullish", "Bearish", "Neutral"),
            "momentum_verdict": ("Bullish", "Bearish", "Neutral", "Oversold", "Overbought"),
            "volatility_verdict": ("Low", "Normal", "High", "Extreme"),
            "volume_verdict": ("Accumulation", "Distribution", "Normal", "Low Interest"),
        },
        "fundamental": {
            "valuation_verdict": ("Undervalued", "Fairly Valued", "Overvalued", "Extremely Overvalued"),
            "quality_verdict": ("High Quality", "Medium Quality", "Low Quality"),
            "growth_verdict": ("High Growth", "Moderate Growth", "Stable", "Declining"),
            "financial_health": ("Strong", "Adequate", "Weak"),
        },
        "sentiment": {"overall_tone": ("Bullish", "Bearish", "Neutral", "Mixed"),
                      "tone_direction": ("Improving", "Stable", "Deteriorating")},
    }[domain]
    valid = isinstance(analysis, dict) and all(
        isinstance(analysis.get(field), dict) and isinstance(analysis[field].get("signal"), str)
        and analysis[field]["signal"].strip().casefold() in {item.casefold() for item in options}
        for field, options in fields.items()
    )
    if valid and domain == "sentiment":
        valid = analysis['overall_tone'].get('confidence') in ('High', 'Medium', 'Low') and all(
            isinstance(analysis.get(key), dict) and isinstance(analysis[key].get('detected'), bool)
            for key in ('controversy_flag', 'catalyst_detected')
        )
    return valid


def validate_research_data(data: dict, now: datetime | None = None) -> list[str]:
    """Reject unusable core sources before fallback scores or final synthesis run.

    These are demo freshness policies, not changes to analytical calculations.
    Retrieval timestamps are distinguished from price bars and statement periods.
    """
    now = now or datetime.now(timezone.utc)
    failed, notes = [], []
    for domain in ("technical", "fundamental", "sentiment"):
        structured = data.get(f"{domain}_structured")
        if (data.get(f"{domain}_error") or data.get(f"{domain}_structured_error")
                or not _usable_specialist(domain, structured, data.get("ticker"))):
            failed.append(domain)

    technical = data.get("technical") or {}
    price = technical.get("price", {}).get("current")
    sma = technical.get("moving_averages", {}).get("sma_20")
    rsi = technical.get("momentum", {}).get("rsi_14")
    required_values = [
        technical.get("moving_averages", {}).get("sma_50"),
        technical.get("momentum", {}).get("momentum_10d"),
        technical.get("trend", {}).get("adx"),
        *[technical.get("trend", {}).get("macd", {}).get(key) for key in ("macd", "signal", "histogram")],
        *[technical.get("volume", {}).get(key) for key in ("ratio", "trend")],
        technical.get("volatility", {}).get("historical_vol"),
        *[technical.get("volatility", {}).get("bollinger", {}).get(key) for key in ("upper", "lower")],
    ]
    sma200 = technical.get("moving_averages", {}).get("sma_200")
    if (not _finite(price, positive=True) or not _finite(sma, positive=True)
            or not _finite(rsi) or not 0 <= float(rsi) <= 100
            or not all(_finite(value) for value in required_values)
            or (sma200 is not None and not _finite(sma200, positive=True))
            or not _recent(technical.get("as_of"), now, timedelta(days=7))):
        failed.append("technical")
    else:
        notes.append(f"Price history last bar: {_timestamp(technical['as_of']).isoformat()}")

    fundamental = data.get("fundamental") or {}
    statements = fundamental.get("financial_statements") or {}
    income = _statement_date(statements.get("income_annual"), now)
    balance = _statement_date(statements.get("balance_annual"), now)
    cash_flow = _statement_date(statements.get("cash_flow_annual"), now)
    company = fundamental.get("company_info") or {}
    stats = fundamental.get("key_stats") or {}
    if (fundamental.get("ticker") != data.get("ticker") or company.get("ticker") != data.get("ticker")
            or not income or not balance or not cash_flow
            or not _finite(stats.get("current_price"), positive=True)
            or not _finite(company.get("market_cap"), positive=True)
            or not _recent(fundamental.get("fetch_time"), now, timedelta(hours=24))):
        failed.append("fundamental")
    else:
        notes.append(f"Fundamentals retrieved: {_timestamp(fundamental['fetch_time']).isoformat()}; "
                     f"income period: {income.date()}; balance period: {balance.date()}; "
                     f"cash-flow period: {cash_flow.date()}")

    sentiment = data.get("sentiment") or {}
    news = sentiment.get("company_news") or {}
    articles = news.get("articles") or []
    published = []
    for article in articles:
        metadata = article.get("metadata", {}) if isinstance(article, dict) else getattr(article, "metadata", {})
        value = article.get("published_at") if isinstance(article, dict) else getattr(article, "published_at", None)
        published.append(value if metadata.get("publication_date_verified") is True else None)
    recent = [_timestamp(value) for value in published if _recent(value, now, timedelta(days=7))]
    if (sentiment.get("ticker") != data.get("ticker") or not news.get("success")
            or not recent or len(recent) != len(articles)):
        failed.append("sentiment")
    else:
        notes.append(f"Newest company article: {max(recent).isoformat()}")
        if news.get("errors"):
            notes.append("Some news sources were unavailable; available company articles were used.")

    if failed:
        domains = ", ".join(sorted(set(failed)))
        raise ResearchDataError(
            f"Research data is missing, invalid, or out of date for: {domains}. "
            "No scores or final analysis were generated. Check data-provider availability and retry."
        )
    return notes
