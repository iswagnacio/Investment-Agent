"""
Report Rendering
================
Human-readable views over AnalysisRecord (contracts.py). The record is the
product; everything here is presentation. Nothing in this module computes —
if a number isn't in the record, it doesn't belong in the report.
"""

from contracts import AnalysisRecord


def render_report(record: AnalysisRecord) -> str:
    """Full single-stock report (what lead_agent.analyze returns/prints)."""
    lines = [
        "=" * 80,
        f"COMPREHENSIVE INVESTMENT ANALYSIS: {record.ticker}",
        "=" * 80,
        f"Company: {record.company or record.ticker}",
        f"Sector: {record.sector or 'Unknown'}",
        f"Industry: {record.industry or 'Unknown'}",
        f"Analysis Date: {record.generated_at}   (run {record.run_id})",
        "=" * 80,
        "",
        "QUANTITATIVE SUMMARY:",
        "---------------------",
    ]

    core_order = ("technical", "fundamental", "sentiment", "perspective")
    extras = sorted(k for k in record.signals if k not in core_order)
    for domain in (*core_order, *extras):
        sig = record.signals.get(domain)
        if not sig or sig.get("score") is None:
            continue
        label = f"{domain.capitalize()} Score:"
        lines.append(f"{label:<20}{sig['score']:>6.1f}/100"
                     + (f"   [{sig['verdict']}]" if sig.get("verdict") else ""))
    lines += [
        f"{'':>20}---------",
        f"{'Overall Score:':<20}{record.overall_score:>6.1f}/100",
        "",
        f"Signal: {record.signal}",
        f"Confidence: {record.confidence:.0f}%",
    ]

    if record.weights:
        blend = "  ".join(f"{k}={v}" for k, v in record.weights.items() if v)
        lines += ["", f"Blend weights: {blend}"]

    if record.data_flags:
        lines += ["", "=" * 80, "PIPELINE NOTES:"]
        lines += [f"- {flag}" for flag in record.data_flags]

    lines += ["=" * 80, "", "DETAILED ANALYSIS:", "==================", "", record.thesis]

    if record.key_risks:
        lines += ["", "KEY RISKS:", *[f"- {r}" for r in record.key_risks]]

    return "\n".join(lines)


def render_quick_line(record: AnalysisRecord) -> str:
    """One-line summary for comparisons / screens."""
    price = f"${record.current_price:.2f}" if record.current_price else "n/a"
    return (f"{record.ticker:<8}{(record.company or '')[:24]:<26}{price:>10}"
            f"{record.overall_score:>8.1f}  {record.signal:<12}{record.confidence:>4.0f}%")
