"""
Intelligent Analyzer - Uses LLM to select and calculate relevant metrics.

This module:
1. Identifies the company's sector
2. Uses LLM to select relevant metrics for that sector
3. Fetches available data from EDGAR
4. Uses LLM to figure out how to calculate missing metrics
5. Generates a comprehensive analysis
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fundamentals.edgar_fetch import EdgarFetcher
from fundamentals.xbrl_parse import XBRLParser, FinancialStatements
from fundamentals.metrics import (
    METRICS_CATALOG,
    MetricDefinition,
    Sector,
    get_sector_from_sic,
    get_metrics_for_sector,
    get_all_required_inputs,
)
from config import config


# --- LLM Integration ---

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class AnalysisRequest:
    """Request for intelligent analysis."""
    ticker: str
    stock_price: Optional[float] = None
    focus_areas: Optional[list[str]] = None  # e.g., ["profitability", "growth"]
    compare_to: Optional[list[str]] = None  # Peer tickers


@dataclass
class MetricResult:
    """Result of calculating a single metric."""
    name: str
    value: Optional[float]
    unit: str
    description: str
    formula_used: str
    data_source: str  # "xbrl_direct", "calculated", "unavailable"
    notes: Optional[str] = None


@dataclass 
class IntelligentAnalysis:
    """Complete analysis result."""
    ticker: str
    company_name: str
    sector: Sector
    analysis_date: str
    
    # LLM-selected metrics and results
    selected_metrics: list[str]
    metric_results: dict[str, MetricResult]
    
    # LLM-generated insights
    summary: str
    key_strengths: list[str]
    key_concerns: list[str]
    sector_specific_notes: str
    
    # Raw data availability
    available_data: list[str]
    missing_data: list[str]


class IntelligentAnalyzer:
    """
    LLM-powered fundamental analyzer.
    
    Usage:
        analyzer = IntelligentAnalyzer(
            user_email="you@example.com",
            anthropic_api_key="sk-..."  # Or set ANTHROPIC_API_KEY env var
        )
        
        analysis = analyzer.analyze("AAPL", stock_price=175.50)
        print(analysis.summary)
    """
    
    def __init__(
        self,
        user_email: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the analyzer.

        Args:
            user_email: Email for SEC API (defaults to config)
            anthropic_api_key: Anthropic API key (defaults to config)
            model: Claude model to use (defaults to config.FAST_MODEL)
        """
        # Use config defaults if not provided
        user_email = user_email or config.USER_EMAIL
        self.model = model or config.FAST_MODEL

        self.fetcher = EdgarFetcher(user_email)
        self.parser = XBRLParser(user_email)

        # Initialize Anthropic client
        api_key = anthropic_api_key or config.get_anthropic_api_key()
        if api_key and HAS_ANTHROPIC:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.use_llm = True
        else:
            self.client = None
            self.use_llm = False
            if not HAS_ANTHROPIC:
                print("Warning: anthropic package not installed. Using rule-based fallback.")
            else:
                print("Warning: No API key provided. Using rule-based fallback.")
    
    def _call_llm(self, system: str, user: str) -> str:
        """Make a call to Claude."""
        if not self.use_llm or self.client is None:
            return ""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text": 
                text_parts.append(block.text)
        return "\n".join(text_parts)
    
    def _get_company_context(self, ticker: str) -> tuple[str, Sector, str]:
        """Get company name, sector, and description."""
        info = self.fetcher.get_company_info(ticker)
        if not info:
            return ticker, Sector.UNKNOWN, ""
        
        sector = get_sector_from_sic(info.sic)
        return info.name, sector, info.sic_description
    
    def _extract_available_data(
        self, 
        statements: FinancialStatements,
        stock_price: Optional[float] = None
    ) -> dict[str, float]:
        """
        Extract all available data points from financial statements.
        Returns a flat dict of metric_name -> latest_value.
        """
        data = {}
        
        # Helper to get latest annual value
        def get_latest(facts_list, key_name):
            annual = [f for f in facts_list if f.form == "10-K"]
            if annual:
                data[key_name] = annual[0].value
        
        # Income Statement
        get_latest(statements.revenue, "revenue")
        get_latest(statements.net_income, "net_income")
        get_latest(statements.gross_profit, "gross_profit")
        get_latest(statements.operating_income, "operating_income")
        get_latest(statements.cost_of_revenue, "cost_of_revenue")
        get_latest(statements.eps_diluted, "eps")
        if "eps" not in data:
            get_latest(statements.eps_basic, "eps")
        
        # Operating expenses
        get_latest(statements.rd_expense, "rd_expense")
        get_latest(statements.sga_expense, "sga_expense")
        
        # Balance Sheet
        get_latest(statements.total_assets, "total_assets")
        get_latest(statements.total_liabilities, "total_liabilities")
        get_latest(statements.stockholders_equity, "stockholders_equity")
        get_latest(statements.cash_and_equivalents, "cash_and_equivalents")
        get_latest(statements.total_debt, "total_debt")
        get_latest(statements.current_assets, "current_assets")
        get_latest(statements.current_liabilities, "current_liabilities")
        get_latest(statements.inventory, "inventory")
        get_latest(statements.accounts_receivable, "accounts_receivable")
        get_latest(statements.accounts_payable, "accounts_payable")
        get_latest(statements.goodwill, "goodwill")
        get_latest(statements.intangible_assets, "intangible_assets")
        
        # Financial sector
        get_latest(statements.interest_income, "interest_income")
        get_latest(statements.interest_expense, "interest_expense")
        
        # Cash Flow
        get_latest(statements.operating_cash_flow, "operating_cash_flow")
        get_latest(statements.capital_expenditures, "capital_expenditures")
        get_latest(statements.dividends_paid, "dividends_paid")
        get_latest(statements.shares_outstanding, "shares_outstanding")
        
        # Add stock price if provided
        if stock_price:
            data["stock_price"] = stock_price
        
        # Calculate derived values that might be needed
        if "stockholders_equity" not in data and "total_assets" in data and "total_liabilities" in data:
            data["stockholders_equity"] = data["total_assets"] - data["total_liabilities"]
        
        # Calculate EBITDA if we have the components
        # EBITDA = Operating Income + Depreciation (approximation)
        if "operating_income" in data:
            # We don't have depreciation directly, so use operating income as proxy
            data["ebitda"] = data["operating_income"]  # Simplified
        
        # Calculate revenue growth if we have history
        annual_revenue = [f for f in statements.revenue if f.form == "10-K"]
        if len(annual_revenue) >= 2:
            current = annual_revenue[0].value
            prior = annual_revenue[1].value
            if prior > 0:
                data["revenue_growth_rate"] = (current - prior) / prior
        
        # Calculate operating margin for Rule of 40
        if "operating_income" in data and "revenue" in data and data["revenue"] > 0:
            data["operating_margin"] = data["operating_income"] / data["revenue"]
        
        return data
    
    def _select_metrics_with_llm(
        self, 
        sector: Sector, 
        company_name: str,
        industry_desc: str,
        available_data: list[str],
        focus_areas: Optional[list[str]] = None
    ) -> list[str]:
        """Use LLM to select the most relevant metrics."""
        
        if not self.use_llm:
            # Fallback: return sector-appropriate metrics
            return get_metrics_for_sector(sector)[:15]
        
        # Build the prompt
        all_metrics = {name: {
            "description": defn.description,
            "required_inputs": defn.required_inputs,
            "sectors": [s.value for s in defn.sectors] if defn.sectors else ["all"],
            "priority": defn.priority
        } for name, defn in METRICS_CATALOG.items()}
        
        system = """You are a financial analyst assistant. Your job is to select the most relevant 
financial metrics to analyze for a given company based on its sector and available data.

Return your response as a JSON object with this structure:
{
    "selected_metrics": ["metric_name_1", "metric_name_2", ...],
    "reasoning": "Brief explanation of why these metrics matter for this company"
}

Select 10-15 metrics. Prioritize:
1. Universal metrics that apply to all companies (margins, ROE, debt ratios)
2. Sector-specific metrics that are particularly important for this industry
3. Metrics where we have the required data available

Do NOT select metrics where key required inputs are missing unless they're critical."""

        user = f"""Company: {company_name}
Sector: {sector.value}
Industry: {industry_desc}

Available data points: {json.dumps(available_data)}

Focus areas requested: {focus_areas or "None specified - provide balanced analysis"}

Available metrics catalog:
{json.dumps(all_metrics, indent=2)}

Select the most relevant metrics for analyzing this company."""

        response = self._call_llm(system, user)
        
        try:
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            result = json.loads(response)
            return result.get("selected_metrics", get_metrics_for_sector(sector)[:15])
        except (json.JSONDecodeError, KeyError):
            # Fallback
            return get_metrics_for_sector(sector)[:15]
    
    def _calculate_metrics(
        self, 
        metric_names: list[str], 
        available_data: dict[str, float]
    ) -> dict[str, MetricResult]:
        """Calculate all selected metrics."""
        results = {}
        
        for name in metric_names:
            if name not in METRICS_CATALOG:
                continue
            
            defn = METRICS_CATALOG[name]
            
            # Check if we have required inputs
            missing_inputs = [inp for inp in defn.required_inputs if inp not in available_data]
            
            if missing_inputs:
                results[name] = MetricResult(
                    name=defn.name,
                    value=None,
                    unit=defn.unit,
                    description=defn.description,
                    formula_used=defn.formula_description,
                    data_source="unavailable",
                    notes=f"Missing: {', '.join(missing_inputs)}"
                )
            else:
                # Calculate the metric
                try:
                    value = defn.calculate(available_data)
                    results[name] = MetricResult(
                        name=defn.name,
                        value=value,
                        unit=defn.unit,
                        description=defn.description,
                        formula_used=defn.formula_description,
                        data_source="calculated"
                    )
                except Exception as e:
                    results[name] = MetricResult(
                        name=defn.name,
                        value=None,
                        unit=defn.unit,
                        description=defn.description,
                        formula_used=defn.formula_description,
                        data_source="unavailable",
                        notes=f"Calculation error: {str(e)}"
                    )
        
        return results
    
    def _generate_insights_with_llm(
        self,
        company_name: str,
        sector: Sector,
        metric_results: dict[str, MetricResult]
    ) -> tuple[str, list[str], list[str], str]:
        """Use LLM to generate insights from calculated metrics."""
        
        if not self.use_llm:
            # Basic fallback summary
            calculated = {k: v for k, v in metric_results.items() if v.value is not None}
            return (
                f"Analysis of {company_name} based on {len(calculated)} metrics.",
                ["Data successfully retrieved from SEC filings"],
                ["Some metrics unavailable due to missing data"],
                f"Standard {sector.value} sector analysis applied."
            )
        
        # Prepare metrics for LLM
        metrics_summary = {}
        for name, result in metric_results.items():
            if result.value is not None:
                # Format the value appropriately
                if result.unit == "percent":
                    formatted = f"{result.value:.1%}"
                elif result.unit == "ratio":
                    formatted = f"{result.value:.2f}"
                elif result.unit == "currency":
                    formatted = f"${result.value:,.0f}"
                elif result.unit == "days":
                    formatted = f"{result.value:.0f} days"
                else:
                    formatted = f"{result.value:.2f}"
                
                metrics_summary[name] = {
                    "display_name": result.name,
                    "value": formatted,
                    "description": result.description
                }
        
        system = """You are a senior financial analyst. Based on the calculated metrics, 
provide a concise analysis of the company.

Return your response as a JSON object:
{
    "summary": "2-3 sentence executive summary of the company's financial health",
    "key_strengths": ["strength 1", "strength 2", "strength 3"],
    "key_concerns": ["concern 1", "concern 2"],
    "sector_notes": "1-2 sentences on how the company compares to typical sector characteristics"
}

Be specific and reference actual metric values. Be balanced - note both positives and negatives."""

        user = f"""Company: {company_name}
Sector: {sector.value}

Calculated Metrics:
{json.dumps(metrics_summary, indent=2)}

Provide your analysis."""

        response = self._call_llm(system, user)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            result = json.loads(response)
            return (
                result.get("summary", "Analysis complete."),
                result.get("key_strengths", []),
                result.get("key_concerns", []),
                result.get("sector_notes", "")
            )
        except (json.JSONDecodeError, KeyError):
            return (
                f"Analysis of {company_name} complete.",
                ["Financial data retrieved successfully"],
                [],
                f"Analyzed as {sector.value} sector company."
            )
    
    def analyze(
        self, 
        ticker: str, 
        stock_price: Optional[float] = None,
        focus_areas: Optional[list[str]] = None
    ) -> Optional[IntelligentAnalysis]:
        """
        Perform intelligent analysis of a company.
        
        Args:
            ticker: Stock ticker symbol
            stock_price: Current stock price (for valuation metrics)
            focus_areas: Optional focus areas like ["profitability", "growth", "debt"]
            
        Returns:
            IntelligentAnalysis object or None if data unavailable
        """
        print(f"[1/5] Getting company information for {ticker}...")
        company_name, sector, industry_desc = self._get_company_context(ticker)
        
        if sector == Sector.UNKNOWN:
            print(f"  Warning: Could not determine sector for {ticker}")
        else:
            print(f"  Identified sector: {sector.value}")
        
        print(f"[2/5] Fetching financial data from SEC EDGAR...")
        statements = self.parser.get_financial_statements(ticker)
        if not statements:
            print(f"  Error: Could not fetch financial data for {ticker}")
            return None
        
        print(f"[3/5] Extracting available data points...")
        available_data = self._extract_available_data(statements, stock_price)
        print(f"  Found {len(available_data)} data points")
        
        print(f"[4/5] Selecting relevant metrics" + (" with LLM..." if self.use_llm else "..."))
        selected_metrics = self._select_metrics_with_llm(
            sector, company_name, industry_desc, 
            list(available_data.keys()), focus_areas
        )
        print(f"  Selected {len(selected_metrics)} metrics to analyze")
        
        print(f"[5/5] Calculating metrics and generating insights...")
        metric_results = self._calculate_metrics(selected_metrics, available_data)
        
        # Count successful calculations
        calculated_count = sum(1 for r in metric_results.values() if r.value is not None)
        print(f"  Successfully calculated {calculated_count}/{len(metric_results)} metrics")
        
        # Generate insights
        summary, strengths, concerns, sector_notes = self._generate_insights_with_llm(
            company_name, sector, metric_results
        )
        
        # Determine what data was missing
        all_required = get_all_required_inputs(selected_metrics)
        missing = [inp for inp in all_required if inp not in available_data]
        
        return IntelligentAnalysis(
            ticker=ticker.upper(),
            company_name=company_name,
            sector=sector,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            selected_metrics=selected_metrics,
            metric_results=metric_results,
            summary=summary,
            key_strengths=strengths,
            key_concerns=concerns,
            sector_specific_notes=sector_notes,
            available_data=list(available_data.keys()),
            missing_data=missing
        )
    
    def format_report(self, analysis: IntelligentAnalysis) -> str:
        """Format analysis as a readable report."""
        lines = [
            "=" * 60,
            f"  INTELLIGENT ANALYSIS: {analysis.ticker}",
            "=" * 60,
            f"Company: {analysis.company_name}",
            f"Sector: {analysis.sector.value.title()}",
            f"Date: {analysis.analysis_date}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            analysis.summary,
            "",
        ]
        
        if analysis.key_strengths:
            lines.append("KEY STRENGTHS")
            lines.append("-" * 40)
            for s in analysis.key_strengths:
                lines.append(f"  ✓ {s}")
            lines.append("")
        
        if analysis.key_concerns:
            lines.append("KEY CONCERNS")
            lines.append("-" * 40)
            for c in analysis.key_concerns:
                lines.append(f"  ⚠ {c}")
            lines.append("")
        
        if analysis.sector_specific_notes:
            lines.append("SECTOR CONTEXT")
            lines.append("-" * 40)
            lines.append(f"  {analysis.sector_specific_notes}")
            lines.append("")
        
        lines.append("METRICS DETAIL")
        lines.append("-" * 40)
        
        for name, result in analysis.metric_results.items():
            if result.value is not None:
                if result.unit == "percent":
                    formatted = f"{result.value:.1%}"
                elif result.unit == "ratio":
                    formatted = f"{result.value:.2f}"
                elif result.unit == "currency":
                    if abs(result.value) >= 1e9:
                        formatted = f"${result.value/1e9:.1f}B"
                    elif abs(result.value) >= 1e6:
                        formatted = f"${result.value/1e6:.1f}M"
                    else:
                        formatted = f"${result.value:,.0f}"
                elif result.unit == "days":
                    formatted = f"{result.value:.0f} days"
                else:
                    formatted = f"{result.value:.2f}"
                
                lines.append(f"  {result.name}: {formatted}")
        
        # Show unavailable metrics
        unavailable = [r for r in analysis.metric_results.values() if r.value is None]
        if unavailable:
            lines.append("")
            lines.append("UNAVAILABLE METRICS")
            lines.append("-" * 40)
            for r in unavailable[:5]:  # Limit to 5
                lines.append(f"  {r.name}: {r.notes}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# --- Demo ---

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    price = float(sys.argv[2]) if len(sys.argv) > 2 else None

    # Use config defaults
    analyzer = IntelligentAnalyzer()

    print(f"\nAnalyzing {ticker}...\n")
    analysis = analyzer.analyze(ticker, stock_price=price)

    if analysis:
        print("\n" + analyzer.format_report(analysis))
    else:
        print(f"Failed to analyze {ticker}")