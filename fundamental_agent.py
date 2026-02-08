"""
Fundamental Analyst Agent Module - FIXED VERSION
=================================
FIXED: Strong ticker anchoring throughout all prompts to prevent LLM hallucination

Uses LLM to intelligently analyze fundamental data based on company
characteristics (sector, industry, size, business model).

Features:
- Intelligent indicator selection based on company type
- Sector-specific analysis frameworks
- Valuation vs historical and peer comparisons
- Earnings quality and financial health assessment
- Dividend sustainability analysis
- Comprehensive investment thesis generation

Usage:
    from fundamental_agent import FundamentalAnalystAgent
    
    agent = FundamentalAnalystAgent()
    analysis = agent.analyze("AAPL")
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

from fundamental import FundamentalCalculator

load_dotenv()


class FundamentalAnalystAgent:
    """
    Uses LLM to intelligently analyze fundamental data.
    Selects relevant metrics based on company characteristics.
    
    FIXED: All prompts now strongly anchor to the ticker being analyzed.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-5"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.calculator = FundamentalCalculator()
        self.llm = ChatAnthropic(model_name=model, temperature=0, max_tokens=4096)
    
    def analyze(self, ticker: str, analysis_type: str = "comprehensive") -> str:
        """
        Perform intelligent fundamental analysis.
        
        Args:
            ticker: Stock ticker symbol
            analysis_type: Type of analysis
                - "comprehensive": Full fundamental analysis
                - "valuation": Focus on valuation metrics
                - "quality": Focus on financial quality
                - "dividend": Focus on dividend sustainability
                - "growth": Focus on growth metrics
                - "for_synthesis": Structured output for lead agent
        
        Returns:
            Detailed fundamental analysis report
        """
        ticker = ticker.upper()
        
        # Fetch fundamental data
        data = self.calculator.get_all_fundamentals(ticker)
        
        # Get formatted data for LLM
        formatted_data = self.calculator.get_formatted_for_llm(ticker)
        
        # Select appropriate prompt based on analysis type
        if analysis_type == "valuation":
            prompt = self._create_valuation_prompt()
        elif analysis_type == "quality":
            prompt = self._create_quality_prompt()
        elif analysis_type == "dividend":
            prompt = self._create_dividend_prompt()
        elif analysis_type == "growth":
            prompt = self._create_growth_prompt()
        elif analysis_type == "for_synthesis":
            prompt = self._create_synthesis_prompt()
        else:
            prompt = self._create_comprehensive_prompt()
        
        # Get company info for context
        company_info = data.get('company_info', {})
        
        # Invoke LLM with STRONG ticker verification
        response = self.llm.invoke(prompt.format(
            ticker=ticker,
            company_name=company_info.get('name', ticker),
            sector=company_info.get('sector', 'Unknown'),
            industry=company_info.get('industry', 'Unknown'),
            market_cap=company_info.get('market_cap', 0),
            fundamental_data=formatted_data,
            raw_data=json.dumps({
                'valuation': data.get('valuation_ratios', {}),
                'profitability': data.get('profitability_ratios', {}),
                'liquidity': data.get('liquidity_ratios', {}),
                'leverage': data.get('leverage_ratios', {}),
                'efficiency': data.get('efficiency_ratios', {}),
                'growth': data.get('growth_metrics', {}),
                'dividends': data.get('dividend_metrics', {}),
                'quality': data.get('quality_scores', {}),
                'peer_comparison': data.get('peer_comparison', {}).get('vs_peers', {}),
            }, indent=2, default=str)
        ))
        
        # Handle response content
        if isinstance(response.content, str):
            analysis = response.content
        elif isinstance(response.content, list):
            text_parts = [item for item in response.content if isinstance(item, str)]
            analysis = '\n'.join(text_parts) if text_parts else str(response.content)
        else:
            analysis = str(response.content)
        
        # CRITICAL: Verify the analysis is actually for the requested ticker
        ticker_count = analysis.upper().count(ticker.upper())
        company_name = company_info.get('name', '')
        
        if ticker_count < 3:
            print(f"⚠️  WARNING: Analysis for {ticker} only mentions ticker {ticker_count} times!")
            print(f"   First 200 chars: {analysis[:200]}")
        
        # Add header to ensure ticker is clear
        header = f"=== FUNDAMENTAL ANALYSIS FOR {ticker} ({company_name}) ===\n\n"
        
        return header + analysis
    
    def _create_comprehensive_prompt(self) -> PromptTemplate:
        """Create prompt for comprehensive fundamental analysis with STRONG ticker anchoring"""
        
        template = """You are a fundamental analysis expert providing institutional-quality investment research.

**CRITICAL: You are analyzing {ticker} ({company_name}). You must ONLY analyze THIS stock. Do NOT confuse it with any other company.**

=== COMPANY INFORMATION ===
**ANALYZING: {ticker} - {company_name}**
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA FOR {ticker} ===
{fundamental_data}

=== RAW METRICS (for reference) ===
{raw_data}

=== YOUR ANALYSIS FRAMEWORK FOR {ticker} ===

**REMINDER: You are analyzing {ticker} ({company_name}). Every statement must be about THIS company.**

Based on {ticker}'s sector ({sector}), industry ({industry}), and size, select and weight the MOST RELEVANT metrics:

1. **SECTOR-SPECIFIC CONSIDERATIONS FOR {ticker}**:
   - Technology: Focus on growth rates, R&D spending, scalability metrics, recurring revenue
   - Financials: Focus on ROE, net interest margin, loan quality, capital ratios
   - Healthcare: Focus on pipeline value, patent cliffs, R&D productivity, regulatory risks
   - Consumer: Focus on same-store sales, brand value, inventory turnover, consumer trends
   - Industrials: Focus on capacity utilization, order backlog, cyclicality, capital intensity
   - Utilities: Focus on dividend safety, regulated returns, rate base growth, leverage
   - Energy: Focus on reserve life, production costs, commodity exposure, capital discipline
   - Real Estate: Focus on FFO, occupancy rates, cap rates, debt maturities

2. **SIZE-SPECIFIC CONSIDERATIONS FOR {ticker}**:
   - Large-cap (>$200B): Emphasize stability, dividend sustainability, market position
   - Mid-cap ($10B-$200B): Balance growth potential with financial stability
   - Small-cap (<$10B): Focus on growth trajectory, liquidity, management quality

3. **BUSINESS MODEL CONSIDERATIONS FOR {ticker}**:
   - Capital-intensive: Focus on ROIC, asset turnover, depreciation policies
   - Asset-light: Focus on margins, scalability, customer acquisition costs
   - Subscription/recurring: Focus on retention rates, LTV/CAC, expansion revenue
   - Cyclical: Focus on leverage, cash reserves, cycle positioning

=== REQUIRED ANALYSIS SECTIONS FOR {ticker} ===

**START YOUR ANALYSIS WITH: "{ticker} ({company_name})..."**

**1. EXECUTIVE SUMMARY FOR {ticker}** (2-3 sentences)
- Begin: "{ticker} presents..."
- Key investment thesis in plain language

**2. VALUATION ASSESSMENT OF {ticker}**
- Is {ticker} fairly valued based on:
  * Historical multiples comparison for {ticker}
  * Peer group comparison
  * Growth-adjusted metrics (PEG, EV/EBITDA vs growth)
- Provide specific numbers and context for {ticker}

**3. FINANCIAL QUALITY ANALYSIS OF {ticker}**
- Earnings quality (cash flow vs. accruals) for {ticker}
- Balance sheet strength (Altman Z-Score interpretation) for {ticker}
- Operational efficiency (Piotroski F-Score interpretation) for {ticker}
- DuPont ROE decomposition insights for {ticker}

**4. GROWTH SUSTAINABILITY OF {ticker}**
- Revenue growth trajectory and drivers for {ticker}
- Margin expansion/compression trends for {ticker}
- Reinvestment rate and returns on reinvested capital for {ticker}

**5. FINANCIAL HEALTH CHECK OF {ticker}**
- Liquidity adequacy for {ticker}'s business model
- Leverage appropriateness for {ticker}'s industry
- Interest coverage and debt serviceability for {ticker}

**6. DIVIDEND ANALYSIS OF {ticker}** (if applicable)
- Payout sustainability for {ticker}
- Coverage from earnings vs. free cash flow for {ticker}
- Growth potential or risks for {ticker}

**7. PEER POSITIONING OF {ticker}**
- How does {ticker} compare to industry peers?
- {ticker}'s competitive advantages or disadvantages

**8. KEY RISKS FOR {ticker}**
- 2-3 specific risks based on {ticker}'s fundamental data

**9. CONCLUSION FOR {ticker}**
- Synthesize the analysis into actionable insights about {ticker}
- Note any areas requiring further investigation for {ticker}

=== IMPORTANT GUIDELINES ===
- Be specific with numbers - cite actual figures from {ticker}'s data
- Explain what the numbers mean in context of {ticker} specifically
- Avoid generic statements - make every observation specific to {ticker}
- DO NOT provide buy/sell recommendations - focus on objective analysis of {ticker}
- Highlight both {ticker}'s strengths and weaknesses
- Keep the analysis to approximately 600-800 words
- **CRITICAL: Your entire response must be about {ticker} ({company_name}). Do NOT analyze any other stock.**

Begin your analysis of {ticker}:"""

        return PromptTemplate.from_template(template)
    
    def _create_synthesis_prompt(self) -> PromptTemplate:
        """Create prompt for structured output (for lead agent integration)"""
        
        template = """You are a fundamental analyst providing STRUCTURED analysis for algorithmic integration.

**ANALYZING: {ticker} ({company_name})**

=== DATA FOR {ticker} ===
{fundamental_data}

=== YOUR TASK ===

Analyze {ticker} and provide a STRUCTURED JSON response (not prose).

Return ONLY valid JSON in this exact format:

{{
  "ticker": "{ticker}",
  "company_name": "{company_name}",
  "llm_analysis": {{
    "valuation_verdict": {{
      "signal": "Undervalued|Fairly Valued|Overvalued|Extremely Overvalued",
      "confidence": "High|Medium|Low",
      "reasoning": "One sentence explaining {ticker}'s valuation"
    }},
    "quality_verdict": {{
      "signal": "High Quality|Medium Quality|Low Quality",
      "confidence": "High|Medium|Low",
      "altman_z": "Safe Zone|Grey Zone|Distress Zone",
      "piotroski_f": "Strong|Neutral|Weak",
      "reasoning": "One sentence about {ticker}'s quality"
    }},
    "growth_verdict": {{
      "signal": "High Growth|Moderate Growth|Stable|Declining",
      "confidence": "High|Medium|Low",
      "revenue_trajectory": "Accelerating|Stable|Decelerating",
      "reasoning": "One sentence about {ticker}'s growth"
    }},
    "financial_health": {{
      "signal": "Strong|Adequate|Weak",
      "leverage_status": "Low|Moderate|High",
      "liquidity_status": "Strong|Adequate|Weak"
    }},
    "key_strengths": ["strength 1 of {ticker}", "strength 2 of {ticker}", "strength 3 of {ticker}"],
    "key_risks": ["risk 1 for {ticker}", "risk 2 for {ticker}", "risk 3 for {ticker}"],
    "investment_thesis": {{
      "bull_case": "One sentence bull case for {ticker}",
      "bear_case": "One sentence bear case for {ticker}"
    }}
  }}
}}

**CRITICAL: All analysis must be about {ticker} ({company_name}). Return ONLY valid JSON.**"""

        return PromptTemplate.from_template(template)
    
    def _create_valuation_prompt(self) -> PromptTemplate:
        """Create prompt focused on valuation analysis"""
        
        template = """You are a valuation expert analyzing whether {ticker} is fairly priced.

**ANALYZING: {ticker} ({company_name})**

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA FOR {ticker} ===
{fundamental_data}

=== RAW METRICS ===
{raw_data}

=== VALUATION ANALYSIS FRAMEWORK FOR {ticker} ===

**You are analyzing {ticker} ONLY. Do not discuss any other stock.**

Analyze {ticker}'s valuation using multiple approaches appropriate for its sector:

1. **RELATIVE VALUATION OF {ticker}**
   Select the most relevant metrics for this {sector} company:
   - P/E analysis for {ticker}: Current vs forward, vs peers, vs historical
   - P/B analysis for {ticker}: Appropriate for asset-heavy businesses
   - P/S analysis for {ticker}: Useful for growth companies or when earnings volatile
   - EV/EBITDA for {ticker}: Best for comparing companies with different capital structures
   - PEG ratio for {ticker}: Growth-adjusted P/E interpretation

2. **YIELD-BASED VALUATION OF {ticker}**
   - Earnings yield vs bond yields for {ticker}
   - FCF yield analysis for {ticker}
   - Dividend yield vs historical and sector for {ticker}

3. **ASSET-BASED CONSIDERATIONS FOR {ticker}**
   - Book value per share analysis for {ticker}
   - Tangible book value if relevant for {ticker}

4. **PEER COMPARISON FOR {ticker}**
   - How do {ticker}'s multiples compare to direct competitors?
   - Premium/discount justification for {ticker}

5. **HISTORICAL CONTEXT FOR {ticker}**
   - {ticker}'s current multiples vs 5-year averages
   - {ticker}'s position in valuation range

6. **VALUATION CONCLUSION FOR {ticker}**
   - Synthesize findings about {ticker}
   - Note what the valuation implies about market expectations for {ticker}
   - Identify if {ticker}'s valuation is justified by fundamentals

DO NOT recommend buy/sell. Provide objective valuation assessment of {ticker} only.
Keep analysis to 400-500 words about {ticker}.

Begin your valuation analysis of {ticker}:"""

        return PromptTemplate.from_template(template)
    
    def _create_quality_prompt(self) -> PromptTemplate:
        """Create prompt focused on financial quality analysis"""
        
        template = """You are a financial quality analyst assessing earnings quality and financial health.

**ANALYZING: {ticker} ({company_name})**

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA FOR {ticker} ===
{fundamental_data}

=== RAW METRICS ===
{raw_data}

=== QUALITY ANALYSIS FRAMEWORK FOR {ticker} ===

**You must analyze {ticker} ONLY. Every statement must be about {ticker}.**

Assess {ticker}'s financial quality using these frameworks:

1. **EARNINGS QUALITY ASSESSMENT OF {ticker}**
   - Quality of Earnings ratio (CFO/Net Income) for {ticker}: Is cash backing earnings?
   - Accruals ratio for {ticker}: High accruals signal potential manipulation
   - Revenue recognition patterns for {ticker} (if observable)
   - Operating leverage effects for {ticker}

2. **ALTMAN Z-SCORE ANALYSIS OF {ticker}**
   - Interpret {ticker}'s Z-Score in context
   - Identify which components are strongest/weakest for {ticker}
   - Appropriate for {ticker}'s industry?

3. **PIOTROSKI F-SCORE BREAKDOWN OF {ticker}**
   - Analyze each of the 9 signals for {ticker} if possible
   - What does {ticker}'s score tell us about financial trajectory?

4. **DUPONT ANALYSIS OF {ticker}**
   - Decompose {ticker}'s ROE into three drivers
   - Which driver is most responsible for {ticker}'s returns?
   - Is {ticker}'s ROE sustainable or engineered through leverage?

5. **BALANCE SHEET QUALITY OF {ticker}**
   - Asset quality considerations for {ticker}
   - Liability structure appropriateness for {ticker}
   - Off-balance sheet concerns for {ticker} (if any indicators)

6. **CASH FLOW QUALITY OF {ticker}**
   - Operating cash flow trends for {ticker}
   - CapEx requirements for {ticker}
   - Free cash flow sustainability for {ticker}

7. **QUALITY CONCLUSION FOR {ticker}**
   - Overall financial quality rating for {ticker}
   - Key areas of strength for {ticker}
   - Key areas of concern for {ticker}
   - Red flags to monitor for {ticker}

Focus on objective quality assessment of {ticker}. No buy/sell recommendations.
Keep analysis to 400-500 words about {ticker}.

Begin your quality analysis of {ticker}:"""

        return PromptTemplate.from_template(template)
    
    def _create_dividend_prompt(self) -> PromptTemplate:
        """Create prompt focused on dividend sustainability analysis"""
        
        template = """You are a dividend analyst assessing dividend safety and sustainability.

**ANALYZING: {ticker} ({company_name})**

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA FOR {ticker} ===
{fundamental_data}

=== RAW METRICS ===
{raw_data}

=== DIVIDEND ANALYSIS FRAMEWORK FOR {ticker} ===

**You are analyzing {ticker}'s dividend only. Do not discuss other stocks.**

Analyze {ticker}'s dividend sustainability using these criteria:

1. **CURRENT DIVIDEND PROFILE OF {ticker}**
   - {ticker}'s current yield vs sector average
   - {ticker}'s current yield vs historical average
   - {ticker}'s yield attractiveness in current rate environment

2. **PAYOUT SUSTAINABILITY OF {ticker}**
   - {ticker}'s earnings payout ratio: Is it sustainable (<60% generally safe)?
   - {ticker}'s FCF payout ratio: Cash-based coverage
   - Coverage ratios for {ticker}: How many times can dividend be paid?

3. **EARNINGS STABILITY OF {ticker}**
   - {ticker}'s earnings volatility (affects dividend safety)
   - {ticker}'s revenue stability
   - {ticker}'s margin trends

4. **BALANCE SHEET SUPPORT FOR {ticker}**
   - {ticker}'s cash reserves relative to dividend obligation
   - {ticker}'s debt levels that could pressure dividend
   - {ticker}'s liquidity adequacy

5. **GROWTH POTENTIAL FOR {ticker}**
   - Room for dividend growth based on {ticker}'s payout ratio
   - {ticker}'s historical dividend growth (if available)
   - {ticker}'s earnings growth to support future increases

6. **SECTOR CONSIDERATIONS FOR {ticker}**
   - How does {ticker}'s dividend policy compare to {sector} norms?
   - Industry-specific dividend expectations for {ticker}

7. **RISK FACTORS FOR {ticker}**
   - What could threaten {ticker}'s dividend?
   - {ticker}'s economic sensitivity
   - {ticker}'s capital allocation priorities

8. **DIVIDEND SUSTAINABILITY SCORE FOR {ticker}**
   - Overall assessment of {ticker}'s dividend safety
   - Likelihood of cuts vs. growth for {ticker}
   - Income investor suitability for {ticker}

For non-dividend payers like {ticker}, discuss:
- Why does {ticker} not pay dividends?
- Likelihood of {ticker} initiating dividends
- {ticker}'s capital allocation priorities

Keep analysis to 400-500 words about {ticker}.

Begin your dividend analysis of {ticker}:"""

        return PromptTemplate.from_template(template)
    
    def _create_growth_prompt(self) -> PromptTemplate:
        """Create prompt focused on growth analysis"""
        
        template = """You are a growth analyst evaluating a company's growth profile and sustainability.

**ANALYZING: {ticker} ({company_name})**

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA FOR {ticker} ===
{fundamental_data}

=== RAW METRICS ===
{raw_data}

=== GROWTH ANALYSIS FRAMEWORK FOR {ticker} ===

**You must analyze {ticker} ONLY. All statements must be about {ticker}.**

Analyze {ticker}'s growth prospects using these dimensions:

1. **HISTORICAL GROWTH REVIEW OF {ticker}**
   - {ticker}'s revenue growth: YoY and multi-year CAGR
   - {ticker}'s earnings growth: Quality and consistency
   - {ticker}'s EBITDA growth: Operating leverage
   - {ticker}'s book value growth: Wealth creation

2. **GROWTH QUALITY ASSESSMENT OF {ticker}**
   - Is {ticker}'s growth organic vs. inorganic (M&A-driven)?
   - Is margin expansion contributing to {ticker}'s earnings growth?
   - Are one-time items inflating/deflating {ticker}'s growth?

3. **GROWTH DRIVERS OF {ticker}**
   - What's driving {ticker}'s growth (based on sector)?
   - Is {ticker} gaining market share vs. market expansion?
   - Does {ticker} have pricing power vs. volume growth?
   - Does {ticker} have new products/services?

4. **REINVESTMENT ANALYSIS OF {ticker}**
   - How much is {ticker} reinvesting for growth?
   - {ticker}'s returns on reinvested capital (ROIC vs WACC implied)
   - {ticker}'s CapEx intensity and trends

5. **MARGIN TRAJECTORY OF {ticker}**
   - {ticker}'s gross margin trends (pricing power, cost control)
   - {ticker}'s operating margin trends (operating leverage)
   - {ticker}'s net margin trends (financial efficiency)

6. **GROWTH SUSTAINABILITY OF {ticker}**
   - Can {ticker} maintain current growth rates?
   - {ticker}'s market opportunity remaining
   - {ticker}'s competitive position strength

7. **GROWTH VS. VALUATION OF {ticker}**
   - Is {ticker}'s growth appropriately reflected in valuation?
   - {ticker}'s PEG ratio interpretation
   - Growth-adjusted metrics for {ticker}

8. **GROWTH RISKS FOR {ticker}**
   - What could derail {ticker}'s growth story?
   - {ticker}'s dependency risks
   - Competition and disruption facing {ticker}

9. **GROWTH OUTLOOK FOR {ticker}**
   - {ticker}'s near-term vs long-term growth potential
   - Catalyst identification for {ticker}
   - Inflection points for {ticker}

Focus on objective growth assessment of {ticker}. No buy/sell recommendations.
Keep analysis to 400-500 words about {ticker}.

Begin your growth analysis of {ticker}:"""

        return PromptTemplate.from_template(template)
    
    def compare_stocks(self, tickers: List[str]) -> str:
        """
        Compare multiple stocks for investment analysis.
        
        Args:
            tickers: List of stock ticker symbols to compare
        
        Returns:
            Comparative analysis report
        """
        if len(tickers) < 2:
            return "Please provide at least 2 tickers for comparison."
        
        if len(tickers) > 5:
            tickers = tickers[:5]  # Limit to 5 stocks
        
        # Fetch data for all tickers
        all_data = {}
        all_formatted = {}
        for ticker in tickers:
            ticker = ticker.upper()
            all_data[ticker] = self.calculator.get_all_fundamentals(ticker)
            all_formatted[ticker] = self.calculator.get_formatted_for_llm(ticker)
        
        # Create comparison prompt
        prompt = self._create_comparison_prompt()
        
        # Combine formatted data
        combined_data = "\n\n" + "="*80 + "\n\n".join([
            f"STOCK: {ticker}\n{data}" 
            for ticker, data in all_formatted.items()
        ])
        
        # Extract key metrics for comparison table
        comparison_table = self._build_comparison_table(all_data)
        
        response = self.llm.invoke(prompt.format(
            tickers=", ".join(tickers),
            num_stocks=len(tickers),
            comparison_table=comparison_table,
            detailed_data=combined_data
        ))
        
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            text_parts = [item for item in response.content if isinstance(item, str)]
            return '\n'.join(text_parts) if text_parts else str(response.content)
        else:
            return str(response.content)
    
    def _build_comparison_table(self, all_data: Dict) -> str:
        """Build a comparison table of key metrics"""
        
        metrics = [
            ('Market Cap', lambda d: f"${d.get('company_info', {}).get('market_cap', 0)/1e9:.1f}B"),
            ('P/E Ratio', lambda d: f"{d.get('valuation_ratios', {}).get('pe_ratio', 0):.1f}"),
            ('PEG Ratio', lambda d: f"{d.get('valuation_ratios', {}).get('peg_ratio', 0):.2f}"),
            ('P/B Ratio', lambda d: f"{d.get('valuation_ratios', {}).get('price_to_book', 0):.2f}"),
            ('EV/EBITDA', lambda d: f"{d.get('valuation_ratios', {}).get('ev_to_ebitda', 0):.1f}"),
            ('Gross Margin', lambda d: f"{d.get('profitability_ratios', {}).get('gross_margin', 0):.1f}%"),
            ('Net Margin', lambda d: f"{d.get('profitability_ratios', {}).get('net_profit_margin', 0):.1f}%"),
            ('ROE', lambda d: f"{d.get('profitability_ratios', {}).get('return_on_equity', 0):.1f}%"),
            ('ROIC', lambda d: f"{d.get('profitability_ratios', {}).get('return_on_invested_capital', 0):.1f}%"),
            ('Revenue Growth', lambda d: f"{d.get('growth_metrics', {}).get('revenue_growth_yoy', 0):.1f}%"),
            ('Earnings Growth', lambda d: f"{d.get('growth_metrics', {}).get('earnings_growth_yoy', 0):.1f}%"),
            ('Debt/Equity', lambda d: f"{d.get('leverage_ratios', {}).get('debt_to_equity', 0):.2f}"),
            ('Current Ratio', lambda d: f"{d.get('liquidity_ratios', {}).get('current_ratio', 0):.2f}"),
            ('Dividend Yield', lambda d: f"{d.get('dividend_metrics', {}).get('dividend_yield', 0):.2f}%"),
            ('Payout Ratio', lambda d: f"{d.get('dividend_metrics', {}).get('payout_ratio', 0):.1f}%"),
            ('Altman Z-Score', lambda d: f"{d.get('quality_scores', {}).get('altman_z_score', 0):.2f}"),
            ('Piotroski F-Score', lambda d: f"{d.get('quality_scores', {}).get('piotroski_f_score', 0)}/9"),
        ]
        
        tickers = list(all_data.keys())
        
        # Build table header
        header = f"{'Metric':<20}" + "".join([f"{t:>15}" for t in tickers])
        separator = "-" * len(header)
        
        rows = [header, separator]
        
        for metric_name, metric_fn in metrics:
            row = f"{metric_name:<20}"
            for ticker in tickers:
                try:
                    value = metric_fn(all_data[ticker])
                except:
                    value = "N/A"
                row += f"{value:>15}"
            rows.append(row)
        
        return "\n".join(rows)
    
    def _create_comparison_prompt(self) -> PromptTemplate:
        """Create prompt for comparative analysis"""
        
        template = """You are an investment analyst comparing {num_stocks} stocks for investment suitability.

=== STOCKS BEING COMPARED ===
{tickers}

=== COMPARISON TABLE ===
{comparison_table}

=== DETAILED DATA FOR EACH STOCK ===
{detailed_data}

=== COMPARATIVE ANALYSIS FRAMEWORK ===

Provide a structured comparison covering:

1. **OVERVIEW**
   - Brief description of each company
   - Are these comparable companies (similar sector/size)?

2. **VALUATION COMPARISON**
   - Which appears most/least expensive on each metric?
   - Which valuation is justified by fundamentals?
   - Rank by overall valuation attractiveness

3. **QUALITY COMPARISON**
   - Compare earnings quality indicators
   - Compare balance sheet strength
   - Rank by financial quality

4. **GROWTH COMPARISON**
   - Compare growth rates and sustainability
   - Which has best growth trajectory?
   - Rank by growth profile

5. **RISK COMPARISON**
   - Compare leverage and liquidity
   - Compare earnings stability
   - Which has lowest risk profile?

6. **DIVIDEND COMPARISON** (if applicable)
   - Compare yields and sustainability
   - Which is best for income?

7. **COMPETITIVE POSITIONING**
   - Key advantages of each
   - Key disadvantages of each

8. **SUMMARY TABLE**
   Create a simple ranking table:
   | Category | Stock 1 | Stock 2 | ... |
   Rank each from 1 (best) to n (worst) for:
   - Valuation
   - Quality
   - Growth
   - Safety
   - Overall

9. **CONCLUSION**
   - Key differentiators
   - Which suits what investor profile?
   - Areas requiring further research

DO NOT make explicit buy/sell recommendations. Provide objective comparison only.
Keep analysis to 700-900 words.

Begin your comparative analysis:"""

        return PromptTemplate.from_template(template)
    
    def batch_analyze(self, tickers: List[str], analysis_type: str = "comprehensive") -> Dict[str, str]:
        """Analyze multiple tickers individually"""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.analyze(ticker, analysis_type)
            except Exception as e:
                results[ticker] = f"Error analyzing {ticker}: {str(e)}"
        return results
    
    def quick_screen(self, ticker: str) -> str:
        """
        Quick fundamental screening - summary view for rapid assessment.
        """
        ticker = ticker.upper()
        data = self.calculator.get_all_fundamentals(ticker)
        
        company_info = data.get('company_info', {})
        val = data.get('valuation_ratios', {})
        prof = data.get('profitability_ratios', {})
        liq = data.get('liquidity_ratios', {})
        lev = data.get('leverage_ratios', {})
        growth = data.get('growth_metrics', {})
        div = data.get('dividend_metrics', {})
        qual = data.get('quality_scores', {})
        
        # Generate quick assessment flags
        flags = []
        
        # Valuation flags
        pe = val.get('pe_ratio', 0)
        if pe > 0:
            if pe < 15:
                flags.append("✓ Low P/E (<15)")
            elif pe > 30:
                flags.append("⚠ High P/E (>30)")
        
        peg = val.get('peg_ratio', 0)
        if peg > 0:
            if peg < 1:
                flags.append("✓ PEG < 1 (potentially undervalued)")
            elif peg > 2:
                flags.append("⚠ PEG > 2 (growth may be overpriced)")
        
        # Profitability flags
        roe = prof.get('return_on_equity', 0)
        if roe > 15:
            flags.append("✓ Strong ROE (>15%)")
        elif roe < 8 and roe > 0:
            flags.append("⚠ Weak ROE (<8%)")
        
        # Leverage flags
        de = lev.get('debt_to_equity', 0)
        if de > 2:
            flags.append("⚠ High Leverage (D/E > 2)")
        elif de < 0.5:
            flags.append("✓ Low Leverage (D/E < 0.5)")
        
        # Liquidity flags
        cr = liq.get('current_ratio', 0)
        if cr < 1:
            flags.append("⚠ Low Liquidity (CR < 1)")
        elif cr > 2:
            flags.append("✓ Strong Liquidity (CR > 2)")
        
        # Quality flags
        z_score = qual.get('altman_z_score', 0)
        if z_score > 3:
            flags.append("✓ Altman Z > 3 (Safe Zone)")
        elif z_score < 1.8:
            flags.append("⚠ Altman Z < 1.8 (Distress Zone)")
        
        f_score = qual.get('piotroski_f_score', 0)
        if f_score >= 7:
            flags.append("✓ Piotroski F ≥ 7 (Strong)")
        elif f_score <= 3:
            flags.append("⚠ Piotroski F ≤ 3 (Weak)")
        
        # Growth flags
        rev_growth = growth.get('revenue_growth_yoy', 0)
        if rev_growth > 20:
            flags.append("✓ Strong Revenue Growth (>20%)")
        elif rev_growth < 0:
            flags.append("⚠ Declining Revenue")
        
        # Dividend flags
        div_yield = div.get('dividend_yield', 0)
        payout = div.get('payout_ratio', 0)
        if div_yield > 4:
            if payout > 80:
                flags.append("⚠ High Yield but High Payout")
            else:
                flags.append("✓ Attractive Dividend (>4%, sustainable)")
        
        screen_result = f"""
=== QUICK FUNDAMENTAL SCREEN: {ticker} ===

Company: {company_info.get('name', ticker)}
Sector: {company_info.get('sector', 'N/A')} | Industry: {company_info.get('industry', 'N/A')}
Market Cap: ${company_info.get('market_cap', 0)/1e9:.1f}B

KEY METRICS AT A GLANCE:
------------------------
Valuation:  P/E: {val.get('pe_ratio', 0):.1f} | PEG: {val.get('peg_ratio', 0):.2f} | P/B: {val.get('price_to_book', 0):.2f}
Profit:     ROE: {prof.get('return_on_equity', 0):.1f}% | Net Margin: {prof.get('net_profit_margin', 0):.1f}%
Growth:     Revenue: {growth.get('revenue_growth_yoy', 0):+.1f}% | Earnings: {growth.get('earnings_growth_yoy', 0):+.1f}%
Health:     D/E: {lev.get('debt_to_equity', 0):.2f} | Current Ratio: {liq.get('current_ratio', 0):.2f}
Quality:    Z-Score: {qual.get('altman_z_score', 0):.2f} | F-Score: {qual.get('piotroski_f_score', 0)}/9
Dividend:   Yield: {div.get('dividend_yield', 0):.2f}% | Payout: {div.get('payout_ratio', 0):.1f}%

SCREENING FLAGS:
----------------
{chr(10).join(flags) if flags else "No significant flags"}

QUICK ASSESSMENT:
- Altman Z-Score: {qual.get('altman_z_interpretation', 'N/A')}
- Piotroski F-Score: {qual.get('piotroski_f_interpretation', 'N/A')}
- Quality of Earnings: {qual.get('quality_of_earnings', 0):.2f}x (CFO/NI)
"""
        return screen_result


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_fundamentals(ticker: str, analysis_type: str = "comprehensive") -> str:
    """Simple function to run fundamental analysis"""
    agent = FundamentalAnalystAgent()
    return agent.analyze(ticker, analysis_type)


def compare_fundamentals(tickers: List[str]) -> str:
    """Simple function to compare stocks"""
    agent = FundamentalAnalystAgent()
    return agent.compare_stocks(tickers)


def quick_screen(ticker: str) -> str:
    """Simple function for quick screening"""
    agent = FundamentalAnalystAgent()
    return agent.quick_screen(ticker)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fundamental Analysis Agent')
    parser.add_argument('tickers', nargs='+', help='Stock ticker(s) to analyze')
    parser.add_argument('--type', '-t', 
                       choices=['comprehensive', 'valuation', 'quality', 'dividend', 'growth', 'quick'],
                       default='comprehensive',
                       help='Type of analysis')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare multiple stocks instead of individual analysis')
    args = parser.parse_args()
    
    agent = FundamentalAnalystAgent()
    
    print(f"\n{'='*70}")
    print(f"Fundamental Analysis Agent")
    print(f"{'='*70}\n")
    
    if args.type == 'quick':
        for ticker in args.tickers:
            print(agent.quick_screen(ticker))
            print("\n" + "="*70 + "\n")
    elif args.compare and len(args.tickers) > 1:
        print(f"Comparing: {', '.join(args.tickers)}\n")
        print(agent.compare_stocks(args.tickers))
    else:
        for ticker in args.tickers:
            print(f"Analyzing {ticker} ({args.type} analysis)...\n")
            print(agent.analyze(ticker, args.type))
            print("\n" + "="*70 + "\n")