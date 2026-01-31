"""
Fundamental Analyst Agent Module
=================================
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
    """
    
    def __init__(self, model: str = "claude-sonnet-4-5"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.calculator = FundamentalCalculator()
        self.llm = ChatAnthropic(model_name=model, temperature=0)
    
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
        else:
            prompt = self._create_comprehensive_prompt()
        
        # Get company info for context
        company_info = data.get('company_info', {})
        
        # Invoke LLM
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
            return response.content
        elif isinstance(response.content, list):
            text_parts = [item for item in response.content if isinstance(item, str)]
            return '\n'.join(text_parts) if text_parts else str(response.content)
        else:
            return str(response.content)
    
    def _create_comprehensive_prompt(self) -> PromptTemplate:
        """Create prompt for comprehensive fundamental analysis"""
        
        template = """You are a fundamental analysis expert providing institutional-quality investment research. 
Analyze the following stock using a systematic approach tailored to its specific characteristics.

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA ===
{fundamental_data}

=== RAW METRICS (for reference) ===
{raw_data}

=== YOUR ANALYSIS FRAMEWORK ===

Based on the company's sector, industry, and size, select and weight the MOST RELEVANT metrics:

1. **SECTOR-SPECIFIC CONSIDERATIONS**:
   - Technology: Focus on growth rates, R&D spending, scalability metrics, recurring revenue
   - Financials: Focus on ROE, net interest margin, loan quality, capital ratios
   - Healthcare: Focus on pipeline value, patent cliffs, R&D productivity, regulatory risks
   - Consumer: Focus on same-store sales, brand value, inventory turnover, consumer trends
   - Industrials: Focus on capacity utilization, order backlog, cyclicality, capital intensity
   - Utilities: Focus on dividend safety, regulated returns, rate base growth, leverage
   - Energy: Focus on reserve life, production costs, commodity exposure, capital discipline
   - Real Estate: Focus on FFO, occupancy rates, cap rates, debt maturities

2. **SIZE-SPECIFIC CONSIDERATIONS**:
   - Large-cap (>$200B): Emphasize stability, dividend sustainability, market position
   - Mid-cap ($10B-$200B): Balance growth potential with financial stability
   - Small-cap (<$10B): Focus on growth trajectory, liquidity, management quality

3. **BUSINESS MODEL CONSIDERATIONS**:
   - Capital-intensive: Focus on ROIC, asset turnover, depreciation policies
   - Asset-light: Focus on margins, scalability, customer acquisition costs
   - Subscription/recurring: Focus on retention rates, LTV/CAC, expansion revenue
   - Cyclical: Focus on leverage, cash reserves, cycle positioning

=== REQUIRED ANALYSIS SECTIONS ===

**1. EXECUTIVE SUMMARY** (2-3 sentences)
- Key investment thesis in plain language

**2. VALUATION ASSESSMENT**
- Is the stock fairly valued based on:
  * Historical multiples comparison
  * Peer group comparison
  * Growth-adjusted metrics (PEG, EV/EBITDA vs growth)
- Provide specific numbers and context

**3. FINANCIAL QUALITY ANALYSIS**
- Earnings quality (cash flow vs. accruals)
- Balance sheet strength (Altman Z-Score interpretation)
- Operational efficiency (Piotroski F-Score interpretation)
- DuPont ROE decomposition insights

**4. GROWTH SUSTAINABILITY**
- Revenue growth trajectory and drivers
- Margin expansion/compression trends
- Reinvestment rate and returns on reinvested capital

**5. FINANCIAL HEALTH CHECK**
- Liquidity adequacy for the business model
- Leverage appropriateness for the industry
- Interest coverage and debt serviceability

**6. DIVIDEND ANALYSIS** (if applicable)
- Payout sustainability
- Coverage from earnings vs. free cash flow
- Growth potential or risks

**7. PEER POSITIONING**
- How does this company compare to industry peers?
- Competitive advantages or disadvantages

**8. KEY RISKS**
- 2-3 specific risks based on the fundamental data

**9. CONCLUSION**
- Synthesize the analysis into actionable insights
- Note any areas requiring further investigation

=== IMPORTANT GUIDELINES ===
- Be specific with numbers - cite actual figures from the data
- Explain what the numbers mean in context of this specific company
- Avoid generic statements - make every observation company-specific
- DO NOT provide buy/sell recommendations - focus on objective analysis
- Highlight both strengths and weaknesses
- Keep the analysis to approximately 600-800 words

Begin your analysis:"""

        return PromptTemplate.from_template(template)
    
    def _create_valuation_prompt(self) -> PromptTemplate:
        """Create prompt focused on valuation analysis"""
        
        template = """You are a valuation expert analyzing whether a stock is fairly priced.

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA ===
{fundamental_data}

=== RAW METRICS ===
{raw_data}

=== VALUATION ANALYSIS FRAMEWORK ===

Analyze the stock's valuation using multiple approaches appropriate for its sector:

1. **RELATIVE VALUATION**
   Select the most relevant metrics for this {sector} company:
   - P/E analysis: Current vs forward, vs peers, vs historical
   - P/B analysis: Appropriate for asset-heavy businesses
   - P/S analysis: Useful for growth companies or when earnings volatile
   - EV/EBITDA: Best for comparing companies with different capital structures
   - PEG ratio: Growth-adjusted P/E interpretation

2. **YIELD-BASED VALUATION**
   - Earnings yield vs bond yields
   - FCF yield analysis
   - Dividend yield vs historical and sector

3. **ASSET-BASED CONSIDERATIONS**
   - Book value per share analysis
   - Tangible book value if relevant

4. **PEER COMPARISON**
   - How do multiples compare to direct competitors?
   - Premium/discount justification

5. **HISTORICAL CONTEXT**
   - Current multiples vs 5-year averages
   - Position in valuation range

6. **VALUATION CONCLUSION**
   - Synthesize findings
   - Note what the valuation implies about market expectations
   - Identify if valuation is justified by fundamentals

DO NOT recommend buy/sell. Provide objective valuation assessment only.
Keep analysis to 400-500 words.

Begin your valuation analysis:"""

        return PromptTemplate.from_template(template)
    
    def _create_quality_prompt(self) -> PromptTemplate:
        """Create prompt focused on financial quality analysis"""
        
        template = """You are a financial quality analyst assessing earnings quality and financial health.

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA ===
{fundamental_data}

=== RAW METRICS ===
{raw_data}

=== QUALITY ANALYSIS FRAMEWORK ===

Assess the company's financial quality using these frameworks:

1. **EARNINGS QUALITY ASSESSMENT**
   - Quality of Earnings ratio (CFO/Net Income): Is cash backing earnings?
   - Accruals ratio: High accruals signal potential manipulation
   - Revenue recognition patterns (if observable)
   - Operating leverage effects

2. **ALTMAN Z-SCORE ANALYSIS**
   - Interpret the Z-Score in context
   - Identify which components are strongest/weakest
   - Appropriate for {industry}?

3. **PIOTROSKI F-SCORE BREAKDOWN**
   - Analyze each of the 9 signals if possible
   - What does the score tell us about financial trajectory?

4. **DUPONT ANALYSIS**
   - Decompose ROE into three drivers
   - Which driver is most responsible for returns?
   - Is the ROE sustainable or engineered through leverage?

5. **BALANCE SHEET QUALITY**
   - Asset quality considerations
   - Liability structure appropriateness
   - Off-balance sheet concerns (if any indicators)

6. **CASH FLOW QUALITY**
   - Operating cash flow trends
   - CapEx requirements
   - Free cash flow sustainability

7. **QUALITY CONCLUSION**
   - Overall financial quality rating
   - Key areas of strength
   - Key areas of concern
   - Red flags to monitor

Focus on objective quality assessment. No buy/sell recommendations.
Keep analysis to 400-500 words.

Begin your quality analysis:"""

        return PromptTemplate.from_template(template)
    
    def _create_dividend_prompt(self) -> PromptTemplate:
        """Create prompt focused on dividend sustainability analysis"""
        
        template = """You are a dividend analyst assessing dividend safety and sustainability.

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA ===
{fundamental_data}

=== RAW METRICS ===
{raw_data}

=== DIVIDEND ANALYSIS FRAMEWORK ===

Analyze dividend sustainability using these criteria:

1. **CURRENT DIVIDEND PROFILE**
   - Current yield vs sector average
   - Current yield vs historical average
   - Yield attractiveness in current rate environment

2. **PAYOUT SUSTAINABILITY**
   - Earnings payout ratio: Is it sustainable (<60% generally safe)?
   - FCF payout ratio: Cash-based coverage
   - Coverage ratios: How many times can dividend be paid?

3. **EARNINGS STABILITY**
   - Earnings volatility (affects dividend safety)
   - Revenue stability
   - Margin trends

4. **BALANCE SHEET SUPPORT**
   - Cash reserves relative to dividend obligation
   - Debt levels that could pressure dividend
   - Liquidity adequacy

5. **GROWTH POTENTIAL**
   - Room for dividend growth based on payout ratio
   - Historical dividend growth (if available)
   - Earnings growth to support future increases

6. **SECTOR CONSIDERATIONS**
   - How does dividend policy compare to {sector} norms?
   - Industry-specific dividend expectations

7. **RISK FACTORS**
   - What could threaten the dividend?
   - Economic sensitivity
   - Capital allocation priorities

8. **DIVIDEND SUSTAINABILITY SCORE**
   - Overall assessment of dividend safety
   - Likelihood of cuts vs. growth
   - Income investor suitability

For non-dividend payers, discuss:
- Why no dividend?
- Likelihood of initiation
- Capital allocation priorities

Keep analysis to 400-500 words.

Begin your dividend analysis:"""

        return PromptTemplate.from_template(template)
    
    def _create_growth_prompt(self) -> PromptTemplate:
        """Create prompt focused on growth analysis"""
        
        template = """You are a growth analyst evaluating a company's growth profile and sustainability.

=== COMPANY INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap:,}

=== FUNDAMENTAL DATA ===
{fundamental_data}

=== RAW METRICS ===
{raw_data}

=== GROWTH ANALYSIS FRAMEWORK ===

Analyze growth prospects using these dimensions:

1. **HISTORICAL GROWTH REVIEW**
   - Revenue growth: YoY and multi-year CAGR
   - Earnings growth: Quality and consistency
   - EBITDA growth: Operating leverage
   - Book value growth: Wealth creation

2. **GROWTH QUALITY ASSESSMENT**
   - Organic vs. inorganic (M&A-driven)?
   - Margin expansion contributing to earnings growth?
   - One-time items inflating/deflating growth?

3. **GROWTH DRIVERS**
   - What's driving the growth (based on sector)?
   - Market share gains vs. market expansion?
   - Pricing power vs. volume growth?
   - New products/services?

4. **REINVESTMENT ANALYSIS**
   - How much is being reinvested for growth?
   - Returns on reinvested capital (ROIC vs WACC implied)
   - CapEx intensity and trends

5. **MARGIN TRAJECTORY**
   - Gross margin trends (pricing power, cost control)
   - Operating margin trends (operating leverage)
   - Net margin trends (financial efficiency)

6. **GROWTH SUSTAINABILITY**
   - Can current growth rates be maintained?
   - Market opportunity remaining
   - Competitive position strength

7. **GROWTH VS. VALUATION**
   - Is growth appropriately reflected in valuation?
   - PEG ratio interpretation
   - Growth-adjusted metrics

8. **GROWTH RISKS**
   - What could derail the growth story?
   - Dependency risks
   - Competition and disruption

9. **GROWTH OUTLOOK**
   - Near-term vs long-term growth potential
   - Catalyst identification
   - Inflection points

Focus on objective growth assessment. No buy/sell recommendations.
Keep analysis to 400-500 words.

Begin your growth analysis:"""

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