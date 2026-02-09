"""
Sentiment Analyst Agent Module
================================
Uses LLM to intelligently interpret news and sentiment data for
investment analysis. Completes the three-agent pipeline alongside
technical_agent and fundamental_agent.

Design:
- Lightweight single-mode agent (for_synthesis only, no detailed/comprehensive)
- Reads raw headlines from SentimentDataFetcher and produces structured signals
- Works even without Alpha Vantage pre-computed scores (Yahoo Finance always available)
- Handles deduplication and event clustering via LLM interpretation
- Produces uniform structured JSON for lead agent consumption

Usage:
    from sentiment_agent import SentimentAnalystAgent
    
    agent = SentimentAnalystAgent()
    signals = agent.analyze("AAPL")  # Returns structured JSON string
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic

from sentiment import SentimentDataFetcher
from technical import TechnicalCalculator  # For stock_info lookup

load_dotenv()


class SentimentAnalystAgent:
    """
    LLM-powered sentiment analysis agent.
    
    Single-mode design (for_synthesis only):
    - Sentiment is inherently noisier than technical/fundamental data
    - A detailed prose mode adds cost with marginal value at 20% weight
    - The LLM's main job is interpreting headlines, not writing reports
    
    Produces structured JSON with:
    - overall_tone: Bullish/Bearish/Neutral/Mixed
    - tone_direction: Improving/Stable/Deteriorating
    - key_events: Deduplicated significant events
    - key_themes: Recurring narrative threads
    - controversy_flag: Whether negative controversy exists
    - catalyst_detected: Upcoming event that could move price
    - news_concentration: Whether coverage is one-event or diverse
    - risk_signals: Sentiment-derived risk flags
    """
    
    def __init__(self, model: str = "claude-sonnet-4-5"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.fetcher = SentimentDataFetcher()
        self.llm = ChatAnthropic(model_name=model, temperature=0, max_tokens=2048)
        
        # For getting company name/industry
        self._tech_calc = TechnicalCalculator()
    
    def analyze(self, ticker: str, company_name: str = None, 
                industry: str = None) -> str:
        """
        Analyze sentiment for a ticker and return structured JSON string.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Optional, fetched automatically if not provided
            industry: Optional, fetched automatically if not provided
        
        Returns:
            String containing structured JSON signals (or raw text if JSON parse fails)
        """
        ticker = ticker.upper()
        
        # Get company info if not provided
        if not company_name or not industry:
            try:
                info = self._tech_calc.get_stock_info(ticker)
                company_name = company_name or info.get('name', ticker)
                industry = industry or info.get('industry', 'Unknown')
            except Exception:
                company_name = company_name or ticker
                industry = industry or 'Unknown'
        
        # Fetch raw sentiment data
        raw_data = self.fetcher.fetch_all(
            ticker=ticker,
            company_name=company_name,
            industry=industry,
            max_company_news=25,
            max_industry_news=10
        )
        
        # Format headlines for LLM consumption
        headlines_text = self._format_headlines_for_llm(raw_data, ticker)
        
        # Get pre-computed sentiment stats if available
        sentiment_stats = self._extract_sentiment_stats(raw_data)
        
        # Build and invoke prompt
        prompt = self._build_prompt(
            ticker=ticker,
            company_name=company_name,
            industry=industry,
            headlines=headlines_text,
            sentiment_stats=sentiment_stats,
            num_articles=raw_data.get('summary', {}).get('total_items', 0),
            sources_used=raw_data.get('summary', {}).get('sources_used', [])
        )
        
        response = self.llm.invoke(prompt)
        
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            text_parts = [item for item in response.content if isinstance(item, str)]
            return '\n'.join(text_parts) if text_parts else str(response.content)
        else:
            return str(response.content)
    
    def _format_headlines_for_llm(self, raw_data: Dict, ticker: str) -> str:
        """
        Format raw news data into a compact headline list for LLM processing.
        Includes source, date, and pre-computed sentiment hint if available.
        """
        sections = []
        
        # Company news
        company_news = raw_data.get('company_news', {})
        articles = company_news.get('articles', [])
        
        if articles:
            sections.append(f"COMPANY NEWS ({len(articles)} articles):")
            for i, article in enumerate(articles[:20], 1):
                date_str = ""
                try:
                    pub = article.published_at
                    if pub.tzinfo:
                        pub = pub.replace(tzinfo=None)
                    days_ago = (datetime.now() - pub).days
                    date_str = f"{days_ago}d ago" if days_ago > 0 else "today"
                except Exception:
                    date_str = "recent"
                
                sentiment_str = ""
                if article.sentiment_hint is not None:
                    sentiment_str = f" [score: {article.sentiment_hint:.2f}]"
                
                sections.append(
                    f"  [{i}] {article.title}"
                    f"\n      Source: {article.source} | {date_str}{sentiment_str}"
                )
        else:
            sections.append("COMPANY NEWS: None found")
        
        # Industry news
        industry_news = raw_data.get('industry_news', {})
        industry_articles = industry_news.get('articles', []) if industry_news else []
        
        if industry_articles:
            sections.append(f"\nINDUSTRY NEWS ({len(industry_articles)} articles):")
            for i, article in enumerate(industry_articles[:8], 1):
                sections.append(f"  [{i}] {article.title}")
        
        return "\n".join(sections)
    
    def _extract_sentiment_stats(self, raw_data: Dict) -> str:
        """
        Extract pre-computed sentiment statistics if available (from Alpha Vantage).
        Returns a summary string or 'None available'.
        """
        articles = raw_data.get('company_news', {}).get('articles', [])
        
        sentiment_values = [
            a.sentiment_hint for a in articles 
            if a.sentiment_hint is not None
        ]
        
        if not sentiment_values:
            return "No pre-computed sentiment scores available. Analyze tone from headlines only."
        
        avg = sum(sentiment_values) / len(sentiment_values)
        positive = sum(1 for v in sentiment_values if v > 0.1)
        negative = sum(1 for v in sentiment_values if v < -0.1)
        neutral = len(sentiment_values) - positive - negative
        
        return (
            f"Pre-computed scores available for {len(sentiment_values)} articles:\n"
            f"  Average score: {avg:.3f} (scale: -1 to +1)\n"
            f"  Distribution: {positive} positive, {neutral} neutral, {negative} negative"
        )
    
    def _build_prompt(self, *, ticker, company_name, industry, headlines,
                       sentiment_stats, num_articles, sources_used) -> str:
        """Build the sentiment analysis prompt as a plain string."""
        
        return f"""You are a sentiment analyst providing STRUCTURED analysis for algorithmic integration.

**ANALYZING: {ticker} ({company_name})**
Industry: {industry}
Data Sources: {', '.join(sources_used) if sources_used else 'yahoo_finance'}
Total Articles: {num_articles}

=== PRE-COMPUTED SENTIMENT STATISTICS ===
{sentiment_stats}

=== RAW NEWS DATA ===
{headlines}

=== YOUR TASK ===

Analyze the news coverage for {ticker} ({company_name}) and provide structured sentiment signals.

IMPORTANT INSTRUCTIONS:
1. DEDUPLICATE: Multiple headlines about the same event count as ONE event, not separate signals. Cluster them.
2. DISTINGUISH: Separate {ticker}-specific news from general industry/market noise.
3. WEIGHT BY RECENCY: News from today/yesterday matters more than week-old news.
4. ASSESS MATERIALITY: A CEO departure matters more than a minor product update.
5. If headlines are scarce or generic, say so honestly — don't fabricate a narrative.

Return ONLY valid JSON in this exact format:

{{
  "ticker": "{ticker}",
  "company_name": "{company_name}",
  "llm_analysis": {{
    "overall_tone": {{
      "signal": "Bullish|Bearish|Neutral|Mixed",
      "confidence": "High|Medium|Low",
      "reasoning": "One sentence summarizing {ticker}'s news tone"
    }},
    "tone_direction": {{
      "signal": "Improving|Stable|Deteriorating",
      "reasoning": "One sentence on whether sentiment is shifting for {ticker}"
    }},
    "key_events": [
      {{
        "event": "Brief description of event 1",
        "impact": "Positive|Negative|Neutral",
        "materiality": "High|Medium|Low"
      }}
    ],
    "key_themes": ["theme 1", "theme 2"],
    "news_concentration": {{
      "signal": "Single Event Dominated|Diverse Coverage|Low Coverage",
      "reasoning": "One sentence on coverage pattern"
    }},
    "controversy_flag": {{
      "detected": false,
      "description": "None" 
    }},
    "catalyst_detected": {{
      "detected": false,
      "description": "None",
      "expected_impact": "None"
    }},
    "industry_context": {{
      "signal": "Tailwind|Headwind|Neutral",
      "reasoning": "One sentence on industry sentiment affecting {ticker}"
    }},
    "risk_signals": ["risk 1 for {ticker}", "risk 2 for {ticker}"]
  }}
}}

**CRITICAL RULES:**
- All analysis must be about {ticker} ({company_name}). Do NOT reference other stocks unless comparing industry context.
- key_events should be DEDUPLICATED — group related headlines into single events (max 5 events).
- If there are fewer than 3 articles, set confidence to "Low" and note limited data.
- Return ONLY valid JSON, no preamble or markdown fences."""
    
    def get_raw_data(self, ticker: str, company_name: str = None, 
                     industry: str = None) -> Dict:
        """
        Get raw sentiment data without LLM analysis.
        Useful for the lead agent's score calculator which needs article objects.
        """
        ticker = ticker.upper()
        
        if not company_name or not industry:
            try:
                info = self._tech_calc.get_stock_info(ticker)
                company_name = company_name or info.get('name', ticker)
                industry = industry or info.get('industry', 'Unknown')
            except Exception:
                company_name = company_name or ticker
                industry = industry or 'Unknown'
        
        return self.fetcher.fetch_all(
            ticker=ticker,
            company_name=company_name,
            industry=industry,
            max_company_news=25,
            max_industry_news=10
        )


# =============================================================================
# Convenience Function
# =============================================================================

def analyze_sentiment(ticker: str) -> str:
    """Simple function to get sentiment analysis"""
    agent = SentimentAnalystAgent()
    return agent.analyze(ticker)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Analyst Agent')
    parser.add_argument('tickers', nargs='+', help='Stock ticker(s) to analyze')
    args = parser.parse_args()
    
    agent = SentimentAnalystAgent()
    
    for ticker in args.tickers:
        print(f"\n{'='*60}")
        print(f"Sentiment Analysis: {ticker}")
        print(f"{'='*60}\n")
        print(agent.analyze(ticker))
        print()