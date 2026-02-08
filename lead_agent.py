"""
General Analyst Agent Module
=============================
Comprehensive investment analysis combining:
- Technical Analysis (price patterns, momentum, trend indicators)
- Fundamental Analysis (valuation, quality, growth metrics)
- Sentiment Analysis (news sentiment, market perception)

UPDATED: Continuous LLM Decision Approach
- LLM self-assesses if detailed fundamental analysis is needed
- Intelligent context-aware decisions (not hardcoded rules)
- Only fetches detailed when truly necessary (~20% of stocks)

Provides:
- Holistic company assessment
- Near-term price outlook
- Position recommendations (buy/sell, calls/puts)
- Risk assessment and confidence levels

Usage:
    from lead_agent import GeneralAnalystAgent
    
    agent = GeneralAnalystAgent()
    analysis = agent.analyze("AAPL")
    
    # Get specific recommendation
    rec = agent.get_recommendation("TSLA")
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

# Import analysis modules
from technical import TechnicalCalculator
from fundamental import FundamentalCalculator
from sentiment import SentimentDataFetcher

# Import specialized agents for LLM-powered insights
try:
    from technical_agent import TechnicalAnalyzerAgent
    HAS_TECHNICAL_AGENT = True
except ImportError:
    HAS_TECHNICAL_AGENT = False

try:
    from fundamental_agent import FundamentalAnalystAgent
    HAS_FUNDAMENTAL_AGENT = True
except ImportError:
    HAS_FUNDAMENTAL_AGENT = False

load_dotenv()


# =============================================================================
# Data Models
# =============================================================================

class Signal(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


class OptionsStrategy(Enum):
    LONG_CALL = "Long Call"
    LONG_PUT = "Long Put"
    COVERED_CALL = "Covered Call"
    PROTECTIVE_PUT = "Protective Put"
    BULL_CALL_SPREAD = "Bull Call Spread"
    BEAR_PUT_SPREAD = "Bear Put Spread"
    IRON_CONDOR = "Iron Condor"
    STRADDLE = "Straddle"
    NO_OPTIONS = "No Options Play"


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    ticker: str
    company_name: str
    sector: str
    industry: str
    current_price: float
    
    # Scores (0-100)
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    overall_score: float
    
    # Signals
    signal: Signal
    confidence: float  # 0-100
    
    # Options
    options_strategy: OptionsStrategy
    options_rationale: str
    
    # Price targets
    price_target_low: float
    price_target_mid: float
    price_target_high: float
    target_timeframe: str
    
    # Key insights
    bull_case: List[str]
    bear_case: List[str]
    catalysts: List[str]
    risks: List[str]
    
    # Full analysis
    technical_summary: str
    fundamental_summary: str
    sentiment_summary: str
    overall_analysis: str
    
    analysis_date: str


# =============================================================================
# Score Calculator
# =============================================================================

class ScoreCalculator:
    """Calculates normalized scores from raw data"""
    
    @staticmethod
    def calculate_technical_score(indicators: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate technical score (0-100) from indicators.
        Returns score and component breakdown.
        """
        scores = {}
        weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'moving_averages': 0.25,
            'volume': 0.10,
            'volatility': 0.10
        }
        
        # Trend Score (MACD, ADX)
        macd = indicators.get('trend', {}).get('macd', {})
        adx = indicators.get('trend', {}).get('adx', 0)
        
        trend_score = 50  # Neutral base
        if macd:
            if macd.get('histogram', 0) > 0:
                trend_score += 15
            else:
                trend_score -= 15
            if macd.get('macd', 0) > macd.get('signal', 0):
                trend_score += 10
            else:
                trend_score -= 10
        if adx > 25:
            trend_score += 10  # Strong trend
        elif adx < 20:
            trend_score -= 5  # Weak trend
        scores['trend'] = max(0, min(100, trend_score))
        
        # Momentum Score (RSI)
        rsi = indicators.get('momentum', {}).get('rsi_14', 50)
        momentum_10d = indicators.get('momentum', {}).get('momentum_10d', 0)
        
        if rsi < 30:
            momentum_score = 80  # Oversold = bullish
        elif rsi > 70:
            momentum_score = 20  # Overbought = bearish
        else:
            momentum_score = 50 + (50 - rsi) * 0.5  # Linear scale
        
        if momentum_10d > 5:
            momentum_score += 10
        elif momentum_10d < -5:
            momentum_score -= 10
        scores['momentum'] = max(0, min(100, momentum_score))
        
        # Moving Average Score
        ma = indicators.get('moving_averages', {})
        price = indicators.get('price', {}).get('current', 0)
        sma_20 = ma.get('sma_20', price)
        sma_50 = ma.get('sma_50', price)
        sma_200 = ma.get('sma_200', price) or price
        
        ma_score = 50
        if price > sma_20:
            ma_score += 10
        else:
            ma_score -= 10
        if price > sma_50:
            ma_score += 15
        else:
            ma_score -= 15
        if price > sma_200:
            ma_score += 15
        else:
            ma_score -= 15
        if sma_20 > sma_50:  # Golden cross signal
            ma_score += 10
        else:  # Death cross signal
            ma_score -= 10
        scores['moving_averages'] = max(0, min(100, ma_score))
        
        # Volume Score
        volume = indicators.get('volume', {})
        vol_ratio = volume.get('ratio', 1)
        vol_trend = volume.get('trend', 0)
        
        volume_score = 50
        if vol_ratio > 1.5:
            volume_score += 20  # High volume
        elif vol_ratio < 0.5:
            volume_score -= 10  # Low volume
        if vol_trend > 0.1:
            volume_score += 15
        elif vol_trend < -0.1:
            volume_score -= 10
        scores['volume'] = max(0, min(100, volume_score))
        
        # Volatility Score (lower is better for stability)
        volatility = indicators.get('volatility', {})
        hist_vol = volatility.get('historical_vol', 0.3)
        bb = volatility.get('bollinger', {})
        
        volatility_score = 50
        if hist_vol < 0.2:
            volatility_score += 20  # Low volatility
        elif hist_vol > 0.5:
            volatility_score -= 20  # High volatility
        
        # Check Bollinger Band position
        bb_upper = bb.get('upper', price * 1.1)
        bb_lower = bb.get('lower', price * 0.9)
        if price < bb_lower:
            volatility_score += 15  # Near lower band = potential bounce
        elif price > bb_upper:
            volatility_score -= 15  # Near upper band = potential pullback
        scores['volatility'] = max(0, min(100, volatility_score))
        
        # Calculate weighted total
        total_score = sum(scores[k] * weights[k] for k in weights)
        
        return round(total_score, 1), scores
    
    @staticmethod
    def calculate_fundamental_score(data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate fundamental score (0-100) from fundamental data.
        Returns score and component breakdown.
        """
        scores = {}
        weights = {
            'valuation': 0.25,
            'profitability': 0.25,
            'growth': 0.20,
            'quality': 0.15,
            'financial_health': 0.15
        }
        
        # Valuation Score
        val = data.get('valuation_ratios', {})
        val_score = 50
        
        pe = val.get('pe_ratio', 0)
        if 0 < pe < 15:
            val_score += 20
        elif 15 <= pe < 25:
            val_score += 10
        elif pe > 40:
            val_score -= 20
        
        peg = val.get('peg_ratio', 0)
        if 0 < peg < 1:
            val_score += 20
        elif 1 <= peg < 2:
            val_score += 5
        elif peg > 3:
            val_score -= 15
        
        fcf_yield = val.get('fcf_yield', 0)
        if fcf_yield > 5:
            val_score += 10
        elif fcf_yield < 0:
            val_score -= 10
        
        scores['valuation'] = max(0, min(100, val_score))
        
        # Profitability Score
        prof = data.get('profitability_ratios', {})
        prof_score = 50
        
        roe = prof.get('return_on_equity', 0)
        if roe > 20:
            prof_score += 25
        elif roe > 15:
            prof_score += 15
        elif roe > 10:
            prof_score += 5
        elif roe < 5:
            prof_score -= 15
        
        net_margin = prof.get('net_profit_margin', 0)
        if net_margin > 20:
            prof_score += 15
        elif net_margin > 10:
            prof_score += 10
        elif net_margin < 0:
            prof_score -= 20
        
        roic = prof.get('return_on_invested_capital', 0)
        if roic > 15:
            prof_score += 10
        elif roic < 5:
            prof_score -= 10
        
        scores['profitability'] = max(0, min(100, prof_score))
        
        # Growth Score
        growth = data.get('growth_metrics', {})
        growth_score = 50
        
        rev_growth = growth.get('revenue_growth_yoy', 0)
        if rev_growth > 20:
            growth_score += 25
        elif rev_growth > 10:
            growth_score += 15
        elif rev_growth > 0:
            growth_score += 5
        elif rev_growth < -10:
            growth_score -= 20
        
        earn_growth = growth.get('earnings_growth_yoy', 0)
        if earn_growth > 20:
            growth_score += 20
        elif earn_growth > 10:
            growth_score += 10
        elif earn_growth < -10:
            growth_score -= 15
        
        scores['growth'] = max(0, min(100, growth_score))
        
        # Quality Score (Altman Z, Piotroski F)
        qual = data.get('quality_scores', {})
        qual_score = 50
        
        z_score = qual.get('altman_z_score', 0)
        if z_score > 3:
            qual_score += 25
        elif z_score > 1.8:
            qual_score += 10
        elif z_score < 1.8:
            qual_score -= 25
        
        f_score = qual.get('piotroski_f_score', 5)
        if f_score >= 7:
            qual_score += 20
        elif f_score >= 5:
            qual_score += 5
        elif f_score <= 3:
            qual_score -= 15
        
        qoe = qual.get('quality_of_earnings', 1)
        if qoe > 1.2:
            qual_score += 5
        elif qoe < 0.8:
            qual_score -= 10
        
        scores['quality'] = max(0, min(100, qual_score))
        
        # Financial Health Score
        liq = data.get('liquidity_ratios', {})
        lev = data.get('leverage_ratios', {})
        health_score = 50
        
        current_ratio = liq.get('current_ratio', 0)
        if current_ratio > 2:
            health_score += 15
        elif current_ratio > 1.5:
            health_score += 10
        elif current_ratio < 1:
            health_score -= 20
        
        de_ratio = lev.get('debt_to_equity', 0)
        if de_ratio < 0.5:
            health_score += 15
        elif de_ratio < 1:
            health_score += 5
        elif de_ratio > 2:
            health_score -= 20
        
        int_coverage = lev.get('interest_coverage', 0)
        if int_coverage > 10:
            health_score += 15
        elif int_coverage > 5:
            health_score += 5
        elif 0 < int_coverage < 2:
            health_score -= 15
        
        scores['financial_health'] = max(0, min(100, health_score))
        
        # Calculate weighted total
        total_score = sum(scores[k] * weights[k] for k in weights)
        
        return round(total_score, 1), scores
    
    @staticmethod
    def calculate_sentiment_score(news_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate sentiment score (0-100) from news/sentiment data.
        Returns score and component breakdown.
        """
        scores = {}
        
        articles = news_data.get('company_news', {}).get('articles', [])
        
        if not articles:
            return 50.0, {'news_sentiment': 50, 'news_volume': 50, 'recency': 50}
        
        # Calculate average sentiment from articles with sentiment hints
        sentiment_values = [a.sentiment_hint for a in articles if a.sentiment_hint is not None]
        
        if sentiment_values:
            # Alpha Vantage sentiment is typically -1 to 1
            avg_sentiment = sum(sentiment_values) / len(sentiment_values)
            # Convert to 0-100 scale
            news_sentiment = (avg_sentiment + 1) * 50
        else:
            # No sentiment data, use neutral
            news_sentiment = 50
        
        scores['news_sentiment'] = max(0, min(100, news_sentiment))
        
        # News volume score (more coverage can be positive or negative)
        num_articles = len(articles)
        if num_articles > 20:
            volume_score = 70  # High coverage
        elif num_articles > 10:
            volume_score = 60
        elif num_articles > 5:
            volume_score = 50
        else:
            volume_score = 40  # Low coverage
        scores['news_volume'] = volume_score
        
        # Recency score (recent news weighted more)
        recent_count = 0
        for a in articles:
            try:
                # Handle timezone-aware datetimes
                pub_time = a.published_at
                now = datetime.now()
                if pub_time.tzinfo is not None:
                    pub_time = pub_time.replace(tzinfo=None)
                if (now - pub_time).days < 3:
                    recent_count += 1
            except:
                pass
        if recent_count > 5:
            recency_score = 70
        elif recent_count > 2:
            recency_score = 60
        else:
            recency_score = 50
        scores['recency'] = recency_score
        
        # Weighted average
        weights = {'news_sentiment': 0.6, 'news_volume': 0.2, 'recency': 0.2}
        total_score = sum(scores[k] * weights[k] for k in weights)
        
        return round(total_score, 1), scores
    
    @staticmethod
    def calculate_overall_score(technical: float, fundamental: float, 
                                sentiment: float) -> float:
        """
        Calculate overall score with dynamic weighting.
        """
        # Base weights
        weights = {
            'technical': 0.35,
            'fundamental': 0.45,
            'sentiment': 0.20
        }
        
        overall = (technical * weights['technical'] + 
                  fundamental * weights['fundamental'] + 
                  sentiment * weights['sentiment'])
        
        return round(overall, 1)
    
    @staticmethod
    def score_to_signal(score: float) -> Signal:
        """Convert score to trading signal"""
        if score >= 75:
            return Signal.STRONG_BUY
        elif score >= 60:
            return Signal.BUY
        elif score >= 40:
            return Signal.HOLD
        elif score >= 25:
            return Signal.SELL
        else:
            return Signal.STRONG_SELL
    
    @staticmethod
    def determine_options_strategy(signal: Signal, volatility: float, 
                                   days_to_earnings: int = None,
                                   sentiment_score: float = 50) -> Tuple[OptionsStrategy, str]:
        """
        Determine appropriate options strategy based on analysis.
        """
        high_vol = volatility > 0.4
        
        if signal == Signal.STRONG_BUY:
            if high_vol:
                return (OptionsStrategy.BULL_CALL_SPREAD, 
                       "Strong bullish signal with high volatility - spread limits risk")
            else:
                return (OptionsStrategy.LONG_CALL,
                       "Strong bullish signal with reasonable volatility - direct call exposure")
        
        elif signal == Signal.BUY:
            return (OptionsStrategy.BULL_CALL_SPREAD,
                   "Moderately bullish - spread provides defined risk/reward")
        
        elif signal == Signal.HOLD:
            if high_vol:
                return (OptionsStrategy.IRON_CONDOR,
                       "Neutral outlook with high IV - collect premium from range-bound movement")
            else:
                return (OptionsStrategy.COVERED_CALL,
                       "Neutral outlook - generate income while holding shares")
        
        elif signal == Signal.SELL:
            return (OptionsStrategy.BEAR_PUT_SPREAD,
                   "Moderately bearish - spread provides defined risk/reward")
        
        else:  # STRONG_SELL
            if high_vol:
                return (OptionsStrategy.BEAR_PUT_SPREAD,
                       "Strong bearish signal with high volatility - spread limits risk")
            else:
                return (OptionsStrategy.LONG_PUT,
                       "Strong bearish signal - direct put exposure")


# =============================================================================
# General Analyst Agent
# =============================================================================

class GeneralAnalystAgent:
    """
    Comprehensive investment analyst combining technical, fundamental,
    and sentiment analysis with LLM-powered insights.
    
    UPDATED: Continuous LLM Decision Approach
    - LLM intelligently decides when detailed fundamental analysis is needed
    - No hardcoded rules - context-aware decisions
    """
    
    def __init__(self, model: str = "claude-sonnet-4-5", use_specialist_agents: bool = True):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        # Initialize data calculators (for raw data and scoring)
        self.technical = TechnicalCalculator()
        self.fundamental = FundamentalCalculator()
        self.sentiment = SentimentDataFetcher()
        self.scorer = ScoreCalculator()
        
        # Initialize specialist agents (for LLM-powered insights)
        self.use_specialist_agents = use_specialist_agents
        self.technical_agent = None
        self.fundamental_agent = None
        
        if use_specialist_agents:
            if HAS_TECHNICAL_AGENT:
                try:
                    self.technical_agent = TechnicalAnalyzerAgent(model=model)
                except Exception as e:
                    print(f"Warning: Could not initialize TechnicalAnalyzerAgent: {e}")
            
            if HAS_FUNDAMENTAL_AGENT:
                try:
                    self.fundamental_agent = FundamentalAnalystAgent(model=model)
                except Exception as e:
                    print(f"Warning: Could not initialize FundamentalAnalystAgent: {e}")
        
        # Initialize LLM for final synthesis
        self.llm = ChatAnthropic(model_name=model, temperature=0)
    
    def gather_data(self, ticker: str) -> Dict[str, Any]:
        """
        SIMPLIFIED DATA GATHERING:
        1. Get all three inputs (technical, fundamental structured, sentiment)
        2. NO preemptive decision about detailed - let synthesis LLM decide
        """
        ticker = ticker.upper()
        data: Dict[str, Any] = {'ticker': ticker}
        
        # Technical data
        try:
            data['technical'] = self.technical.get_all_indicators(ticker)
            data['stock_info'] = self.technical.get_stock_info(ticker)
        except Exception as e:
            data['technical'] = {}
            data['stock_info'] = {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown'}
            data['technical_error'] = str(e)
        
        # Technical agent insight
        if self.use_specialist_agents and self.technical_agent:
            try:
                data['technical_insight'] = self.technical_agent.analyze(ticker)
            except Exception as e:
                data['technical_insight'] = None
        else:
            data['technical_insight'] = None
        
        # Fundamental raw data (for scoring)
        try:
            data['fundamental'] = self.fundamental.get_all_fundamentals(ticker)
        except Exception as e:
            data['fundamental'] = {}
            data['fundamental_error'] = str(e)
        
        # Fundamental structured analysis (ALWAYS get this)
        if self.use_specialist_agents and self.fundamental_agent:
            try:
                structured_analysis = self.fundamental_agent.analyze(
                    ticker, 
                    analysis_type="for_synthesis"
                )
                data['fundamental_structured'] = structured_analysis
            except Exception as e:
                data['fundamental_structured'] = None
                data['fundamental_structured_error'] = str(e)
        else:
            data['fundamental_structured'] = None
        
        # NOTE: We do NOT fetch detailed here - synthesis LLM will decide
        data['fundamental_detailed'] = None
        
        # Sentiment data
        try:
            company_name = data.get('stock_info', {}).get('name', ticker)
            industry = data.get('stock_info', {}).get('industry', '')
            data['sentiment'] = self.sentiment.fetch_all(
                ticker, company_name, industry, 
                max_company_news=25, max_industry_news=10
            )
        except Exception as e:
            data['sentiment'] = {}
            data['sentiment_error'] = str(e)
        
        return data
    
    def calculate_scores(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all scores from gathered data.
        """
        scores: Dict[str, Any] = {}
        
        # Technical score
        if data.get('technical'):
            tech_score, tech_breakdown = self.scorer.calculate_technical_score(data['technical'])
            scores['technical'] = {'score': tech_score, 'breakdown': tech_breakdown}
        else:
            scores['technical'] = {'score': 50.0, 'breakdown': {}}
        
        # Fundamental score
        if data.get('fundamental'):
            fund_score, fund_breakdown = self.scorer.calculate_fundamental_score(data['fundamental'])
            scores['fundamental'] = {'score': fund_score, 'breakdown': fund_breakdown}
        else:
            scores['fundamental'] = {'score': 50.0, 'breakdown': {}}
        
        # Sentiment score
        if data.get('sentiment'):
            sent_score, sent_breakdown = self.scorer.calculate_sentiment_score(data['sentiment'])
            scores['sentiment'] = {'score': sent_score, 'breakdown': sent_breakdown}
        else:
            scores['sentiment'] = {'score': 50.0, 'breakdown': {}}
        
        # Overall score
        scores['overall'] = self.scorer.calculate_overall_score(
            scores['technical']['score'],
            scores['fundamental']['score'],
            scores['sentiment']['score']
        )
        
        # Signal
        scores['signal'] = self.scorer.score_to_signal(scores['overall'])
        
        # Confidence based on data quality and agreement
        tech_fund_diff = abs(scores['technical']['score'] - scores['fundamental']['score'])
        if tech_fund_diff < 10:
            confidence = 85
        elif tech_fund_diff < 20:
            confidence = 70
        else:
            confidence = 55
        
        # Adjust for data availability
        if not data.get('technical'):
            confidence -= 15
        if not data.get('fundamental'):
            confidence -= 20
        if not data.get('sentiment'):
            confidence -= 5
        
        scores['confidence'] = max(30, min(95, confidence))
        
        return scores
    
    def calculate_price_targets(self, data: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate price targets based on analysis.
        """
        current_price = data.get('technical', {}).get('price', {}).get('current', 0)
        if not current_price:
            current_price = data.get('fundamental', {}).get('valuation_ratios', {}).get('price', 0)
        
        if not current_price:
            return {'low': 0, 'mid': 0, 'high': 0, 'timeframe': '3-6 months'}
        
        # Base adjustment on overall score
        score = scores['overall']
        
        # Score to expected return mapping
        if score >= 75:
            expected_return = 0.15  # +15%
            range_width = 0.10
        elif score >= 60:
            expected_return = 0.08  # +8%
            range_width = 0.08
        elif score >= 40:
            expected_return = 0.0  # flat
            range_width = 0.06
        elif score >= 25:
            expected_return = -0.08  # -8%
            range_width = 0.08
        else:
            expected_return = -0.15  # -15%
            range_width = 0.10
        
        # Adjust for volatility
        hist_vol = data.get('technical', {}).get('volatility', {}).get('historical_vol', 0.3)
        range_width = max(range_width, hist_vol * 0.5)
        
        mid_target = current_price * (1 + expected_return)
        low_target = mid_target * (1 - range_width)
        high_target = mid_target * (1 + range_width)
        
        return {
            'low': round(low_target, 2),
            'mid': round(mid_target, 2),
            'high': round(high_target, 2),
            'current': round(current_price, 2),
            'timeframe': '3-6 months'
        }
    
    def generate_llm_analysis(self, data: Dict[str, Any], scores: Dict[str, Any], 
                             targets: Dict[str, Any]) -> Dict[str, Any]:
        """
        TWO-STAGE SYNTHESIS WITH SELF-ASSESSMENT:
        
        Stage 1: LLM decides if it needs detailed fundamental
        Stage 2: LLM synthesizes (with or without detailed)
        
        This approach lets the LLM see full context before deciding,
        and only costs 1 extra LLM call when detailed is actually needed.
        """
        
        ticker = data['ticker']
        stock_info = data.get('stock_info', {})
        
        # Prepare summaries
        tech_summary = self._prepare_technical_summary(data)
        fund_summary = self._prepare_fundamental_summary(data)
        sent_summary = self._format_sentiment_data(data.get('sentiment', {}))[:800]
        
        scores_text = f"""Technical: {scores['technical']['score']:.1f}/100
Fundamental: {scores['fundamental']['score']:.1f}/100
Sentiment: {scores['sentiment']['score']:.1f}/100
Overall: {scores['overall']:.1f}/100"""
        
        tech = data.get('technical', {})
        volatility = tech.get('volatility', {}).get('historical_vol', 0.3)
        options_strategy, options_rationale = self.scorer.determine_options_strategy(
            scores['signal'], volatility, sentiment_score=scores['sentiment']['score']
        )
        
        # =========================================================================
        # STAGE 1: SELF-ASSESSMENT - Does LLM need detailed fundamental?
        # =========================================================================
        
        assessment_prompt = self._create_assessment_prompt()
        
        assessment_response = self.llm.invoke(assessment_prompt.format(
            ticker=ticker,
            company_name=stock_info.get('name', ticker),
            sector=stock_info.get('sector', 'Unknown'),
            industry=stock_info.get('industry', 'Unknown'),
            current_price=targets['current'],
            technical_summary=tech_summary,
            fundamental_summary=fund_summary,
            sentiment_summary=sent_summary,
            scores_summary=scores_text,
            signal=scores['signal'].value,
            confidence=scores['confidence']
        ))
        
        # Parse assessment
        if isinstance(assessment_response.content, str):
            assessment_text = assessment_response.content
        elif isinstance(assessment_response.content, list):
            text_parts = [item for item in assessment_response.content if isinstance(item, str)]
            assessment_text = '\n'.join(text_parts) if text_parts else str(assessment_response.content)
        else:
            assessment_text = str(assessment_response.content)
        
        needs_detailed, reason = self._parse_assessment(assessment_text)
        
        # =========================================================================
        # CONDITIONAL FETCH: Get detailed if LLM determined it's needed
        # =========================================================================
        
        fund_detailed = None
        if needs_detailed:
            print(f"\n🔍 {ticker}: LLM requesting detailed fundamental analysis")
            print(f"   Reason: {reason}\n")
            
            try:
                # IMPORTANT: Verify we're fetching for the correct ticker
                print(f"   Fetching detailed analysis for: {ticker}")
                fund_detailed = self.fundamental_agent.analyze(
                    ticker,
                    analysis_type="comprehensive"
                )
                
                # Verify the returned analysis is for the correct ticker
                if ticker.upper() not in fund_detailed[:200].upper():
                    print(f"   WARNING: Detailed analysis may be for wrong stock!")
                    print(f"   Expected: {ticker}, checking content...")
                
                # Update fundamental summary to include detailed
                fund_summary = self._prepare_fundamental_summary(data, detailed=fund_detailed)
            except Exception as e:
                print(f"   Failed to fetch detailed: {e}")
                reason = f"Requested but failed: {e}"
        
        # =========================================================================
        # STAGE 2: FINAL SYNTHESIS
        # =========================================================================
        
        synthesis_prompt = self._create_synthesis_prompt()
        
        # Build dynamic strings for template
        if fund_detailed:
            detailed_status = "received detailed fundamental analysis"
            detail_fetch_info = f"✓ DETAILED FUNDAMENTAL ANALYSIS WAS FETCHED\nReason: {reason}"
            additional_context = "- How does the detailed fundamental context resolve any ambiguities?"
        else:
            detailed_status = "determined structured signals are sufficient"
            detail_fetch_info = "✓ STRUCTURED SIGNALS WERE SUFFICIENT"
            additional_context = ""
        
        synthesis_response = self.llm.invoke(synthesis_prompt.format(
            ticker=ticker,
            company_name=stock_info.get('name', ticker),
            sector=stock_info.get('sector', 'Unknown'),
            industry=stock_info.get('industry', 'Unknown'),
            current_price=targets['current'],
            technical_summary=tech_summary,
            fundamental_summary=fund_summary,
            sentiment_summary=sent_summary,
            scores_summary=scores_text,
            signal=scores['signal'].value,
            confidence=scores['confidence'],
            price_target_low=targets['low'],
            price_target_mid=targets['mid'],
            price_target_high=targets['high'],
            target_timeframe=targets['timeframe'],
            options_strategy=options_strategy.value,
            detailed_status=detailed_status,
            detail_fetch_info=detail_fetch_info,
            additional_context=additional_context
        ))
        
        if isinstance(synthesis_response.content, str):
            analysis_text = synthesis_response.content
        elif isinstance(synthesis_response.content, list):
            text_parts = [item for item in synthesis_response.content if isinstance(item, str)]
            analysis_text = '\n'.join(text_parts) if text_parts else str(synthesis_response.content)
        else:
            analysis_text = str(synthesis_response.content)
        
        return {
            'full_analysis': analysis_text,
            'options_strategy': options_strategy,
            'options_rationale': options_rationale,
            'used_detailed_fundamental': bool(fund_detailed),
            'detail_fetch_reason': reason if fund_detailed else None,
            'llm_requested_detailed': needs_detailed
        }
    
    def _parse_assessment(self, text: str) -> tuple:
        """
        Parse the LLM's self-assessment response.
        
        Expected format:
        <decision>YES</decision>
        <reason>explanation</reason>
        
        Returns: (needs_detailed: bool, reason: str)
        """
        needs_detailed = False
        reason = "Structured signals sufficient"
        
        try:
            # Look for <decision> tags
            if '<decision>' in text and '</decision>' in text:
                start = text.find('<decision>') + 10
                end = text.find('</decision>')
                decision = text[start:end].strip().upper()
                needs_detailed = decision == 'YES'
            
            # Look for <reason> tags
            if '<reason>' in text and '</reason>' in text:
                start = text.find('<reason>') + 8
                end = text.find('</reason>')
                reason = text[start:end].strip()
        except Exception as e:
            # Fallback: search for keywords
            text_upper = text.upper()
            if 'YES' in text_upper and ('NEED' in text_upper or 'REQUIRE' in text_upper):
                needs_detailed = True
                # Try to extract reason
                lines = text.split('\n')
                for line in lines:
                    if 'reason' in line.lower() and len(line) > 20:
                        reason = line.split(':', 1)[-1].strip()
                        break
        
        return needs_detailed, reason
    
    def _prepare_technical_summary(self, data: Dict) -> str:
        """Prepare technical summary for LLM"""
        tech = data.get('technical', {})
        tech_agent_insight = data.get('technical_insight')
        
        if tech_agent_insight:
            return f"""Technical Analyst Report:
{tech_agent_insight[:600]}

Key Metrics:
- RSI(14): {tech.get('momentum', {}).get('rsi_14', 50):.1f}
- MACD: {'Bullish' if tech.get('trend', {}).get('macd', {}).get('histogram', 0) > 0 else 'Bearish'}
- Volume: {tech.get('volume', {}).get('ratio', 1):.2f}x average"""
        else:
            return self._format_technical_data(tech)[:1000]
    
    def _prepare_fundamental_summary(self, data: Dict, detailed: str = None) -> str:
        """
        Prepare fundamental summary for LLM.
        If detailed is provided, includes it; otherwise just structured.
        """
        ticker = data.get('ticker', 'UNKNOWN')
        fund_structured = data.get('fundamental_structured')
        
        if not fund_structured or not isinstance(fund_structured, dict):
            # Fallback to basic metrics
            fund = data.get('fundamental', {})
            val = fund.get('valuation_ratios', {})
            prof = fund.get('profitability_ratios', {})
            qual = fund.get('quality_scores', {})
            
            return f"""Basic Fundamental Metrics for {ticker}:
P/E: {val.get('pe_ratio', 0):.1f} | PEG: {val.get('peg_ratio', 0):.2f} | P/B: {val.get('price_to_book', 0):.2f}
ROE: {prof.get('return_on_equity', 0):.1f}% | Net Margin: {prof.get('net_profit_margin', 0):.1f}%
Altman Z: {qual.get('altman_z_score', 0):.2f} | Piotroski F: {qual.get('piotroski_f_score', 0)}/9"""
        
        llm_analysis = fund_structured.get('llm_analysis', {})
        
        # Start with structured
        summary = f"""=== FUNDAMENTAL ANALYST (STRUCTURED) FOR {ticker} ===

{self._format_structured_signals_compact(llm_analysis)}"""
        
        # Add detailed if provided
        if detailed:
            summary += f"""

{'='*60}
=== DETAILED FUNDAMENTAL ANALYSIS FOR {ticker} ===
{'='*60}

{detailed[:2000]}

[Full detailed analysis available for deep context]"""
        
        return summary
    
    def _format_structured_signals_compact(self, llm_analysis: Dict) -> str:
        """Compact formatting of structured signals"""
        
        val = llm_analysis.get('valuation_verdict', {})
        qual = llm_analysis.get('quality_verdict', {})
        growth = llm_analysis.get('growth_verdict', {})
        health = llm_analysis.get('financial_health', {})
        
        return f"""VALUATION: {val.get('signal', 'N/A')} (Confidence: {val.get('confidence', 'N/A')})
  → {val.get('reasoning', 'N/A')}

QUALITY: {qual.get('signal', 'N/A')} (Confidence: {qual.get('confidence', 'N/A')})
  → Z: {qual.get('altman_z', 'N/A')} | F: {qual.get('piotroski_f', 'N/A')}
  → {qual.get('reasoning', 'N/A')}

GROWTH: {growth.get('signal', 'N/A')} (Confidence: {growth.get('confidence', 'N/A')})
  → {growth.get('revenue_trajectory', 'N/A')}
  → {growth.get('reasoning', 'N/A')}

HEALTH: {health.get('signal', 'N/A')}
  → Leverage: {health.get('leverage_status', 'N/A')} | Liquidity: {health.get('liquidity_status', 'N/A')}

STRENGTHS: {', '.join(llm_analysis.get('key_strengths', [])[:3])}
RISKS: {', '.join(llm_analysis.get('key_risks', [])[:3])}

THESIS:
Bull: {llm_analysis.get('investment_thesis', {}).get('bull_case', 'N/A')}
Bear: {llm_analysis.get('investment_thesis', {}).get('bear_case', 'N/A')}"""
    
    def _format_technical_data(self, tech: Dict[str, Any]) -> str:
        """Format technical data for LLM"""
        if not tech:
            return "No technical data available"
        
        price = tech.get('price', {})
        ma = tech.get('moving_averages', {})
        momentum = tech.get('momentum', {})
        trend = tech.get('trend', {})
        vol = tech.get('volatility', {})
        volume = tech.get('volume', {})
        
        macd = trend.get('macd', {})
        bb = vol.get('bollinger', {})
        
        return f"""
=== TECHNICAL INDICATORS ===

PRICE ACTION:
- Current Price: ${price.get('current', 0):.2f}
- 1-Day Change: {price.get('change_1d', 0):+.2f}%
- 5-Day Change: {price.get('change_5d', 0):+.2f}%
- 20-Day Change: {price.get('change_20d', 0):+.2f}%

MOVING AVERAGES:
- SMA(20): ${ma.get('sma_20', 0):.2f}
- SMA(50): ${ma.get('sma_50', 0):.2f}
- SMA(200): ${ma.get('sma_200', 0):.2f if ma.get('sma_200') else 'N/A'}
- Price vs SMA(20): {'Above' if price.get('current', 0) > ma.get('sma_20', 0) else 'Below'}
- Price vs SMA(50): {'Above' if price.get('current', 0) > ma.get('sma_50', 0) else 'Below'}

MOMENTUM:
- RSI(14): {momentum.get('rsi_14', 50):.2f}
- RSI Signal: {'Oversold' if momentum.get('rsi_14', 50) < 30 else 'Overbought' if momentum.get('rsi_14', 50) > 70 else 'Neutral'}

TREND:
- MACD Histogram: {macd.get('histogram', 0):.4f}
- MACD Signal: {'Bullish' if macd.get('histogram', 0) > 0 else 'Bearish'}
- ADX: {trend.get('adx', 0):.2f}

VOLUME:
- Volume Ratio: {volume.get('ratio', 0):.2f}x average
- Volume Trend: {volume.get('trend', 0):+.2%}
"""
    
    def _format_sentiment_data(self, sent: Dict[str, Any]) -> str:
        """Format sentiment data for LLM"""
        if not sent:
            return "No sentiment data available"
        
        summary = sent.get('summary', {})
        company_news = sent.get('company_news', {})
        
        articles = company_news.get('articles', [])
        
        # Get recent headlines
        headlines = []
        for article in articles[:10]:
            sentiment_str = f" [Sentiment: {article.sentiment_hint:.2f}]" if article.sentiment_hint else ""
            headlines.append(f"- {article.title[:80]}...{sentiment_str}")
        
        return f"""
=== SENTIMENT DATA ===

SUMMARY:
- Total Items: {summary.get('total_items', 0)}
- Has Sentiment Scores: {summary.get('has_sentiment_scores', False)}

RECENT NEWS ({len(articles)} articles):
{chr(10).join(headlines) if headlines else 'No recent news'}
"""
    
    def _create_assessment_prompt(self) -> PromptTemplate:
        """
        STAGE 1 PROMPT: Self-assessment
        
        The LLM evaluates if it needs more fundamental context BEFORE
        attempting to synthesize a recommendation.
        """
        
        template = """You are the LEAD INVESTMENT ANALYST preparing to synthesize a recommendation for {ticker}.

You have received inputs from three specialist analysts. Your job is to FIRST assess whether you have sufficient information, THEN (in a separate step) you'll create the full synthesis.

=== STOCK INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector} | Industry: {industry}
Current Price: ${current_price}

=== INPUTS RECEIVED ===

TECHNICAL ANALYST:
{technical_summary}

FUNDAMENTAL ANALYST (STRUCTURED SIGNALS):
{fundamental_summary}

SENTIMENT ANALYST:
{sentiment_summary}

=== QUANTITATIVE SCORES ===
{scores_summary}
Signal: {signal}
Confidence: {confidence}%

=== YOUR ASSESSMENT TASK ===

Before attempting to synthesize a recommendation, evaluate:

**Can you make a CLEAR, CONFIDENT investment recommendation with the information above?**

Consider:
1. Are fundamental signals internally consistent?
2. Do fundamental signals align or conflict with technical/sentiment?
3. Are there edge cases or ambiguities that need deeper fundamental context?
4. Is this a high-stakes decision (strong buy/sell) where you need to be certain?
5. Would detailed fundamental analysis (600-800 words covering valuation, quality, growth in depth) help resolve any uncertainties?

**Decision Criteria:**

NEED DETAILED if:
- Fundamental signals conflict internally (e.g., "High Quality" + "Declining Growth" + "Undervalued")
- Cross-domain conflicts (e.g., fundamentals bullish but technicals broken, sentiment negative)
- Edge case sector/situation (biotech, utilities with unusual metrics, distressed companies)
- High conviction signal (score >75 or <25) but want to be absolutely certain
- Multiple low-confidence fundamental signals

DON'T NEED DETAILED if:
- Signals are clear and aligned across all three domains
- This is a neutral/hold situation (structured is sufficient for "wait and see")
- Technical/sentiment so strong they override fundamental nuances
- Fundamental signals are clear and high-confidence

=== OUTPUT FORMAT ===

Respond in EXACTLY this format:

<decision>YES</decision>
<reason>One sentence explaining why you need detailed fundamental analysis</reason>

OR

<decision>NO</decision>
<reason>One sentence explaining why structured signals are sufficient</reason>

Then STOP. Do not provide the full synthesis yet - that comes in the next step.

Your assessment:"""

        return PromptTemplate.from_template(template)
    
    def _create_synthesis_prompt(self) -> PromptTemplate:
        """
        STAGE 2 PROMPT: Final synthesis
        
        Now that we have all necessary information (with or without detailed),
        create the complete investment recommendation.
        """
        
        template = """You are the LEAD INVESTMENT ANALYST creating a final investment recommendation.

CRITICAL: You are analyzing {{ticker}} ({{company_name}}). Do NOT confuse this with any other stock.

You have already assessed your information needs. {detailed_status}

=== STOCK INFORMATION ===
**ANALYZING: {{ticker}} - {{company_name}}**
Sector: {{sector}} | Industry: {{industry}}
Current Price: ${{current_price}}

=== SPECIALIST INPUTS ===

TECHNICAL:
{{technical_summary}}

FUNDAMENTAL:
{{fundamental_summary}}

SENTIMENT:
{{sentiment_summary}}

=== QUANTITATIVE SCORES ===
{{scores_summary}}
Signal: {{signal}}
Confidence: {{confidence}}%

Price Targets ({{target_timeframe}}): ${{price_target_low:.2f}} / ${{price_target_mid:.2f}} / ${{price_target_high:.2f}}
Options: {{options_strategy}}

============================================================
{detail_fetch_info}
============================================================
============================================================

=== YOUR SYNTHESIS TASK ===

**REMINDER: You are analyzing {{ticker}} ({{company_name}}) - stay focused on THIS stock only.**

Create a comprehensive investment recommendation integrating all inputs FOR {{ticker}}.

**1. EXECUTIVE SUMMARY** (3-4 sentences)
- Start with: "{{ticker}} ({{company_name}})..." 
- Unified investment thesis across all three domains
- Key conviction and primary driver
- Why this is a buy/hold/sell opportunity for {{ticker}}

**2. SYNTHESIS: TECHNICAL + FUNDAMENTAL + SENTIMENT FOR {{ticker}}**
- How do the three domains interact for {{ticker}}?
- Where do specialists agree about {{ticker}} (higher conviction)?
- Where do they disagree about {{ticker}} (requires judgment)?
{additional_context}

**3. POSITION RECOMMENDATION FOR {{ticker}}**

EQUITY POSITION:
- Action: [Buy/Add/Hold/Reduce/Sell + conviction level]
- Position Size: [% based on confidence: 1-2% low confidence, 3-5% medium, 5-7% high]
- Entry Strategy: $XX.XX [specific technical level]
- Stop Loss: $XX.XX [below support + fundamental invalidation]
- Profit Target: $XX.XX [resistance + fundamental upside]
- Time Horizon: [days/weeks/months based on catalysts]

OPTIONS STRATEGY:
- Recommended: {{options_strategy}}
- Strike: [specific strike based on targets]
- Expiration: [date based on catalyst timing]
- Risk/Reward: [bull/base/bear scenarios with expected P&L]

**4. MONITORING FRAMEWORK FOR {{ticker}}**
Three specific triggers to watch for {{ticker}}:
- Technical: "If {{ticker}} price breaks $XX..."
- Fundamental: "Watch {{ticker}} quarterly earnings for..."
- Sentiment: "If negative news emerges about {{ticker}}..."

**5. BULL/BEAR CASES FOR {{ticker}}** (integrated across domains)
Bull Case (3 bullets): Each showing convergence of positive signals for {{ticker}}
Bear Case (3 bullets): Each showing risks across domains for {{ticker}}

=== GUIDELINES ===
- Be specific with prices, dates, percentages
- Own your synthesis - don't just summarize
- Show reasoning when specialists disagree
- Action-oriented: help investor make decisions
- 800-1000 words total
- **CRITICAL: Your entire analysis must be about {{ticker}} ({{company_name}}). Do not analyze any other stock.**

Begin your analysis of {{ticker}}:"""

        return PromptTemplate.from_template(template)
    
    def analyze(self, ticker: str) -> str:
        """
        Perform complete analysis and return formatted report.
        """
        ticker = ticker.upper()
        
        print(f"Gathering data for {ticker}...")
        data = self.gather_data(ticker)
        
        print("Calculating scores...")
        scores = self.calculate_scores(data)
        
        print("Calculating price targets...")
        targets = self.calculate_price_targets(data, scores)
        
        print("Generating comprehensive analysis...")
        llm_result = self.generate_llm_analysis(data, scores, targets)
        
        # Create header
        header = f"""
{'='*80}
COMPREHENSIVE INVESTMENT ANALYSIS: {ticker}
{'='*80}
Company: {data.get('stock_info', {}).get('name', ticker)}
Sector: {data.get('stock_info', {}).get('sector', 'Unknown')}
Industry: {data.get('stock_info', {}).get('industry', 'Unknown')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*80}

QUANTITATIVE SUMMARY:
---------------------
Technical Score:    {scores['technical']['score']:>6.1f}/100
Fundamental Score:  {scores['fundamental']['score']:>6.1f}/100
Sentiment Score:    {scores['sentiment']['score']:>6.1f}/100
                    ─────────
Overall Score:      {scores['overall']:>6.1f}/100

Signal: {scores['signal'].value}
Confidence: {scores['confidence']}%

Price Targets ({targets['timeframe']}):
  Current: ${targets['current']:.2f}
  Low:     ${targets['low']:.2f} ({((targets['low']/targets['current'])-1)*100:+.1f}%)
  Mid:     ${targets['mid']:.2f} ({((targets['mid']/targets['current'])-1)*100:+.1f}%)
  High:    ${targets['high']:.2f} ({((targets['high']/targets['current'])-1)*100:+.1f}%)

{'='*80}
DETAILED FUNDAMENTAL ANALYSIS: {'✓ USED' if llm_result['used_detailed_fundamental'] else '✗ NOT NEEDED'}
{f"Reason: {llm_result['detail_fetch_reason']}" if llm_result['used_detailed_fundamental'] else "Structured signals were sufficient"}
{'='*80}

DETAILED ANALYSIS:
==================

{llm_result['full_analysis']}
"""
        
        return header
    
    def get_quick_recommendation(self, ticker: str) -> Dict[str, Any]:
        """
        Get quick recommendation without full LLM analysis.
        Faster for screening multiple stocks.
        """
        ticker = ticker.upper()
        
        data = self.gather_data(ticker)
        scores = self.calculate_scores(data)
        targets = self.calculate_price_targets(data, scores)
        
        volatility = data.get('technical', {}).get('volatility', {}).get('historical_vol', 0.3)
        options_strategy, options_rationale = self.scorer.determine_options_strategy(
            scores['signal'], volatility, sentiment_score=scores['sentiment']['score']
        )
        
        return {
            'ticker': ticker,
            'company': data.get('stock_info', {}).get('name', ticker),
            'current_price': targets['current'],
            'scores': {
                'technical': scores['technical']['score'],
                'fundamental': scores['fundamental']['score'],
                'sentiment': scores['sentiment']['score'],
                'overall': scores['overall']
            },
            'signal': scores['signal'].value,
            'confidence': scores['confidence'],
            'price_targets': targets,
            'options': {
                'strategy': options_strategy.value,
                'rationale': options_rationale
            }
        }
    
    def compare_stocks(self, tickers: List[str]) -> str:
        """
        Compare multiple stocks for investment selection.
        """
        if len(tickers) < 2:
            return "Please provide at least 2 tickers for comparison."
        
        print(f"Comparing {len(tickers)} stocks...")
        
        results = []
        for ticker in tickers[:5]:  # Limit to 5
            print(f"  Analyzing {ticker}...")
            rec = self.get_quick_recommendation(ticker)
            results.append(rec)
        
        # Sort by overall score
        results.sort(key=lambda x: x['scores']['overall'], reverse=True)
        
        # Create comparison table
        header = f"""
{'='*100}
STOCK COMPARISON ANALYSIS
{'='*100}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{'Ticker':<10}{'Company':<25}{'Price':>10}{'Tech':>8}{'Fund':>8}{'Sent':>8}{'Overall':>10}{'Signal':<15}{'Conf':>8}
{'-'*100}"""
        
        rows = []
        for r in results:
            rows.append(
                f"{r['ticker']:<10}"
                f"{r['company'][:23]:<25}"
                f"${r['current_price']:>8.2f}"
                f"{r['scores']['technical']:>8.1f}"
                f"{r['scores']['fundamental']:>8.1f}"
                f"{r['scores']['sentiment']:>8.1f}"
                f"{r['scores']['overall']:>10.1f}"
                f"{r['signal']:<15}"
                f"{r['confidence']:>7.0f}%"
            )
        
        details = []
        for r in results:
            pt = r['price_targets']
            details.append(f"""
{r['ticker']}:
  Price Targets: ${pt['low']:.2f} / ${pt['mid']:.2f} / ${pt['high']:.2f}
  Options: {r['options']['strategy']} - {r['options']['rationale'][:60]}...""")
        
        return header + "\n" + "\n".join(rows) + "\n" + "-"*100 + "\n".join(details)
    
    def screen_portfolio(self, tickers: List[str]) -> str:
        """
        Screen a portfolio of stocks and provide overview.
        """
        print(f"Screening portfolio of {len(tickers)} stocks...")
        
        results = []
        for ticker in tickers:
            try:
                rec = self.get_quick_recommendation(ticker)
                results.append(rec)
            except Exception as e:
                print(f"  Error with {ticker}: {e}")
        
        if not results:
            return "No stocks could be analyzed."
        
        # Categorize by signal
        strong_buys = [r for r in results if r['signal'] == 'Strong Buy']
        buys = [r for r in results if r['signal'] == 'Buy']
        holds = [r for r in results if r['signal'] == 'Hold']
        sells = [r for r in results if r['signal'] == 'Sell']
        strong_sells = [r for r in results if r['signal'] == 'Strong Sell']
        
        # Calculate portfolio metrics
        avg_tech = sum(r['scores']['technical'] for r in results) / len(results)
        avg_fund = sum(r['scores']['fundamental'] for r in results) / len(results)
        avg_sent = sum(r['scores']['sentiment'] for r in results) / len(results)
        avg_overall = sum(r['scores']['overall'] for r in results) / len(results)
        
        report = f"""
{'='*80}
PORTFOLIO SCREENING REPORT
{'='*80}
Stocks Analyzed: {len(results)}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

PORTFOLIO HEALTH:
-----------------
Average Technical Score:    {avg_tech:.1f}/100
Average Fundamental Score:  {avg_fund:.1f}/100
Average Sentiment Score:    {avg_sent:.1f}/100
Average Overall Score:      {avg_overall:.1f}/100

SIGNAL DISTRIBUTION:
--------------------
Strong Buy:  {len(strong_buys)} ({', '.join(r['ticker'] for r in strong_buys) or 'None'})
Buy:         {len(buys)} ({', '.join(r['ticker'] for r in buys) or 'None'})
Hold:        {len(holds)} ({', '.join(r['ticker'] for r in holds) or 'None'})
Sell:        {len(sells)} ({', '.join(r['ticker'] for r in sells) or 'None'})
Strong Sell: {len(strong_sells)} ({', '.join(r['ticker'] for r in strong_sells) or 'None'})

TOP PICKS (by Overall Score):
-----------------------------"""
        
        sorted_results = sorted(results, key=lambda x: x['scores']['overall'], reverse=True)
        for i, r in enumerate(sorted_results[:5], 1):
            report += f"\n{i}. {r['ticker']} - Score: {r['scores']['overall']:.1f}, Signal: {r['signal']}"
        
        report += f"""

CONCERNS (Low Scores):
----------------------"""
        for r in sorted_results[-3:]:
            if r['scores']['overall'] < 50:
                report += f"\n  {r['ticker']} - Score: {r['scores']['overall']:.1f}, Signal: {r['signal']}"
        
        return report


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_stock(ticker: str) -> str:
    """Simple function to analyze a single stock"""
    agent = GeneralAnalystAgent()
    return agent.analyze(ticker)


def get_recommendation(ticker: str) -> Dict:
    """Simple function to get quick recommendation"""
    agent = GeneralAnalystAgent()
    return agent.get_quick_recommendation(ticker)


def compare_stocks(tickers: List[str]) -> str:
    """Simple function to compare stocks"""
    agent = GeneralAnalystAgent()
    return agent.compare_stocks(tickers)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='General Analyst Agent')
    parser.add_argument('tickers', nargs='+', help='Stock ticker(s) to analyze')
    parser.add_argument('--quick', '-q', action='store_true', 
                       help='Quick recommendation only (no LLM analysis)')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare multiple stocks')
    parser.add_argument('--screen', '-s', action='store_true',
                       help='Screen portfolio')
    args = parser.parse_args()
    
    agent = GeneralAnalystAgent()
    
    if args.screen:
        print(agent.screen_portfolio(args.tickers))
    elif args.compare and len(args.tickers) > 1:
        print(agent.compare_stocks(args.tickers))
    elif args.quick:
        for ticker in args.tickers:
            rec = agent.get_quick_recommendation(ticker)
            print(f"\n{ticker}: {rec['signal']} (Score: {rec['scores']['overall']:.1f}, Confidence: {rec['confidence']}%)")
            print(f"  Targets: ${rec['price_targets']['low']:.2f} / ${rec['price_targets']['mid']:.2f} / ${rec['price_targets']['high']:.2f}")
            print(f"  Options: {rec['options']['strategy']}")
    else:
        for ticker in args.tickers:
            print(agent.analyze(ticker))
            print("\n")