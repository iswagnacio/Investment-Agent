"""
Deterministic Scoring
=====================
Signal taxonomy and the ScoreCalculator that turns raw indicator/fundamental/
sentiment data into 0-100 scores. Pure functions — no LLM, no network.
Moved out of the lead agent in ROADMAP.md Phase 4.
"""

from enum import Enum
from typing import Any, Dict, Tuple

from investment import config


# =============================================================================
# Data Models
# =============================================================================

class Signal(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


# The persisted result of an analysis lives in contracts.AnalysisRecord;
# rendering lives in reporting.render_report. (Phase 1: structure first,
# prose second — and no price targets / options / P&L.)


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
    def calculate_sentiment_score(news_data: Dict[str, Any], 
                                   structured: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate sentiment score (0-100) from news/sentiment data.
        
        UPDATED: Incorporates LLM-derived structured signals when available.
        This ensures the 20% sentiment weight is meaningful even without
        Alpha Vantage pre-computed scores.
        
        Priority:
        1. LLM-derived tone (from sentiment_agent) — most reliable
        2. Alpha Vantage pre-computed scores — numeric but limited availability  
        3. Article count/recency heuristics — always available fallback
        """
        scores = {}
        
        articles = news_data.get('company_news', {}).get('articles', [])
        
        if not articles and not structured:
            return 50.0, {'news_sentiment': 50, 'news_volume': 50, 'recency': 50, 'llm_tone': 50}
        
        # =====================================================================
        # LLM-derived tone score (highest priority when available)
        # =====================================================================
        llm_tone_score = None
        if isinstance(structured, dict):
            llm_analysis = structured.get('llm_analysis', {})
            if llm_analysis:
                # Map overall_tone to score
                tone = llm_analysis.get('overall_tone', {})
                tone_signal = tone.get('signal', 'Neutral').lower()
                tone_confidence = tone.get('confidence', 'Low').lower()
                
                tone_map = {'bullish': 75, 'neutral': 50, 'mixed': 50, 'bearish': 25}
                base_tone = tone_map.get(tone_signal, 50)
                
                # Adjust by confidence
                confidence_multiplier = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
                conf = confidence_multiplier.get(tone_confidence, 0.5)
                
                # Move away from 50 based on confidence
                llm_tone_score = 50 + (base_tone - 50) * conf
                
                # Adjust by tone direction
                direction = llm_analysis.get('tone_direction', {}).get('signal', 'Stable').lower()
                if direction == 'improving':
                    llm_tone_score = min(100, llm_tone_score + 8)
                elif direction == 'deteriorating':
                    llm_tone_score = max(0, llm_tone_score - 8)
                
                # Controversy penalty
                controversy = llm_analysis.get('controversy_flag', {})
                if controversy.get('detected', False):
                    llm_tone_score = max(0, llm_tone_score - 12)
                
                # Catalyst bonus/penalty
                catalyst = llm_analysis.get('catalyst_detected', {})
                if catalyst.get('detected', False):
                    impact = catalyst.get('expected_impact', '').lower()
                    if 'positive' in impact or 'bullish' in impact:
                        llm_tone_score = min(100, llm_tone_score + 5)
                    elif 'negative' in impact or 'bearish' in impact:
                        llm_tone_score = max(0, llm_tone_score - 5)
        
        scores['llm_tone'] = max(0, min(100, llm_tone_score)) if llm_tone_score is not None else 50
        
        # =====================================================================
        # Pre-computed sentiment scores (Alpha Vantage, when available)
        # =====================================================================
        sentiment_values = [a.sentiment_hint for a in articles if a.sentiment_hint is not None]
        
        if sentiment_values:
            avg_sentiment = sum(sentiment_values) / len(sentiment_values)
            news_sentiment = (avg_sentiment + 1) * 50  # Map [-1, 1] to [0, 100]
        else:
            news_sentiment = 50
        
        scores['news_sentiment'] = max(0, min(100, news_sentiment))
        
        # =====================================================================
        # Volume and recency (heuristic, always available)
        # =====================================================================
        num_articles = len(articles)
        if num_articles > 20:
            volume_score = 70
        elif num_articles > 10:
            volume_score = 60
        elif num_articles > 5:
            volume_score = 50
        else:
            volume_score = 40
        scores['news_volume'] = volume_score
        
        recent_count = 0
        for a in articles:
            try:
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
        
        # =====================================================================
        # Weighted total — shift weights based on available data
        # =====================================================================
        if llm_tone_score is not None and sentiment_values:
            # Best case: LLM + pre-computed + heuristics
            weights = {'llm_tone': 0.45, 'news_sentiment': 0.25, 'news_volume': 0.15, 'recency': 0.15}
        elif llm_tone_score is not None:
            # LLM available but no pre-computed scores (most common)
            weights = {'llm_tone': 0.55, 'news_sentiment': 0.05, 'news_volume': 0.20, 'recency': 0.20}
        elif sentiment_values:
            # Pre-computed available but no LLM (sentiment_agent not loaded)
            weights = {'llm_tone': 0.05, 'news_sentiment': 0.55, 'news_volume': 0.20, 'recency': 0.20}
        else:
            # Neither available — heuristic only
            weights = {'llm_tone': 0.10, 'news_sentiment': 0.10, 'news_volume': 0.40, 'recency': 0.40}
        
        total_score = sum(scores[k] * weights[k] for k in weights)
        
        return round(total_score, 1), scores
    
    @staticmethod
    def calculate_overall_score(technical: float, fundamental: float,
                                sentiment: float,
                                weights: Dict[str, float] = None) -> float:
        weights = weights or config.BLEND_WEIGHTS
        overall = (technical * weights['technical'] +
                  fundamental * weights['fundamental'] +
                  sentiment * weights['sentiment'])
        return round(overall, 1)
    
    @staticmethod
    def score_to_signal(score: float) -> Signal:
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
    

