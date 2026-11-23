"""
General Analyst Agent Module
=============================
Comprehensive investment analysis combining:
- Technical Analysis (price patterns, momentum, trend indicators)
- Fundamental Analysis (valuation, quality, growth metrics)
- Sentiment Analysis (news sentiment, market perception)

UPDATED: Unified structured pipeline for both technical and fundamental agents
- Both specialist agents provide structured JSON signals via for_synthesis
- Lead agent self-assesses whether EITHER domain needs detailed analysis
- Symmetric handling: _prepare_technical_summary mirrors _prepare_fundamental_summary
- Validated fetch with retry for both detailed analyses

Provides:
- Holistic company assessment (qualitative judgment, no trade construction)
- Signal + conviction with an explicit "what would change my mind"
- Risk assessment and confidence levels
- A persisted AnalysisRecord per run (data/runs/<date>/<ticker>.json)

Usage:
    from investment.agents.lead import GeneralAnalystAgent
    
    agent = GeneralAnalystAgent()
    analysis = agent.analyze("AAPL")
    
    # Get specific recommendation
    rec = agent.get_recommendation("TSLA")
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

from investment import config
from investment.contracts import AnalysisRecord
from investment.registry import (
    Registry,
    cap_post_blend,
    effective_custom_weight,
    load_registry,
)
from investment.reporting import render_report
from investment.scoring import ScoreCalculator, Signal

# Import analysis modules
from investment.calculators.technical import TechnicalCalculator
from investment.calculators.fundamental import FundamentalCalculator
from investment.calculators.sentiment import SentimentDataFetcher

# Import specialized agents for LLM-powered insights
try:
    from investment.agents.technical import TechnicalAnalyzerAgent
    HAS_TECHNICAL_AGENT = True
except ImportError:
    HAS_TECHNICAL_AGENT = False

try:
    from investment.agents.fundamental import FundamentalAnalystAgent
    HAS_FUNDAMENTAL_AGENT = True
except ImportError:
    HAS_FUNDAMENTAL_AGENT = False

try:
    from investment.agents.sentiment import SentimentAnalystAgent
    HAS_SENTIMENT_AGENT = True
except ImportError:
    HAS_SENTIMENT_AGENT = False

try:
    from investment.agents.perspective import PerspectiveAnalystAgent, PerspectiveNotCachedError
    HAS_PERSPECTIVE_AGENT = True
except ImportError:
    HAS_PERSPECTIVE_AGENT = False

load_dotenv()


# =============================================================================
# Ticker Validation Helper
# =============================================================================

def validate_ticker_in_text(text: str, expected_ticker: str, min_mentions: int = 3) -> bool:
    """
    Validate that the LLM output is actually about the expected ticker.
    Returns True if the ticker appears at least min_mentions times.
    """
    if not text or not expected_ticker:
        return False
    
    # Count mentions of the ticker (case-insensitive, word boundary)
    pattern = r'\b' + re.escape(expected_ticker.upper()) + r'\b'
    mentions = len(re.findall(pattern, text.upper()))
    
    return mentions >= min_mentions


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract JSON object from LLM response text.
    Handles cases where JSON is wrapped in markdown code blocks or has a header.
    """
    if not text:
        return None
    
    # Try to find JSON block in markdown code fence
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find a raw JSON object (first { to last })
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
    
    return None


# =============================================================================
# General Analyst Agent
# =============================================================================

class GeneralAnalystAgent:
    """
    Comprehensive investment analyst combining technical, fundamental,
    and sentiment analysis with LLM-powered insights.
    
    UPDATED: Unified structured pipeline
    - Both technical and fundamental agents provide structured JSON via for_synthesis
    - Assessment stage evaluates BOTH domains for detailed analysis needs
    - Symmetric _prepare_*_summary methods
    - Validated fetch with retry for both detailed analyses
    """
    
    def __init__(self, model: str = config.STRONG_MODEL, use_specialist_agents: bool = True, use_perspective: bool = False):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.model = model
        self.technical = TechnicalCalculator()
        self.fundamental = FundamentalCalculator()
        self.sentiment = SentimentDataFetcher()
        self.scorer = ScoreCalculator()

        # Assessment registry (Phase 3): core weights + custom agents.
        # A broken registry.toml must not take the pipeline down.
        try:
            self.registry = load_registry()
        except Exception as e:
            print(f"Warning: invalid registry.toml ({e}); using defaults")
            self.registry = Registry()
        
        self.use_specialist_agents = use_specialist_agents
        self.technical_agent = None
        self.fundamental_agent = None
        self.sentiment_agent = None
        self.use_perspective = use_perspective
        self.perspective_agent = None

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
            
            if HAS_SENTIMENT_AGENT:
                try:
                    self.sentiment_agent = SentimentAnalystAgent(model=model)
                except Exception as e:
                    print(f"Warning: Could not initialize SentimentAnalystAgent: {e}")

        if use_perspective and HAS_PERSPECTIVE_AGENT:
            try:
                self.perspective_agent = PerspectiveAnalystAgent(model=model)
            except Exception as e:
                print(f"Warning: Could not initialize PerspectiveAnalystAgent: {e}")

        self.llm = ChatAnthropic(model_name=model, temperature=0, max_tokens=4096)
    
    def gather_data(self, ticker: str, perspective: str = None) -> Dict[str, Any]:
        """
        Gather all data for analysis.
        
        UPDATED: Both technical and fundamental agents use for_synthesis mode.
        Detailed analysis for either is deferred to the assessment stage.
        """
        ticker = ticker.upper()
        data: Dict[str, Any] = {'ticker': ticker}
        
        # =====================================================================
        # Technical data (raw indicators for scoring)
        # =====================================================================
        try:
            data['technical'] = self.technical.get_all_indicators(ticker)
            data['stock_info'] = self.technical.get_stock_info(ticker)
        except Exception as e:
            data['technical'] = {}
            data['stock_info'] = {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown'}
            data['technical_error'] = str(e)
        
        # =====================================================================
        # Technical agent structured signals (NEW: mirrors fundamental pattern)
        # =====================================================================
        if self.use_specialist_agents and self.technical_agent:
            try:
                raw_response = self.technical_agent.analyze(
                    ticker, analysis_type="for_synthesis"
                )
                
                # Validate the response is about the right ticker
                if not validate_ticker_in_text(raw_response, ticker, min_mentions=2):
                    print(f"⚠️  WARNING: technical for_synthesis response may be for wrong ticker!")
                    raw_response = self.technical_agent.analyze(
                        ticker, analysis_type="for_synthesis"
                    )
                
                # Parse JSON from response
                parsed = extract_json_from_text(raw_response)
                if parsed and isinstance(parsed, dict):
                    data['technical_structured'] = parsed
                    print(f"   ✓ Parsed structured technical signals for {ticker}")
                else:
                    data['technical_structured'] = raw_response
                    print(f"   ⚠ Could not parse JSON from technical for_synthesis, using raw text")
                    
            except Exception as e:
                data['technical_structured'] = None
                data['technical_structured_error'] = str(e)
        else:
            data['technical_structured'] = None
        
        # Detailed technical deferred to assessment stage
        data['technical_detailed'] = None
        
        # =====================================================================
        # Fundamental raw data (for scoring)
        # =====================================================================
        try:
            data['fundamental'] = self.fundamental.get_all_fundamentals(ticker)
        except Exception as e:
            data['fundamental'] = {}
            data['fundamental_error'] = str(e)
        
        # =====================================================================
        # Fundamental agent structured signals
        # =====================================================================
        if self.use_specialist_agents and self.fundamental_agent:
            try:
                raw_response = self.fundamental_agent.analyze(
                    ticker, analysis_type="for_synthesis"
                )
                
                if not validate_ticker_in_text(raw_response, ticker, min_mentions=2):
                    print(f"⚠️  WARNING: fundamental for_synthesis response may be for wrong ticker!")
                    raw_response = self.fundamental_agent.analyze(
                        ticker, analysis_type="for_synthesis"
                    )
                
                parsed = extract_json_from_text(raw_response)
                if parsed and isinstance(parsed, dict):
                    data['fundamental_structured'] = parsed
                    print(f"   ✓ Parsed structured fundamental signals for {ticker}")
                else:
                    data['fundamental_structured'] = raw_response
                    print(f"   ⚠ Could not parse JSON from fundamental for_synthesis, using raw text")
                    
            except Exception as e:
                data['fundamental_structured'] = None
                data['fundamental_structured_error'] = str(e)
        else:
            data['fundamental_structured'] = None
        
        # Detailed fundamental deferred to assessment stage
        data['fundamental_detailed'] = None
        
        # =====================================================================
        # Sentiment data (raw, for scoring)
        # =====================================================================
        try:
            company_name = data.get('stock_info', {}).get('name', ticker)
            industry = data.get('stock_info', {}).get('industry', '')
            
            if self.use_specialist_agents and self.sentiment_agent:
                # Use sentiment agent's get_raw_data to avoid double-fetching
                data['sentiment'] = self.sentiment_agent.get_raw_data(
                    ticker, company_name, industry
                )
            else:
                data['sentiment'] = self.sentiment.fetch_all(
                    ticker, company_name, industry, 
                    max_company_news=25, max_industry_news=10
                )
        except Exception as e:
            data['sentiment'] = {}
            data['sentiment_error'] = str(e)
        
        # =====================================================================
        # Sentiment agent structured signals
        # =====================================================================
        if self.use_specialist_agents and self.sentiment_agent:
            try:
                raw_response = self.sentiment_agent.analyze(
                    ticker, 
                    company_name=data.get('stock_info', {}).get('name'),
                    industry=data.get('stock_info', {}).get('industry')
                )
                
                parsed = extract_json_from_text(raw_response)
                if parsed and isinstance(parsed, dict):
                    data['sentiment_structured'] = parsed
                    print(f"   ✓ Parsed structured sentiment signals for {ticker}")
                else:
                    data['sentiment_structured'] = raw_response
                    print(f"   ⚠ Could not parse JSON from sentiment analysis, using raw text")
                    
            except Exception as e:
                data['sentiment_structured'] = None
                data['sentiment_structured_error'] = str(e)
        else:
            data['sentiment_structured'] = None
        
        # --- Perspective: reads offline-distilled cache; never distills here ---
        data['perspective'] = None
        if self.perspective_agent and perspective:
            try:
                pres = self.perspective_agent.analyze(
                    ticker=ticker,
                    investor=perspective,
                    fundamental_data=data.get('fundamental'),
                    technical_data=data.get('technical'),
                    sentiment_data=data.get('sentiment'),
                    analysis_type="for_synthesis",
                )
                data['perspective'] = pres
                print(f"   ✓ Perspective applied: {pres['investor']} "
                      f"({pres['verdict']}, score {pres['score']})")
            except PerspectiveNotCachedError as e:
                print(f"   ⚠ {e}")
            except Exception as e:
                print(f"   ⚠ Perspective analysis failed: {e}")

        # --- Custom assessments (Phase 3): registry-declared, workspace-backed --
        # Each is a methodology artifact in data/workspaces/<key>/ applied by the
        # same machinery as investor perspectives. Permissions are enforced here:
        # the agent only receives the data sources its manifest lists.
        data['custom_assessments'] = {}
        for spec in (self.registry.enabled_custom() if self.registry else []):
            if not HAS_PERSPECTIVE_AGENT:
                print(f"   ⚠ {spec.key}: perspective machinery unavailable — skipped")
                break
            if spec.permissions.max_llm_calls < 1:
                print(f"   ⚠ {spec.key}: per-run LLM budget is 0 — skipped")
                continue
            allowed = spec.permissions.data_sources
            try:
                applier = PerspectiveAnalystAgent(
                    model=self.model, cache_dir=str(config.WORKSPACES_DIR))
                res = applier.analyze(
                    ticker=ticker,
                    investor=spec.key,
                    fundamental_data=data.get('fundamental') if 'fundamental' in allowed else None,
                    technical_data=data.get('technical') if 'technical' in allowed else None,
                    sentiment_data=data.get('sentiment') if 'sentiment' in allowed else None,
                    analysis_type="for_synthesis",
                )
                data['custom_assessments'][spec.key] = res
                print(f"   ✓ Custom assessment {spec.key}: "
                      f"{res['verdict']} (score {res['score']})")
            except PerspectiveNotCachedError:
                print(f"   ⚠ {spec.key}: no artifact at "
                      f"{spec.workspace_dir}/framework.json — author it first; skipped")
            except Exception as e:
                print(f"   ⚠ {spec.key}: assessment failed: {e}")

        return data
    
    def calculate_scores(self, data: Dict[str, Any], perspective_weight: float = None) -> Dict[str, Any]:
        """Calculate all scores from gathered data."""
        scores: Dict[str, Any] = {}
        
        if data.get('technical'):
            tech_score, tech_breakdown = self.scorer.calculate_technical_score(data['technical'])
            scores['technical'] = {'score': tech_score, 'breakdown': tech_breakdown}
        else:
            scores['technical'] = {'score': 50.0, 'breakdown': {}}
        
        if data.get('fundamental'):
            fund_score, fund_breakdown = self.scorer.calculate_fundamental_score(data['fundamental'])
            scores['fundamental'] = {'score': fund_score, 'breakdown': fund_breakdown}
        else:
            scores['fundamental'] = {'score': 50.0, 'breakdown': {}}
        
        if data.get('sentiment'):
            sent_score, sent_breakdown = self.scorer.calculate_sentiment_score(
                data['sentiment'], data.get('sentiment_structured')
            )
            scores['sentiment'] = {'score': sent_score, 'breakdown': sent_breakdown}
        else:
            scores['sentiment'] = {'score': 50.0, 'breakdown': {}}
        
        # Core blend weights come from the registry (Phase 3); registry-less
        # callers (tests, bare instances) fall back to config.BLEND_WEIGHTS.
        reg = getattr(self, 'registry', None)
        base_w = dict(reg.core_weights) if reg else dict(config.BLEND_WEIGHTS)

        base_overall = self.scorer.calculate_overall_score(
            scores['technical']['score'],
            scores['fundamental']['score'],
            scores['sentiment']['score'],
            weights=base_w,
        )

        # Post-blend WITHOUT touching ScoreCalculator: perspective + custom
        # assessments each contribute (score, weight); provably identical to
        # re-scaling the core weights by (1 - total_pw), because base_overall
        # is already their weighted average:
        #   (1 - Σw)(core blend) + Σ(w_i · s_i)
        post_entries = []  # (name, score, weight)

        persp = data.get('perspective')
        if persp and perspective_weight:
            pw = self._effective_perspective_weight(perspective_weight, persp.get('metadata', {}))
            scores['perspective'] = {'score': persp['score'], 'breakdown': persp.get('metadata', {})}
            post_entries.append(('perspective', persp['score'], pw))
        else:
            scores['perspective'] = None

        for key, res in (data.get('custom_assessments') or {}).items():
            spec = reg.custom.get(key) if reg else None
            if spec is None:
                continue
            w = effective_custom_weight(
                spec,
                artifact_confidence=res.get('metadata', {}).get('distillation_confidence'),
            )
            post_entries.append((key, res['score'], w))

        # The core domains always keep the majority (registry.MAX_TOTAL_POST_WEIGHT).
        post_entries = cap_post_blend(post_entries)
        total_pw = sum(w for _, _, w in post_entries)

        scores['custom'] = {name: {'score': s, 'weight': w}
                            for name, s, w in post_entries if name != 'perspective'}

        if total_pw > 0:
            blended = base_overall * (1 - total_pw) + sum(s * w for _, s, w in post_entries)
            scores['overall'] = round(blended, 1)
            weights = {k: round(v * (1 - total_pw), 3) for k, v in base_w.items()}
            for name, _, w in post_entries:
                weights[name] = w
            weights.setdefault('perspective', 0.0)
            scores['weights'] = weights
        else:
            scores['overall'] = base_overall
            scores['weights'] = {**base_w, 'perspective': 0.0}

        scores['signal'] = self.scorer.score_to_signal(scores['overall'])
        
        tech_fund_diff = abs(scores['technical']['score'] - scores['fundamental']['score'])
        if tech_fund_diff < 10:
            confidence = 85
        elif tech_fund_diff < 20:
            confidence = 70
        else:
            confidence = 55
        
        if not data.get('technical'):
            confidence -= 15
        if not data.get('fundamental'):
            confidence -= 20
        if not data.get('sentiment'):
            confidence -= 5
        
        if persp and persp.get('metadata', {}).get('distillation_confidence') == 'Low':
            confidence -= 5
        
        scores['confidence'] = max(30, min(95, confidence))
        
        return scores
    
    def _effective_perspective_weight(self, requested: float, meta: Dict) -> float:
        """Clamp to cap, then shrink when the distillation is weak (corrected plan, DP-6)."""
        pw = max(0.0, min(config.MAX_PERSPECTIVE_WEIGHT, requested))
        factor = {'High': 1.0, 'Medium': 0.7, 'Low': 0.4}.get(
            meta.get('distillation_confidence', 'Medium'), 0.7)
        return round(pw * factor, 3)

    @staticmethod
    def _current_price(data: Dict[str, Any]) -> Optional[float]:
        """Best-effort spot price from already-gathered data (context only)."""
        price = data.get('technical', {}).get('price', {}).get('current', 0)
        if not price:
            price = data.get('fundamental', {}).get('valuation_ratios', {}).get('price', 0)
        return round(price, 2) if price else None


    # =========================================================================
    # Validated Fetch Helpers (with retry for both domains)
    # =========================================================================
    
    def _fetch_detailed_with_validation(self, domain: str, ticker: str, 
                                         max_retries: int = 2) -> Optional[str]:
        """
        Fetch detailed analysis with ticker validation and retry.
        Works for both 'technical' and 'fundamental' domains.
        """
        if domain == "technical":
            agent = self.technical_agent
            analysis_type = "comprehensive"
        elif domain == "fundamental":
            agent = self.fundamental_agent
            analysis_type = "comprehensive"
        else:
            return None
        
        if not agent:
            return None
        
        for attempt in range(max_retries):
            try:
                print(f"   Fetching detailed {domain} analysis for: {ticker} (attempt {attempt + 1})")
                detailed = agent.analyze(ticker, analysis_type=analysis_type)
                
                # Validate the response is about the correct ticker
                if validate_ticker_in_text(detailed, ticker, min_mentions=5):
                    print(f"   ✓ Detailed {domain} analysis validated for {ticker}")
                    return detailed
                else:
                    wrong_match = re.search(r'ANALYZING:\s*(\w+)', detailed)
                    wrong_ticker = wrong_match.group(1) if wrong_match else "UNKNOWN"
                    print(f"   ⚠️  Attempt {attempt + 1}: {domain} analysis is for {wrong_ticker}, not {ticker}!")
                    
                    if attempt < max_retries - 1:
                        print(f"   Retrying...")
                    else:
                        print(f"   ✗ Failed after {max_retries} attempts. Discarding detailed {domain} analysis.")
                        return None
                        
            except Exception as e:
                print(f"   ✗ Error on attempt {attempt + 1}: {e}")
                if attempt >= max_retries - 1:
                    return None
        
        return None
    
    # =========================================================================
    # Summary Preparation (Symmetric for both domains)
    # =========================================================================
    
    def _prepare_technical_summary(self, data: Dict, detailed: str = None) -> str:
        """
        Prepare technical summary for LLM.
        
        UPDATED: Now mirrors _prepare_fundamental_summary pattern.
        Uses structured signals from technical agent when available.
        Falls back to raw indicator formatting.
        """
        ticker = data.get('ticker', 'UNKNOWN')
        tech_structured = data.get('technical_structured')
        
        # Check if we got a properly parsed dict
        if isinstance(tech_structured, dict):
            llm_analysis = tech_structured.get('llm_analysis', {})
            selected = tech_structured.get('selected_indicators', [])
            selection_reason = tech_structured.get('selection_reasoning', '')
            
            if llm_analysis:
                summary = f"""=== TECHNICAL ANALYST (STRUCTURED) FOR {ticker} ===
Selected Indicators: {', '.join(selected)}
Selection Rationale: {selection_reason}

{self._format_technical_structured_signals(llm_analysis)}"""
            else:
                # Dict but no llm_analysis key - fall back to raw metrics
                summary = self._format_basic_technical_metrics(data, ticker)
        elif isinstance(tech_structured, str) and len(tech_structured) > 50:
            # Got raw string - use it directly
            summary = f"""=== TECHNICAL ANALYST SIGNALS FOR {ticker} ===

{tech_structured[:1500]}"""
        else:
            # No structured data - fall back to raw indicator formatting
            summary = self._format_basic_technical_metrics(data, ticker)
        
        # Add detailed if provided
        if detailed:
            summary += f"""

{'='*60}
=== DETAILED TECHNICAL ANALYSIS FOR {ticker} ===
{'='*60}

{detailed[:2000]}

[Full detailed analysis available for deep context]"""
        
        return summary
    
    def _prepare_perspective_summary(self, data: Dict) -> str:
        """Format the perspective signal for the synthesis prompt. Empty string if none."""
        persp = data.get('perspective')
        if not persp:
            return ""
        m = persp.get('metadata', {})
        risks = persp.get('key_risks', [])
        risk_str = ("\nKey risks flagged: " + "; ".join(risks)) if risks else ""
        return (
            f"{persp['investor']} — verdict: {persp['verdict']}, "
            f"score: {persp['score']}/100 "
            f"(analysis confidence: {m.get('analysis_confidence', '?')}, "
            f"distillation confidence: {m.get('distillation_confidence', '?')})\n"
            f"{persp['narrative']}{risk_str}\n"
            f"NOTE: models the investor's stated heuristics applied to current data — "
            f"not the investor's actual judgment. Weight accordingly."
        )

    def _format_technical_structured_signals(self, llm_analysis: Dict) -> str:
        """Compact formatting of structured technical signals (mirrors fundamental)."""
        trend = llm_analysis.get('trend_verdict', {})
        momentum = llm_analysis.get('momentum_verdict', {})
        volatility = llm_analysis.get('volatility_verdict', {})
        volume = llm_analysis.get('volume_verdict', {})
        levels = llm_analysis.get('key_levels', {})
        outlook = llm_analysis.get('near_term_outlook', {})
        
        return f"""TREND: {trend.get('signal', 'N/A')} (Strength: {trend.get('strength', 'N/A')}, Confidence: {trend.get('confidence', 'N/A')})
  -> {trend.get('reasoning', 'N/A')}

MOMENTUM: {momentum.get('signal', 'N/A')} (Confidence: {momentum.get('confidence', 'N/A')})
  -> {momentum.get('reasoning', 'N/A')}

VOLATILITY: {volatility.get('signal', 'N/A')}
  -> Bollinger Position: {volatility.get('bollinger_position', 'N/A')}
  -> {volatility.get('implication', 'N/A')}

VOLUME: {volume.get('signal', 'N/A')}
  -> {volume.get('reasoning', 'N/A')}

KEY LEVELS:
  Support:    ${levels.get('support_1', 0):.2f} / ${levels.get('support_2', 0):.2f}
  Resistance: ${levels.get('resistance_1', 0):.2f} / ${levels.get('resistance_2', 0):.2f}
  Derivation: {levels.get('derivation', 'N/A')}

PATTERN: {llm_analysis.get('pattern_detected', 'N/A')}

NEAR-TERM OUTLOOK: {outlook.get('bias', 'N/A')} ({outlook.get('timeframe', 'N/A')})
  Catalyst Level: {outlook.get('catalyst_level', 'N/A')}

RISK FLAGS: {', '.join(llm_analysis.get('risk_flags', ['None identified']))}"""
    
    def _format_basic_technical_metrics(self, data: Dict, ticker: str) -> str:
        """Format basic technical metrics as fallback."""
        tech = data.get('technical', {})
        price = tech.get('price', {})
        ma = tech.get('moving_averages', {})
        momentum = tech.get('momentum', {})
        trend = tech.get('trend', {})
        vol = tech.get('volatility', {})
        volume = tech.get('volume', {})
        macd = trend.get('macd', {})
        
        current = price.get('current', 0)
        sma_20 = ma.get('sma_20', 0)
        sma_50 = ma.get('sma_50', 0)
        
        return f"""=== BASIC TECHNICAL METRICS FOR {ticker} ===
Price: ${current:.2f} | 1D: {price.get('change_1d', 0):+.2f}% | 5D: {price.get('change_5d', 0):+.2f}% | 20D: {price.get('change_20d', 0):+.2f}%
Moving Avgs: SMA(20): ${sma_20:.2f} | SMA(50): ${sma_50:.2f} | Price {'Above' if current > sma_50 else 'Below'} SMA(50)
Momentum: RSI(14): {momentum.get('rsi_14', 50):.1f} | {'Oversold' if momentum.get('rsi_14', 50) < 30 else 'Overbought' if momentum.get('rsi_14', 50) > 70 else 'Neutral'}
Trend: MACD {'Bullish' if macd.get('histogram', 0) > 0 else 'Bearish'} | ADX: {trend.get('adx', 0):.1f}
Volume: {volume.get('ratio', 1):.2f}x average | Trend: {volume.get('trend', 0):+.2%}
Volatility: {vol.get('historical_vol', 0):.2%} annualized"""
    
    def _prepare_fundamental_summary(self, data: Dict, detailed: str = None) -> str:
        """
        Prepare fundamental summary for LLM.
        Handles both dict (parsed JSON) and string (raw text) formats.
        """
        ticker = data.get('ticker', 'UNKNOWN')
        fund_structured = data.get('fundamental_structured')
        
        if isinstance(fund_structured, dict):
            llm_analysis = fund_structured.get('llm_analysis', {})
            
            if llm_analysis:
                summary = f"""=== FUNDAMENTAL ANALYST (STRUCTURED) FOR {ticker} ===

{self._format_fundamental_structured_signals(llm_analysis)}"""
            else:
                summary = self._format_basic_fundamental_metrics(data, ticker)
        elif isinstance(fund_structured, str) and len(fund_structured) > 50:
            summary = f"""=== FUNDAMENTAL ANALYST SIGNALS FOR {ticker} ===

{fund_structured[:1500]}"""
        else:
            summary = self._format_basic_fundamental_metrics(data, ticker)
        
        if detailed:
            summary += f"""

{'='*60}
=== DETAILED FUNDAMENTAL ANALYSIS FOR {ticker} ===
{'='*60}

{detailed[:2000]}

[Full detailed analysis available for deep context]"""
        
        return summary
    
    def _format_basic_fundamental_metrics(self, data: Dict, ticker: str) -> str:
        """Format basic fundamental metrics as fallback."""
        fund = data.get('fundamental', {})
        val = fund.get('valuation_ratios', {})
        prof = fund.get('profitability_ratios', {})
        qual = fund.get('quality_scores', {})
        growth = fund.get('growth_metrics', {})
        lev = fund.get('leverage_ratios', {})
        liq = fund.get('liquidity_ratios', {})
        
        return f"""=== BASIC FUNDAMENTAL METRICS FOR {ticker} ===
Valuation: P/E: {val.get('pe_ratio', 0):.1f} | PEG: {val.get('peg_ratio', 0):.2f} | P/B: {val.get('price_to_book', 0):.2f} | EV/EBITDA: {val.get('ev_to_ebitda', 0):.1f}
Profitability: ROE: {prof.get('return_on_equity', 0):.1f}% | Net Margin: {prof.get('net_profit_margin', 0):.1f}% | ROIC: {prof.get('return_on_invested_capital', 0):.1f}%
Growth: Rev YoY: {growth.get('revenue_growth_yoy', 0):.1f}% | Earnings YoY: {growth.get('earnings_growth_yoy', 0):.1f}%
Quality: Altman Z: {qual.get('altman_z_score', 0):.2f} ({qual.get('altman_z_interpretation', 'N/A')}) | Piotroski F: {qual.get('piotroski_f_score', 0)}/9
Health: D/E: {lev.get('debt_to_equity', 0):.2f} | Current Ratio: {liq.get('current_ratio', 0):.2f}"""
    
    def _format_fundamental_structured_signals(self, llm_analysis: Dict) -> str:
        """Compact formatting of structured fundamental signals."""
        val = llm_analysis.get('valuation_verdict', {})
        qual = llm_analysis.get('quality_verdict', {})
        growth = llm_analysis.get('growth_verdict', {})
        health = llm_analysis.get('financial_health', {})
        
        return f"""VALUATION: {val.get('signal', 'N/A')} (Confidence: {val.get('confidence', 'N/A')})
  -> {val.get('reasoning', 'N/A')}

QUALITY: {qual.get('signal', 'N/A')} (Confidence: {qual.get('confidence', 'N/A')})
  -> Z: {qual.get('altman_z', 'N/A')} | F: {qual.get('piotroski_f', 'N/A')}
  -> {qual.get('reasoning', 'N/A')}

GROWTH: {growth.get('signal', 'N/A')} (Confidence: {growth.get('confidence', 'N/A')})
  -> {growth.get('revenue_trajectory', 'N/A')}
  -> {growth.get('reasoning', 'N/A')}

HEALTH: {health.get('signal', 'N/A')}
  -> Leverage: {health.get('leverage_status', 'N/A')} | Liquidity: {health.get('liquidity_status', 'N/A')}

STRENGTHS: {', '.join(llm_analysis.get('key_strengths', [])[:3])}
RISKS: {', '.join(llm_analysis.get('key_risks', [])[:3])}

THESIS:
Bull: {llm_analysis.get('investment_thesis', {}).get('bull_case', 'N/A')}
Bear: {llm_analysis.get('investment_thesis', {}).get('bear_case', 'N/A')}"""
    
    def _prepare_sentiment_summary(self, data: Dict) -> str:
        """
        Prepare sentiment summary for LLM.
        Mirrors _prepare_technical_summary and _prepare_fundamental_summary pattern.
        Uses structured signals from sentiment agent when available,
        falls back to raw headline formatting.
        """
        ticker = data.get('ticker', 'UNKNOWN')
        sent_structured = data.get('sentiment_structured')
        
        # Check if we got a properly parsed dict from sentiment agent
        if isinstance(sent_structured, dict):
            llm_analysis = sent_structured.get('llm_analysis', {})
            
            if llm_analysis:
                summary = f"""=== SENTIMENT ANALYST (STRUCTURED) FOR {ticker} ===

{self._format_sentiment_structured_signals(llm_analysis)}"""
            else:
                # Dict but no llm_analysis key - fall back to raw headlines
                summary = self._format_basic_sentiment_data(data, ticker)
        elif isinstance(sent_structured, str) and len(sent_structured) > 50:
            # Got raw string - use it directly
            summary = f"""=== SENTIMENT ANALYST SIGNALS FOR {ticker} ===

{sent_structured[:1500]}"""
        else:
            # No structured data - fall back to raw headline formatting
            summary = self._format_basic_sentiment_data(data, ticker)
        
        return summary
    
    def _format_sentiment_structured_signals(self, llm_analysis: Dict) -> str:
        """Compact formatting of structured sentiment signals (mirrors other domains)."""
        tone = llm_analysis.get('overall_tone', {})
        direction = llm_analysis.get('tone_direction', {})
        concentration = llm_analysis.get('news_concentration', {})
        controversy = llm_analysis.get('controversy_flag', {})
        catalyst = llm_analysis.get('catalyst_detected', {})
        industry_ctx = llm_analysis.get('industry_context', {})
        
        # Format key events
        events = llm_analysis.get('key_events', [])
        events_text = "None identified"
        if events:
            event_lines = []
            for e in events[:5]:
                event_lines.append(
                    f"  - {e.get('event', 'N/A')} "
                    f"[{e.get('impact', 'N/A')}, {e.get('materiality', 'N/A')} materiality]"
                )
            events_text = "\n".join(event_lines)
        
        themes = llm_analysis.get('key_themes', [])
        
        return f"""OVERALL TONE: {tone.get('signal', 'N/A')} (Confidence: {tone.get('confidence', 'N/A')})
  -> {tone.get('reasoning', 'N/A')}

TONE DIRECTION: {direction.get('signal', 'N/A')}
  -> {direction.get('reasoning', 'N/A')}

KEY EVENTS (deduplicated):
{events_text}

KEY THEMES: {', '.join(themes) if themes else 'None identified'}

NEWS CONCENTRATION: {concentration.get('signal', 'N/A')}
  -> {concentration.get('reasoning', 'N/A')}

CONTROVERSY: {'⚠ DETECTED — ' + controversy.get('description', '') if controversy.get('detected') else 'None detected'}

CATALYST: {'DETECTED — ' + catalyst.get('description', '') + ' (Impact: ' + catalyst.get('expected_impact', 'Unknown') + ')' if catalyst.get('detected') else 'None detected'}

INDUSTRY CONTEXT: {industry_ctx.get('signal', 'N/A')}
  -> {industry_ctx.get('reasoning', 'N/A')}

RISK SIGNALS: {', '.join(llm_analysis.get('risk_signals', ['None identified']))}"""
    
    def _format_basic_sentiment_data(self, data: Dict, ticker: str) -> str:
        """Format basic sentiment data as fallback (raw headlines)."""
        sent = data.get('sentiment', {})
        if not sent:
            return f"=== BASIC SENTIMENT DATA FOR {ticker} ===\nNo sentiment data available"
        
        summary = sent.get('summary', {})
        company_news = sent.get('company_news', {})
        articles = company_news.get('articles', [])
        
        headlines = []
        for article in articles[:10]:
            sentiment_str = f" [score: {article.sentiment_hint:.2f}]" if article.sentiment_hint else ""
            headlines.append(f"  - {article.title[:80]}...{sentiment_str}")
        
        return f"""=== BASIC SENTIMENT DATA FOR {ticker} ===
Total Items: {summary.get('total_items', 0)}
Has Pre-computed Scores: {summary.get('has_sentiment_scores', False)}

Recent Headlines ({len(articles)} articles):
{chr(10).join(headlines) if headlines else '  No recent news'}"""
    
    # =========================================================================
    # LLM Analysis Generation (Two-Stage with Unified Assessment)
    # =========================================================================
    
    def generate_llm_analysis(self, data: Dict[str, Any],
                             scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Two-stage synthesis with unified self-assessment for BOTH domains.
        
        Stage 1: LLM evaluates whether it needs detailed analysis for 
                 technical, fundamental, or both.
        Stage 2: Conditional fetch + final synthesis.
        """
        
        ticker = data['ticker']
        stock_info = data.get('stock_info', {})
        company_name = stock_info.get('name', ticker)
        sector = stock_info.get('sector', 'Unknown')
        industry = stock_info.get('industry', 'Unknown')
        
        # Prepare summaries from structured signals (all three symmetric)
        tech_summary = self._prepare_technical_summary(data)
        fund_summary = self._prepare_fundamental_summary(data)
        sent_summary = self._prepare_sentiment_summary(data)
        
        scores_text = f"""Technical: {scores['technical']['score']:.1f}/100
Fundamental: {scores['fundamental']['score']:.1f}/100
Sentiment: {scores['sentiment']['score']:.1f}/100
Overall: {scores['overall']:.1f}/100"""
        
        current_price = self._current_price(data) or 0
        
        # =====================================================================
        # STAGE 1: UNIFIED ASSESSMENT — Technical AND Fundamental
        # =====================================================================
        
        assessment_prompt = self._build_assessment_prompt(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            industry=industry,
            current_price=current_price,
            technical_summary=tech_summary,
            fundamental_summary=fund_summary,
            sentiment_summary=sent_summary,
            scores_summary=scores_text,
            signal=scores['signal'].value,
            confidence=scores['confidence']
        )
        
        assessment_response = self.llm.invoke(assessment_prompt)
        
        if isinstance(assessment_response.content, str):
            assessment_text = assessment_response.content
        elif isinstance(assessment_response.content, list):
            text_parts = [item for item in assessment_response.content if isinstance(item, str)]
            assessment_text = '\n'.join(text_parts) if text_parts else str(assessment_response.content)
        else:
            assessment_text = str(assessment_response.content)
        
        needs_tech_detailed, tech_reason, needs_fund_detailed, fund_reason = \
            self._parse_unified_assessment(assessment_text)
        
        # =====================================================================
        # CONDITIONAL FETCH: Get detailed for whichever domains need it
        # =====================================================================
        
        tech_detailed = None
        fund_detailed = None
        
        if needs_tech_detailed:
            print(f"\n🔍 {ticker}: LLM requesting detailed TECHNICAL analysis")
            print(f"   Reason: {tech_reason}\n")
            tech_detailed = self._fetch_detailed_with_validation("technical", ticker)
            if tech_detailed:
                tech_summary = self._prepare_technical_summary(data, detailed=tech_detailed)
            else:
                tech_reason = "Requested but validation failed after retries"
        
        if needs_fund_detailed:
            print(f"\n🔍 {ticker}: LLM requesting detailed FUNDAMENTAL analysis")
            print(f"   Reason: {fund_reason}\n")
            fund_detailed = self._fetch_detailed_with_validation("fundamental", ticker)
            if fund_detailed:
                fund_summary = self._prepare_fundamental_summary(data, detailed=fund_detailed)
            else:
                fund_reason = "Requested but validation failed after retries"
        
        # =====================================================================
        # STAGE 2: FINAL SYNTHESIS
        # =====================================================================
        
        # Build detail status text
        detail_lines = []
        if tech_detailed:
            detail_lines.append(f"TECHNICAL: DETAILED ANALYSIS FETCHED — {tech_reason}")
        else:
            detail_lines.append("TECHNICAL: Structured signals sufficient")
        if fund_detailed:
            detail_lines.append(f"FUNDAMENTAL: DETAILED ANALYSIS FETCHED — {fund_reason}")
        else:
            detail_lines.append("FUNDAMENTAL: Structured signals sufficient")
        detail_fetch_info = "\n".join(detail_lines)
        
        additional_context_parts = []
        if tech_detailed:
            additional_context_parts.append(
                f"- How does the detailed technical context clarify {ticker}'s price action and key levels?"
            )
        if fund_detailed:
            additional_context_parts.append(
                f"- How does the detailed fundamental context resolve ambiguities about {ticker}'s valuation/quality?"
            )
        additional_context = "\n".join(additional_context_parts)
        
        synthesis_prompt = self._build_synthesis_prompt(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            industry=industry,
            current_price=current_price,
            technical_summary=tech_summary,
            fundamental_summary=fund_summary,
            sentiment_summary=sent_summary,
            scores_summary=scores_text,
            perspective_summary=self._prepare_perspective_summary(data),
            signal=scores['signal'].value,
            confidence=scores['confidence'],
            detail_fetch_info=detail_fetch_info,
            additional_context=additional_context
        )
        
        synthesis_response = self.llm.invoke(synthesis_prompt)
        
        if isinstance(synthesis_response.content, str):
            analysis_text = synthesis_response.content
        elif isinstance(synthesis_response.content, list):
            text_parts = [item for item in synthesis_response.content if isinstance(item, str)]
            analysis_text = '\n'.join(text_parts) if text_parts else str(synthesis_response.content)
        else:
            analysis_text = str(synthesis_response.content)
        
        # Final validation
        if not validate_ticker_in_text(analysis_text, ticker, min_mentions=3):
            print(f"⚠️  WARNING: Final synthesis may not be about {ticker}!")
            analysis_text = (
                f"[NOTE: This analysis is for {ticker} ({company_name})]\n\n"
                + analysis_text
            )
        
        return {
            'full_analysis': analysis_text,
            'used_detailed_technical': bool(tech_detailed),
            'used_detailed_fundamental': bool(fund_detailed),
            'tech_detail_reason': tech_reason if tech_detailed else None,
            'fund_detail_reason': fund_reason if fund_detailed else None,
            'llm_requested_tech_detailed': needs_tech_detailed,
            'llm_requested_fund_detailed': needs_fund_detailed,
        }
    
    def _parse_unified_assessment(self, text: str) -> Tuple[bool, str, bool, str]:
        """
        Parse the unified assessment response.
        Returns: (needs_tech_detailed, tech_reason, needs_fund_detailed, fund_reason)
        """
        needs_tech = False
        tech_reason = "Structured signals sufficient"
        needs_fund = False
        fund_reason = "Structured signals sufficient"
        
        try:
            # Parse technical decision
            if '<technical_decision>' in text and '</technical_decision>' in text:
                start = text.find('<technical_decision>') + len('<technical_decision>')
                end = text.find('</technical_decision>')
                decision = text[start:end].strip().upper()
                needs_tech = decision == 'YES'
            
            if '<technical_reason>' in text and '</technical_reason>' in text:
                start = text.find('<technical_reason>') + len('<technical_reason>')
                end = text.find('</technical_reason>')
                tech_reason = text[start:end].strip()
            
            # Parse fundamental decision
            if '<fundamental_decision>' in text and '</fundamental_decision>' in text:
                start = text.find('<fundamental_decision>') + len('<fundamental_decision>')
                end = text.find('</fundamental_decision>')
                decision = text[start:end].strip().upper()
                needs_fund = decision == 'YES'
            
            if '<fundamental_reason>' in text and '</fundamental_reason>' in text:
                start = text.find('<fundamental_reason>') + len('<fundamental_reason>')
                end = text.find('</fundamental_reason>')
                fund_reason = text[start:end].strip()
                
        except Exception as e:
            # Fallback: look for keywords
            text_upper = text.upper()
            if 'TECHNICAL' in text_upper and 'YES' in text_upper and 'NEED' in text_upper:
                needs_tech = True
                tech_reason = "Detected from unstructured assessment"
            if 'FUNDAMENTAL' in text_upper and 'YES' in text_upper and 'NEED' in text_upper:
                needs_fund = True
                fund_reason = "Detected from unstructured assessment"
        
        return needs_tech, tech_reason, needs_fund, fund_reason
    
    # =========================================================================
    # Prompt Builders (plain f-strings to avoid template variable conflicts)
    # =========================================================================
    
    def _build_assessment_prompt(self, *, ticker, company_name, sector, industry,
                                  current_price, technical_summary, fundamental_summary,
                                  sentiment_summary, scores_summary, signal, confidence) -> str:
        """
        Build the unified assessment prompt.
        UPDATED: Now evaluates BOTH technical and fundamental detail needs.
        """
        
        return f"""You are the LEAD INVESTMENT ANALYST preparing to synthesize a recommendation for {ticker}.

You have received STRUCTURED SIGNALS from both specialist analysts. Your job is to assess whether you need DETAILED analysis from either or both domains before creating the final synthesis.

=== STOCK INFORMATION ===
Ticker: {ticker}
Company: {company_name}
Sector: {sector} | Industry: {industry}
Current Price: ${current_price}

=== SPECIALIST INPUTS (STRUCTURED SIGNALS) ===

TECHNICAL ANALYST:
{technical_summary}

FUNDAMENTAL ANALYST:
{fundamental_summary}

SENTIMENT ANALYST:
{sentiment_summary}

=== QUANTITATIVE SCORES ===
{scores_summary}
Signal: {signal}
Confidence: {confidence}%

=== YOUR ASSESSMENT TASK ===

Evaluate SEPARATELY whether you need detailed analysis for each domain.

**TECHNICAL — Do you need detailed technical analysis?**

NEED DETAILED TECHNICAL if:
- Technical signals conflict internally (e.g., trend bullish but momentum overbought with bearish divergence)
- Key levels are unclear or the pattern detected is ambiguous
- Cross-domain conflict: technicals disagree significantly with fundamentals/sentiment
- High-volatility stock where price action nuance matters (biotech, meme stocks, recent IPOs)
- Near a critical inflection point (testing major support/resistance, SMA crossover imminent)

DON'T NEED DETAILED TECHNICAL if:
- Technical signals are clear and internally consistent
- This is a hold/neutral situation where technical timing is less critical
- Fundamentals are the primary driver (e.g., value plays, dividend stocks)
- Technical structured signals already provide sufficient key levels and pattern context

**FUNDAMENTAL — Do you need detailed fundamental analysis?**

NEED DETAILED FUNDAMENTAL if:
- Fundamental signals conflict internally (e.g., "High Quality" + "Declining Growth" + "Undervalued")
- Cross-domain conflicts (fundamentals bullish but technicals broken, sentiment negative)
- Edge case sector/situation (biotech, utilities with unusual metrics, distressed companies)
- High conviction signal (score >75 or <25) where you want to be certain
- Multiple low-confidence fundamental signals

DON'T NEED DETAILED FUNDAMENTAL if:
- Signals are clear and aligned across all three domains
- Neutral/hold situation (structured is sufficient for "wait and see")
- Technical/sentiment so strong they override fundamental nuances
- Fundamental signals are clear and high-confidence

=== OUTPUT FORMAT ===

Respond in EXACTLY this format:

<technical_decision>YES or NO</technical_decision>
<technical_reason>One sentence explaining why for {ticker}</technical_reason>

<fundamental_decision>YES or NO</fundamental_decision>
<fundamental_reason>One sentence explaining why for {ticker}</fundamental_reason>

CRITICAL: You are assessing {ticker} ({company_name}). Do not reference any other stock.
Then STOP. Do not provide the full synthesis yet.

Your assessment:"""
    
    def _build_synthesis_prompt(self, *, ticker, company_name, sector, industry,
                                 current_price, technical_summary, fundamental_summary,
                                 sentiment_summary, scores_summary, signal, confidence,
                                 detail_fetch_info, additional_context,
                                 perspective_summary: str = "") -> str:
        """Build the final synthesis prompt as a plain string."""
        
        return f"""You are the LEAD INVESTMENT ANALYST creating a final investment recommendation.

CRITICAL: You are analyzing {ticker} ({company_name}). Do NOT confuse this with any other stock.
You must ONLY discuss {ticker}. If you find yourself writing about a different company, STOP and correct yourself.

=== STOCK INFORMATION ===
**ANALYZING: {ticker} - {company_name}**
Sector: {sector} | Industry: {industry}
Current Price: ${current_price}

=== SPECIALIST INPUTS ===

TECHNICAL:
{technical_summary}

FUNDAMENTAL:
{fundamental_summary}

SENTIMENT:
{sentiment_summary}

=== INVESTOR PERSPECTIVE ===
{perspective_summary if perspective_summary else "(no investor perspective requested)"}

=== QUANTITATIVE SCORES ===
{scores_summary}
Signal: {signal}
Confidence: {confidence}%

============================================================
DETAIL FETCH STATUS:
{detail_fetch_info}
============================================================

=== YOUR SYNTHESIS TASK ===

**REMINDER: You are analyzing {ticker} ({company_name}) — stay focused on THIS stock only.**

Create a comprehensive investment recommendation integrating all inputs FOR {ticker}.

**1. EXECUTIVE SUMMARY** (3-4 sentences)
- Start with: "{ticker} ({company_name})..." 
- Unified investment thesis across all three domains
- Key conviction and primary driver
- Why this is a buy/hold/sell opportunity for {ticker}

**2. SYNTHESIS: TECHNICAL + FUNDAMENTAL + SENTIMENT FOR {ticker}**
- How do the three domains interact for {ticker}?
- Where do specialists agree about {ticker} (higher conviction)?
- Where do they disagree about {ticker} (requires judgment)?
{additional_context}

**3. VERDICT & CONVICTION FOR {ticker}**
- Verdict: [Buy/Add/Hold/Reduce/Sell] with conviction level [High/Medium/Low]
- The single strongest reason FOR this verdict, and the single strongest reason AGAINST it
- What would change your mind: 2-3 concrete, observable developments that would flip this verdict
- Time Horizon: over what period does this thesis play out [weeks/months/years, tied to catalysts]

**4. MONITORING FRAMEWORK FOR {ticker}**
Three specific triggers to watch for {ticker}:
- Technical: "If {ticker} price breaks $XX..." [use key levels from structured signals]
- Fundamental: "Watch {ticker} quarterly earnings for..."
- Sentiment: "If negative news emerges about {ticker}..."

**5. BULL/BEAR CASES FOR {ticker}** (integrated across domains)
Bull Case (3 bullets): Each showing convergence of positive signals for {ticker}
Bear Case (3 bullets): Each showing risks across domains for {ticker}

=== GUIDELINES ===
- Be specific: name the metrics, levels, and dates your reasoning rests on
- This is an EVALUATION, not trade construction: no position sizing, entry/stop
  prices, profit targets, or options strategies
- Own your synthesis — don't just summarize each domain separately
- Show reasoning when specialists disagree
- Make the thesis falsifiable: a reader should know exactly what evidence would disprove it
- 800-1000 words total
- **CRITICAL: Your entire analysis must be about {ticker} ({company_name}). Do not analyze any other stock.**

Begin your analysis of {ticker}:"""
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def analyze_to_record(self, ticker: str, perspective: str = None,
                          perspective_weight: float = config.DEFAULT_PERSPECTIVE_WEIGHT,
                          save: bool = True,
                          data_validator: Optional[Callable[[Dict[str, Any]], Any]] = None) -> AnalysisRecord:
        """Run a full evaluation and return (and by default persist) the AnalysisRecord.

        This is the headless entrypoint the Phase 2 runner uses; analyze() is
        the interactive wrapper that renders the record as a report.
        """
        ticker = ticker.upper()

        print(f"Gathering data for {ticker}...")
        data = self.gather_data(ticker, perspective=perspective)
        if data_validator is not None:
            data_validator(data)

        print("Calculating scores...")
        scores = self.calculate_scores(data, perspective_weight=perspective_weight)

        print("Generating comprehensive analysis...")
        llm_result = self.generate_llm_analysis(data, scores)

        record = self.build_record(data, scores, llm_result)
        if save:
            path = record.save()
            print(f"Saved analysis record -> {path}")
        return record

    def analyze(self, ticker: str, perspective: str = None,
                perspective_weight: float = config.DEFAULT_PERSPECTIVE_WEIGHT, *,
                save: bool = True,
                data_validator: Optional[Callable[[Dict[str, Any]], Any]] = None) -> str:
        """Run a full evaluation, persist the AnalysisRecord, return the rendered report."""
        record = self.analyze_to_record(ticker, perspective=perspective,
                                        perspective_weight=perspective_weight, save=save,
                                        data_validator=data_validator)
        return render_report(record)

    def build_record(self, data: Dict[str, Any], scores: Dict[str, Any],
                     llm_result: Dict[str, Any] = None) -> AnalysisRecord:
        """Assemble the persisted AnalysisRecord from pipeline outputs.

        The record is the product; the printed report is rendered from it
        (reporting.render_report). llm_result is optional so quick/screening
        paths can produce thesis-less records.
        """
        info = data.get('stock_info', {})
        llm_result = llm_result or {}

        signals: Dict[str, Any] = {}
        for domain in ('technical', 'fundamental', 'sentiment'):
            entry: Dict[str, Any] = {
                'score': scores[domain]['score'],
                'breakdown': scores[domain].get('breakdown', {}),
            }
            structured = data.get(f'{domain}_structured')
            if isinstance(structured, dict):
                entry['structured'] = structured
            signals[domain] = entry

        key_risks: List[str] = []
        persp = data.get('perspective')
        if persp:
            signals['perspective'] = {
                'score': persp.get('score'),
                'verdict': persp.get('verdict'),
                'narrative': persp.get('narrative'),
                'metadata': persp.get('metadata', {}),
            }
            key_risks = list(persp.get('key_risks', []) or [])

        for key, res in (data.get('custom_assessments') or {}).items():
            signals[key] = {
                'score': res.get('score'),
                'verdict': res.get('verdict'),
                'narrative': res.get('narrative'),
                'metadata': res.get('metadata', {}),
            }

        data_flags = [f"{k}: {v}" for k, v in data.items() if k.endswith('_error')]
        if llm_result.get('used_detailed_technical'):
            data_flags.append(f"detailed technical fetched — {llm_result.get('tech_detail_reason')}")
        if llm_result.get('used_detailed_fundamental'):
            data_flags.append(f"detailed fundamental fetched — {llm_result.get('fund_detail_reason')}")

        return AnalysisRecord(
            ticker=data['ticker'],
            company=info.get('name', data['ticker']),
            sector=info.get('sector', 'Unknown'),
            industry=info.get('industry', 'Unknown'),
            current_price=self._current_price(data),
            model=self.model,
            signals=signals,
            weights=scores['weights'],
            overall_score=scores['overall'],
            signal=scores['signal'].value,
            confidence=scores['confidence'],
            thesis=llm_result.get('full_analysis', ''),
            key_risks=key_risks,
            data_flags=data_flags,
        )

    def get_quick_recommendation(self, ticker: str) -> Dict[str, Any]:
        """Get quick recommendation without full LLM synthesis (not persisted)."""
        ticker = ticker.upper()

        data = self.gather_data(ticker)
        scores = self.calculate_scores(data)

        return {
            'ticker': ticker,
            'company': data.get('stock_info', {}).get('name', ticker),
            'current_price': self._current_price(data),
            'scores': {
                'technical': scores['technical']['score'],
                'fundamental': scores['fundamental']['score'],
                'sentiment': scores['sentiment']['score'],
                'overall': scores['overall']
            },
            'signal': scores['signal'].value,
            'confidence': scores['confidence'],
        }
    
    def compare_stocks(self, tickers: List[str]) -> str:
        """Compare multiple stocks for investment selection."""
        if len(tickers) < 2:
            return "Please provide at least 2 tickers for comparison."
        
        print(f"Comparing {len(tickers)} stocks...")
        
        results = []
        for ticker in tickers[:5]:
            print(f"  Analyzing {ticker}...")
            rec = self.get_quick_recommendation(ticker)
            results.append(rec)
        
        results.sort(key=lambda x: x['scores']['overall'], reverse=True)
        
        header = f"""
{'='*100}
STOCK COMPARISON ANALYSIS
{'='*100}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{'Ticker':<10}{'Company':<25}{'Price':>10}{'Tech':>8}{'Fund':>8}{'Sent':>8}{'Overall':>10}{'Signal':<15}{'Conf':>8}
{'-'*100}"""
        
        rows = []
        for r in results:
            price = f"${r['current_price']:>8.2f}" if r['current_price'] else f"{'n/a':>9}"
            rows.append(
                f"{r['ticker']:<10}"
                f"{r['company'][:23]:<25}"
                f"{price}"
                f"{r['scores']['technical']:>8.1f}"
                f"{r['scores']['fundamental']:>8.1f}"
                f"{r['scores']['sentiment']:>8.1f}"
                f"{r['scores']['overall']:>10.1f}"
                f"{r['signal']:<15}"
                f"{r['confidence']:>7.0f}%"
            )

        return header + "\n" + "\n".join(rows) + "\n" + "-"*100
    
    def screen_portfolio(self, tickers: List[str]) -> str:
        """Screen a portfolio of stocks and provide overview."""
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
        
        strong_buys = [r for r in results if r['signal'] == 'Strong Buy']
        buys = [r for r in results if r['signal'] == 'Buy']
        holds = [r for r in results if r['signal'] == 'Hold']
        sells = [r for r in results if r['signal'] == 'Sell']
        strong_sells = [r for r in results if r['signal'] == 'Strong Sell']
        
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
    agent = GeneralAnalystAgent()
    return agent.analyze(ticker)


def get_recommendation(ticker: str) -> Dict:
    agent = GeneralAnalystAgent()
    return agent.get_quick_recommendation(ticker)


def compare_stocks(tickers: List[str]) -> str:
    agent = GeneralAnalystAgent()
    return agent.compare_stocks(tickers)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='General Analyst Agent')
    parser.add_argument('tickers', nargs='+', help='Stock ticker(s) to analyze')
    parser.add_argument('--quick', '-q', action='store_true', 
                       help='Quick recommendation only (no LLM analysis)')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare multiple stocks')
    parser.add_argument('--screen', '-s', action='store_true',
                       help='Screen portfolio')
    parser.add_argument('--perspective', '-p', default=None,
                        help='Investor perspective to apply (must be distilled first)')
    parser.add_argument('--perspective-weight', type=float,
                        default=config.DEFAULT_PERSPECTIVE_WEIGHT,
                        help=f'Perspective weight 0–{config.MAX_PERSPECTIVE_WEIGHT} '
                             f'(default {config.DEFAULT_PERSPECTIVE_WEIGHT})')
    args = parser.parse_args()
    
    agent = GeneralAnalystAgent(use_perspective=bool(args.perspective))
    
    if args.screen:
        print(agent.screen_portfolio(args.tickers))
    elif args.compare and len(args.tickers) > 1:
        print(agent.compare_stocks(args.tickers))
    elif args.quick:
        for ticker in args.tickers:
            rec = agent.get_quick_recommendation(ticker)
            print(f"\n{ticker}: {rec['signal']} (Score: {rec['scores']['overall']:.1f}, Confidence: {rec['confidence']}%)")
    else:
        for ticker in args.tickers:
            print(agent.analyze(ticker, perspective=args.perspective,
                                perspective_weight=args.perspective_weight))
            print("\n")


if __name__ == "__main__":
    main()
