"""
Market Data Analyzer Agent
Handles technical analysis, price action, and trend identification for stocks.

Based on Investment design document specifications:
- Fetch current quotes and price history
- Calculate technical indicators (MA, RSI, MACD, Bollinger Bands)
- Identify trend direction and momentum
- Track volume and volatility patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
import logging


@dataclass
class TechnicalSignal:
    """Represents a technical analysis signal"""
    indicator: str
    signal: str  # 'bullish', 'bearish', 'neutral'
    value: float
    description: str
    strength: float  # 0-1 confidence score


@dataclass
class MarketDataAnalysis:
    """Complete market data analysis result"""
    symbol: str
    timestamp: datetime
    current_price: float
    price_change_pct: float
    trend: str  # 'uptrend', 'downtrend', 'sideways'
    momentum: str  # 'strong', 'moderate', 'weak'
    signals: List[TechnicalSignal]
    volatility: float
    volume_trend: str
    support_levels: List[float]
    resistance_levels: List[float]
    overall_score: float  # -1 to 1, bearish to bullish


class MarketDataAnalyzer:
    """
    Technical analysis agent for stock market data.
    Fetches price data and calculates technical indicators.
    """
    
    def __init__(self, lookback_days: int = 90):
        """
        Initialize Market Data Analyzer
        
        Args:
            lookback_days: Number of days of historical data to analyze
        """
        self.lookback_days = lookback_days
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, symbol: str) -> MarketDataAnalysis:
        """
        Perform complete technical analysis on a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            MarketDataAnalysis object with all technical indicators
        """
        self.logger.info(f"Analyzing market data for {symbol}")
        
        # Fetch price data
        df = self._fetch_price_data(symbol)
        
        if df is None or df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Generate signals
        signals = self._generate_signals(df)
        
        # Determine trend and momentum
        trend = self._determine_trend(df)
        momentum = self._determine_momentum(df)
        
        # Find support/resistance levels
        support_levels, resistance_levels = self._find_support_resistance(df)
        
        # Calculate volatility
        volatility = self._calculate_volatility(df)
        
        # Analyze volume trend
        volume_trend = self._analyze_volume_trend(df)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(signals, trend, momentum)
        
        # Get current price info
        current_price = df['Close'].iloc[-1]
        price_change_pct = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        return MarketDataAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
            price_change_pct=price_change_pct,
            trend=trend,
            momentum=momentum,
            signals=signals,
            volatility=volatility,
            volume_trend=volume_trend,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            overall_score=overall_score
        )
    
    def _fetch_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical price data from Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return None
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean() if len(df) >= 200 else np.nan
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Average True Range (ATR) for volatility
        df['ATR'] = self._calculate_atr(df)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = pd.Series(np.abs(df['High'] - df['Close'].shift()), index=df.index)
        low_close = pd.Series(np.abs(df['Low'] - df['Close'].shift()), index=df.index)
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _generate_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate trading signals from technical indicators"""
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # RSI Signals
        if latest['RSI'] < 30:
            signals.append(TechnicalSignal(
                indicator='RSI',
                signal='bullish',
                value=latest['RSI'],
                description=f'RSI oversold at {latest["RSI"]:.1f}',
                strength=0.8
            ))
        elif latest['RSI'] > 70:
            signals.append(TechnicalSignal(
                indicator='RSI',
                signal='bearish',
                value=latest['RSI'],
                description=f'RSI overbought at {latest["RSI"]:.1f}',
                strength=0.8
            ))
        
        # MACD Signals
        if prev['MACD'] < prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
            signals.append(TechnicalSignal(
                indicator='MACD',
                signal='bullish',
                value=latest['MACD'],
                description='MACD bullish crossover',
                strength=0.7
            ))
        elif prev['MACD'] > prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
            signals.append(TechnicalSignal(
                indicator='MACD',
                signal='bearish',
                value=latest['MACD'],
                description='MACD bearish crossover',
                strength=0.7
            ))
        
        # Moving Average Crossovers
        if not pd.isna(latest['SMA_50']):
            if prev['SMA_20'] < prev['SMA_50'] and latest['SMA_20'] > latest['SMA_50']:
                signals.append(TechnicalSignal(
                    indicator='MA_Cross',
                    signal='bullish',
                    value=latest['SMA_20'],
                    description='Golden cross: 20-day MA crossed above 50-day MA',
                    strength=0.75
                ))
            elif prev['SMA_20'] > prev['SMA_50'] and latest['SMA_20'] < latest['SMA_50']:
                signals.append(TechnicalSignal(
                    indicator='MA_Cross',
                    signal='bearish',
                    value=latest['SMA_20'],
                    description='Death cross: 20-day MA crossed below 50-day MA',
                    strength=0.75
                ))
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_Lower']:
            signals.append(TechnicalSignal(
                indicator='Bollinger',
                signal='bullish',
                value=latest['Close'],
                description='Price below lower Bollinger Band',
                strength=0.6
            ))
        elif latest['Close'] > latest['BB_Upper']:
            signals.append(TechnicalSignal(
                indicator='Bollinger',
                signal='bearish',
                value=latest['Close'],
                description='Price above upper Bollinger Band',
                strength=0.6
            ))
        
        # Volume Signals
        if latest['Volume_Ratio'] > 1.5:
            # High volume - confirm the trend
            price_direction = 'bullish' if latest['Close'] > prev['Close'] else 'bearish'
            signals.append(TechnicalSignal(
                indicator='Volume',
                signal=price_direction,
                value=latest['Volume_Ratio'],
                description=f'High volume ({latest["Volume_Ratio"]:.1f}x average) confirming {price_direction} move',
                strength=0.5
            ))
        
        return signals
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine overall price trend"""
        latest = df.iloc[-1]
        
        # Check moving averages alignment
        if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_50']):
            if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
                return 'uptrend'
            elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
                return 'downtrend'
        
        # Fallback to simple price comparison
        month_ago_price = df['Close'].iloc[-20] if len(df) >= 20 else df['Close'].iloc[0]
        price_change = (latest['Close'] - month_ago_price) / month_ago_price
        
        if price_change > 0.05:
            return 'uptrend'
        elif price_change < -0.05:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _determine_momentum(self, df: pd.DataFrame) -> str:
        """Determine price momentum strength"""
        latest = df.iloc[-1]
        
        # Use RSI and recent price change
        week_ago_price = df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[0]
        week_change = abs((latest['Close'] - week_ago_price) / week_ago_price)
        
        rsi_momentum = abs(latest['RSI'] - 50) / 50  # 0-1 scale
        
        combined_momentum = (week_change * 10 + rsi_momentum) / 2
        
        if combined_momentum > 0.5:
            return 'strong'
        elif combined_momentum > 0.25:
            return 'moderate'
        else:
            return 'weak'
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels using local minima/maxima"""
        
        # Get recent highs and lows
        recent_df = df.tail(window)
        
        # Find local maxima (resistance)
        highs = recent_df['High'].values
        resistance = []
        for i in range(2, len(highs) - 2):
            if highs[i] == max(highs[i-2:i+3]):
                resistance.append(highs[i])
        
        # Find local minima (support)
        lows = recent_df['Low'].values
        support = []
        for i in range(2, len(lows) - 2):
            if lows[i] == min(lows[i-2:i+3]):
                support.append(lows[i])
        
        # Return top 3 of each
        resistance = sorted(set(resistance), reverse=True)[:3]
        support = sorted(set(support))[:3]
        
        return support, resistance
    
    def _calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate recent volatility (standard deviation of returns)"""
        returns = df['Close'].pct_change().tail(period)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return volatility
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> str:
        """Analyze recent volume trend"""
        latest = df.iloc[-1]
        
        if latest['Volume_Ratio'] > 1.2:
            return 'increasing'
        elif latest['Volume_Ratio'] < 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_overall_score(self, signals: List[TechnicalSignal], 
                                trend: str, momentum: str) -> float:
        """
        Calculate overall technical score from -1 (bearish) to 1 (bullish)
        """
        score = 0.0
        
        # Weight signals by their strength
        for signal in signals:
            if signal.signal == 'bullish':
                score += signal.strength
            elif signal.signal == 'bearish':
                score -= signal.strength
        
        # Add trend component
        if trend == 'uptrend':
            score += 0.3
        elif trend == 'downtrend':
            score -= 0.3
        
        # Add momentum component
        momentum_weight = {'strong': 0.2, 'moderate': 0.1, 'weak': 0.0}
        if trend == 'uptrend':
            score += momentum_weight[momentum]
        elif trend == 'downtrend':
            score -= momentum_weight[momentum]
        
        # Normalize to -1 to 1
        score = max(-1.0, min(1.0, score / 3))
        
        return score
    
    def format_analysis(self, analysis: MarketDataAnalysis) -> str:
        """Format analysis results as human-readable text"""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Market Data Analysis: {analysis.symbol}")
        output.append(f"{'='*60}")
        output.append(f"Timestamp: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Current Price: ${analysis.current_price:.2f} ({analysis.price_change_pct:+.2f}%)")
        output.append(f"\nTrend: {analysis.trend.upper()}")
        output.append(f"Momentum: {analysis.momentum.upper()}")
        output.append(f"Volatility: {analysis.volatility:.2%} (annualized)")
        output.append(f"Volume Trend: {analysis.volume_trend}")
        
        if analysis.support_levels:
            output.append(f"\nSupport Levels: {', '.join([f'${x:.2f}' for x in analysis.support_levels])}")
        if analysis.resistance_levels:
            output.append(f"Resistance Levels: {', '.join([f'${x:.2f}' for x in analysis.resistance_levels])}")
        
        output.append(f"\nTechnical Signals ({len(analysis.signals)}):")
        for signal in analysis.signals:
            output.append(f"  [{signal.signal.upper()}] {signal.indicator}: {signal.description} (strength: {signal.strength:.1%})")
        
        output.append(f"\nOverall Technical Score: {analysis.overall_score:+.2f} ")
        if analysis.overall_score > 0.3:
            output.append("(Bullish)")
        elif analysis.overall_score < -0.3:
            output.append("(Bearish)")
        else:
            output.append("(Neutral)")
        
        output.append(f"{'='*60}\n")
        
        return '\n'.join(output)


def main():
    """Test the Market Data Analyzer"""
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with a symbol
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    
    analyzer = MarketDataAnalyzer(lookback_days=90)
    
    try:
        analysis = analyzer.analyze(symbol)
        print(analyzer.format_analysis(analysis))
        
        # Also print JSON for integration testing
        print("\nJSON Output:")
        import json
        result = {
            'symbol': analysis.symbol,
            'timestamp': analysis.timestamp.isoformat(),
            'current_price': analysis.current_price,
            'price_change_pct': analysis.price_change_pct,
            'trend': analysis.trend,
            'momentum': analysis.momentum,
            'volatility': analysis.volatility,
            'volume_trend': analysis.volume_trend,
            'overall_score': analysis.overall_score,
            'signals': [
                {
                    'indicator': s.indicator,
                    'signal': s.signal,
                    'value': s.value,
                    'description': s.description,
                    'strength': s.strength
                }
                for s in analysis.signals
            ]
        }
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())