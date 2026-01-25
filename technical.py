# technical_analyzer.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class TechnicalCalculator:
    """
    Pure technical indicator calculations - no LLM, just math.
    """
    
    def __init__(self, period: str = "6mo", interval: str = "1d"):
        """
        Args:
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        self.period = period
        self.interval = interval
    
    def get_price_data(self, ticker: str) -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=self.period, interval=self.interval)
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            return df
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get stock metadata (sector, industry, market cap)"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'name': info.get('longName', ticker),
                'current_price': info.get('currentPrice', 0)
            }
        except Exception as e:
            return {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'name': ticker,
                'current_price': 0
            }
    
    def calculate_sma(self, df: pd.DataFrame, window: int) -> float:
        """Calculate Simple Moving Average - return latest value"""
        return df['Close'].rolling(window=window).mean().iloc[-1]
    
    def calculate_ema(self, df: pd.DataFrame, window: int) -> float:
        """Calculate Exponential Moving Average - return latest value"""
        return df['Close'].ewm(span=window, adjust=False).mean().iloc[-1]
    
    def calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> float:
        """Calculate Relative Strength Index - return latest value"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_macd(self, df: pd.DataFrame, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD - return latest values"""
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, 
                                  window: int = 20, num_std: float = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands - return latest values"""
        sma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'upper': upper_band.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower_band.iloc[-1],
            'width': upper_band.iloc[-1] - lower_band.iloc[-1]
        }
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate historical volatility (standard deviation of returns)"""
        returns = df['Close'].pct_change()
        volatility = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)  # Annualized
        return volatility
    
    def calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-related metrics"""
        avg_volume = df['Volume'].mean()
        current_volume = df['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Volume trend (last 5 days vs previous 5 days)
        recent_volume = df['Volume'].iloc[-5:].mean()
        prev_volume = df['Volume'].iloc[-10:-5].mean()
        volume_trend = (recent_volume - prev_volume) / prev_volume if prev_volume > 0 else 0
        
        return {
            'current': current_volume,
            'average': avg_volume,
            'ratio': volume_ratio,
            'trend': volume_trend
        }
    
    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        """Calculate Average True Range (volatility measure)"""
        high_low = df['High'] - df['Low']
        high_close = pd.Series((df['High'] - df['Close'].shift()).abs(), index=df.index)
        low_close = pd.Series((df['Low'] - df['Close'].shift()).abs(), index=df.index)
        
        ranges = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        })
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window).mean().iloc[-1]
        return atr
    
    def calculate_momentum(self, df: pd.DataFrame, window: int = 10) -> float:
        """Calculate price momentum (rate of change)"""
        momentum = ((df['Close'].iloc[-1] - df['Close'].iloc[-window]) / 
                   df['Close'].iloc[-window] * 100)
        return momentum
    
    def calculate_adx(self, df: pd.DataFrame, window: int = 14) -> float:
        """Calculate Average Directional Index (trend strength)"""
        # Simplified ADX calculation
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self.calculate_atr(df, window)
        
        pos_di = 100 * (pos_dm.rolling(window).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window).mean() / atr)
        
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
        adx = dx.rolling(window).mean().iloc[-1]
        
        return adx if not pd.isna(adx) else 0
    
    def get_all_indicators(self, ticker: str) -> Dict:
        """Calculate all available technical indicators"""
        df = self.get_price_data(ticker)
        current_price = df['Close'].iloc[-1]
        
        indicators = {
            'price': {
                'current': current_price,
                'change_1d': ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100),
                'change_5d': ((current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100) if len(df) > 5 else 0,
                'change_20d': ((current_price - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100) if len(df) > 20 else 0,
            },
            'moving_averages': {
                'sma_20': self.calculate_sma(df, 20),
                'sma_50': self.calculate_sma(df, 50),
                'sma_200': self.calculate_sma(df, 200) if len(df) >= 200 else None,
                'ema_12': self.calculate_ema(df, 12),
                'ema_26': self.calculate_ema(df, 26),
            },
            'momentum': {
                'rsi_14': self.calculate_rsi(df, 14),
                'momentum_10d': self.calculate_momentum(df, 10),
            },
            'trend': {
                'macd': self.calculate_macd(df),
                'adx': self.calculate_adx(df),
            },
            'volatility': {
                'bollinger': self.calculate_bollinger_bands(df),
                'atr': self.calculate_atr(df),
                'historical_vol': self.calculate_volatility(df),
            },
            'volume': self.calculate_volume_metrics(df)
        }
        
        return indicators


# Example usage
if __name__ == "__main__":
    calc = TechnicalCalculator(period="6mo", interval="1d")
    
    print("=== Stock Info ===")
    info = calc.get_stock_info("AAPL")
    print(f"Name: {info['name']}")
    print(f"Sector: {info['sector']}")
    print(f"Industry: {info['industry']}")
    print(f"Market Cap: ${info['market_cap']:,}")
    
    print("\n=== All Indicators ===")
    indicators = calc.get_all_indicators("AAPL")
    
    print(f"\nPrice: ${indicators['price']['current']:.2f}")
    print(f"SMA(20): ${indicators['moving_averages']['sma_20']:.2f}")
    print(f"RSI: {indicators['momentum']['rsi_14']:.2f}")
    print(f"MACD: {indicators['trend']['macd']['macd']:.2f}")
    print(f"Volatility: {indicators['volatility']['historical_vol']:.2%}")