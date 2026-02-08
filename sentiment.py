"""
Sentiment Data Fetcher Module
=============================
Fetches news data from multiple sources for sentiment analysis.
Designed to support investment decisions by analyzing company and industry sentiment.

Data Sources:
- Yahoo Finance (free, no API key required)
- Finnhub (free tier: 60 calls/min)
- NewsAPI (free tier: 100 calls/day)
- Alpha Vantage (free tier: 25 calls/day, includes pre-computed sentiment scores)

Usage:
    from sentiment_data_fetcher import SentimentDataFetcher
    
    fetcher = SentimentDataFetcher()
    data = fetcher.fetch_all(
        ticker="AAPL",
        company_name="Apple Inc",
        industry="Technology"
    )
"""

import os
import json
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from dotenv import load_dotenv
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration settings"""
    CACHE_DIR = "data/news_cache"
    CACHE_EXPIRY_HOURS = {'company': 1, 'industry': 2, 'macro': 4}
    REQUEST_TIMEOUT = 30
    RATE_LIMIT = {'finnhub': 60, 'newsapi': 100, 'alphavantage': 5}
    
    # API Keys from environment
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')


# =============================================================================
# Data Models
# =============================================================================

class NewsSource(Enum):
    FINNHUB = "finnhub"
    NEWS_API = "newsapi"
    ALPHA_VANTAGE = "alphavantage"
    YAHOO_FINANCE = "yahoo_finance"


class ContentCategory(Enum):
    COMPANY = "company"
    INDUSTRY = "industry"
    MACRO = "macro"
    SOCIAL = "social"


@dataclass
class NewsArticle:
    """Standardized news article"""
    title: str
    content: str
    source: str
    source_type: NewsSource
    published_at: datetime
    url: Optional[str] = None
    category: ContentCategory = ContentCategory.COMPANY
    ticker: Optional[str] = None
    industry: Optional[str] = None
    sentiment_hint: Optional[float] = None
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title, 'content': self.content, 'source': self.source,
            'source_type': self.source_type.value, 'published_at': self.published_at.isoformat(),
            'url': self.url, 'category': self.category.value, 'ticker': self.ticker,
            'industry': self.industry, 'sentiment_hint': self.sentiment_hint,
            'keywords': self.keywords, 'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NewsArticle':
        return cls(
            title=data['title'], content=data['content'], source=data['source'],
            source_type=NewsSource(data['source_type']),
            published_at=datetime.fromisoformat(data['published_at']),
            url=data.get('url'), category=ContentCategory(data.get('category', 'company')),
            ticker=data.get('ticker'), industry=data.get('industry'),
            sentiment_hint=data.get('sentiment_hint'), keywords=data.get('keywords', []),
            metadata=data.get('metadata', {})
        )


@dataclass
class FetchResult:
    """Fetch operation result"""
    success: bool
    source: NewsSource
    articles: List[NewsArticle] = field(default_factory=list)
    error: Optional[str] = None
    cached: bool = False
    
    @property
    def total_items(self) -> int:
        return len(self.articles)


# =============================================================================
# Cache Manager
# =============================================================================

class CacheManager:
    """Handles caching of fetched data"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, identifier: str, category: str) -> str:
        date_str = datetime.now().strftime('%Y%m%d')
        return hashlib.md5(f"{identifier}_{category}_{date_str}".encode()).hexdigest()
    
    def get(self, identifier: str, category: str = 'company') -> Optional[List[Dict]]:
        cache_path = os.path.join(self.cache_dir, f"{self._get_cache_key(identifier, category)}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cached_at = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_at > timedelta(hours=Config.CACHE_EXPIRY_HOURS.get(category, 2)):
                return None
            logger.info(f"Cache hit: {identifier}")
            return data.get('items', [])
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, identifier: str, items: List[Dict], category: str = 'company'):
        cache_path = os.path.join(self.cache_dir, f"{self._get_cache_key(identifier, category)}.json")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'identifier': identifier, 'category': category, 
                          'cached_at': datetime.now().isoformat(), 'items': items}, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")


# =============================================================================
# Base Fetcher
# =============================================================================

class BaseFetcher(ABC):
    """Abstract base class for all fetchers"""
    
    def __init__(self):
        self.source: NewsSource = None
        self.timeout = Config.REQUEST_TIMEOUT
        self._last_request_time = 0
    
    @abstractmethod
    def fetch(self, query: str, max_items: int = 20, **kwargs) -> FetchResult:
        pass
    
    def _rate_limit(self, source_name: str):
        min_interval = 60.0 / Config.RATE_LIMIT.get(source_name, 60)
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    def _parse_datetime(self, dt_string: str) -> datetime:
        for fmt in ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
            try:
                return datetime.strptime(dt_string, fmt)
            except (ValueError, TypeError):
                continue
        return datetime.now()
    
    def _create_error_result(self, error_msg: str) -> FetchResult:
        return FetchResult(success=False, source=self.source, error=error_msg)


# =============================================================================
# Yahoo Finance Fetcher (No API key required)
# =============================================================================

class YahooFinanceFetcher(BaseFetcher):
    """Fetches news from Yahoo Finance - NO API KEY REQUIRED"""
    
    def __init__(self):
        super().__init__()
        self.source = NewsSource.YAHOO_FINANCE
    
    def fetch(self, query: str, max_items: int = 20, **kwargs) -> FetchResult:
        try:
            import yfinance as yf
            ticker = yf.Ticker(query.upper())
            news = ticker.news
            if not news:
                return FetchResult(success=True, source=self.source, articles=[], error="No news found")
            
            articles = []
            for item in news[:max_items]:
                # Handle both old and new yfinance API structures
                if 'content' in item:
                    # New structure: data is nested under 'content'
                    content = item.get('content', {})
                    title = content.get('title', '')
                    summary = content.get('summary', content.get('description', title))
                    publisher = content.get('provider', {}).get('displayName', 'Yahoo Finance')
                    pub_time = content.get('pubDate', '')
                    url = content.get('canonicalUrl', {}).get('url', '')
                    
                    # Parse datetime from ISO format
                    try:
                        published_at = datetime.fromisoformat(pub_time.replace('Z', '+00:00')) if pub_time else datetime.now()
                    except:
                        published_at = datetime.now()
                else:
                    # Old structure: data at top level
                    title = item.get('title', '')
                    summary = item.get('summary', title)
                    publisher = item.get('publisher', 'Yahoo Finance')
                    pub_time = item.get('providerPublishTime', 0)
                    url = item.get('link', '')
                    published_at = datetime.fromtimestamp(pub_time) if isinstance(pub_time, (int, float)) else datetime.now()
                
                if title:  # Only add if we have a title
                    articles.append(NewsArticle(
                        title=title,
                        content=summary,
                        source=publisher,
                        source_type=self.source,
                        published_at=published_at,
                        url=url,
                        category=ContentCategory.COMPANY,
                        ticker=query.upper()
                    ))
            
            logger.info(f"Yahoo Finance: fetched {len(articles)} articles for {query}")
            return FetchResult(success=True, source=self.source, articles=articles)
        except ImportError:
            return self._create_error_result("yfinance required. Install: pip install yfinance")
        except Exception as e:
            return self._create_error_result(f"Yahoo Finance fetch failed: {e}")


# =============================================================================
# Finnhub Fetcher (60 calls/minute free)
# =============================================================================

class FinnhubFetcher(BaseFetcher):
    """Fetches company news from Finnhub. Free tier: 60 calls/min"""
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.source = NewsSource.FINNHUB
        self.api_key = api_key or Config.FINNHUB_API_KEY
    
    def fetch(self, query: str, max_items: int = 20, **kwargs) -> FetchResult:
        if not self.api_key:
            return self._create_error_result("Set FINNHUB_API_KEY environment variable")
        
        days_back = kwargs.get('days_back', 7)
        ticker = query.upper()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self._rate_limit('finnhub')
        data = self._make_request(f"{self.BASE_URL}/company-news", {
            'symbol': ticker, 'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'), 'token': self.api_key
        })
        
        if data is None:
            return self._create_error_result("Failed to fetch from Finnhub")
        
        articles = []
        for item in data[:max_items]:
            articles.append(NewsArticle(
                title=item.get('headline', ''), content=item.get('summary', ''),
                source=item.get('source', 'Finnhub'), source_type=self.source,
                published_at=datetime.fromtimestamp(item.get('datetime', 0)),
                url=item.get('url'), category=ContentCategory.COMPANY, ticker=ticker,
                keywords=item.get('related', '').split(',') if item.get('related') else []
            ))
        logger.info(f"Finnhub: fetched {len(articles)} articles for {ticker}")
        return FetchResult(success=True, source=self.source, articles=articles)
    
    def fetch_market_news(self, category: str = 'general', max_items: int = 20) -> FetchResult:
        if not self.api_key:
            return self._create_error_result("Finnhub API key not configured")
        self._rate_limit('finnhub')
        data = self._make_request(f"{self.BASE_URL}/news", {'category': category, 'token': self.api_key})
        if data is None:
            return self._create_error_result("Failed to fetch market news")
        articles = [NewsArticle(
            title=item.get('headline', ''), content=item.get('summary', ''),
            source=item.get('source', 'Finnhub'), source_type=self.source,
            published_at=datetime.fromtimestamp(item.get('datetime', 0)),
            url=item.get('url'), category=ContentCategory.MACRO
        ) for item in data[:max_items]]
        return FetchResult(success=True, source=self.source, articles=articles)


# =============================================================================
# NewsAPI Fetcher (100 requests/day free)
# =============================================================================

class NewsAPIFetcher(BaseFetcher):
    """Fetches from NewsAPI.org. Free tier: 100 requests/day"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.source = NewsSource.NEWS_API
        self.api_key = api_key or Config.NEWS_API_KEY
    
    def fetch(self, query: str, max_items: int = 20, **kwargs) -> FetchResult:
        if not self.api_key:
            return self._create_error_result("Set NEWS_API_KEY environment variable")
        
        self._rate_limit('newsapi')
        data = self._make_request(f"{self.BASE_URL}/everything", {
            'q': query, 'language': kwargs.get('language', 'en'),
            'sortBy': kwargs.get('sort_by', 'publishedAt'),
            'pageSize': min(max_items, 100), 'apiKey': self.api_key
        })
        
        if data is None or data.get('status') != 'ok':
            return self._create_error_result(data.get('message', 'NewsAPI error') if data else "Request failed")
        
        articles = [NewsArticle(
            title=item.get('title', ''),
            content=item.get('description', '') or item.get('content', ''),
            source=item.get('source', {}).get('name', 'NewsAPI'),
            source_type=self.source,
            published_at=self._parse_datetime(item.get('publishedAt', '')),
            url=item.get('url'),
            category=kwargs.get('category', ContentCategory.COMPANY)
        ) for item in data.get('articles', [])[:max_items]]
        
        logger.info(f"NewsAPI: fetched {len(articles)} articles for '{query}'")
        return FetchResult(success=True, source=self.source, articles=articles)


# =============================================================================
# Alpha Vantage Fetcher (25 requests/day free - INCLUDES SENTIMENT!)
# =============================================================================

class AlphaVantageFetcher(BaseFetcher):
    """Fetches news with pre-computed sentiment scores. Free: 25 requests/day"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.source = NewsSource.ALPHA_VANTAGE
        self.api_key = api_key or Config.ALPHA_VANTAGE_API_KEY
    
    def fetch(self, query: str, max_items: int = 20, **kwargs) -> FetchResult:
        if not self.api_key:
            return self._create_error_result("Set ALPHA_VANTAGE_API_KEY environment variable")
        
        self._rate_limit('alphavantage')
        data = self._make_request(self.BASE_URL, {
            'function': 'NEWS_SENTIMENT', 'tickers': query.upper(),
            'limit': min(max_items, 200), 'apikey': self.api_key
        })
        
        if data is None:
            return self._create_error_result("Failed to fetch from Alpha Vantage")
        if 'Note' in data or 'Information' in data:
            return self._create_error_result(data.get('Note', data.get('Information', 'API limit')))
        if 'feed' not in data:
            return self._create_error_result("No news feed in response")
        
        articles = []
        for item in data.get('feed', [])[:max_items]:
            ticker_sentiments = {t.get('ticker'): {
                'relevance': float(t.get('relevance_score', 0)),
                'sentiment': float(t.get('ticker_sentiment_score', 0)),
                'label': t.get('ticker_sentiment_label')
            } for t in item.get('ticker_sentiment', [])}
            
            articles.append(NewsArticle(
                title=item.get('title', ''), content=item.get('summary', ''),
                source=item.get('source', 'Alpha Vantage'), source_type=self.source,
                published_at=self._parse_datetime(item.get('time_published', '')),
                url=item.get('url'), category=ContentCategory.COMPANY, ticker=query.upper(),
                sentiment_hint=item.get('overall_sentiment_score', 0),
                keywords=[t.get('topic') for t in item.get('topics', [])],
                metadata={'sentiment_label': item.get('overall_sentiment_label'), 'ticker_sentiments': ticker_sentiments}
            ))
        logger.info(f"Alpha Vantage: fetched {len(articles)} articles with sentiment for {query}")
        return FetchResult(success=True, source=self.source, articles=articles)


# =============================================================================
# Main Aggregator: SentimentDataFetcher
# =============================================================================

class SentimentDataFetcher:
    """Main class that aggregates all data sources for sentiment analysis."""
    
    def __init__(self, cache_enabled: bool = True):
        self.cache = CacheManager() if cache_enabled else None
        self.yahoo = YahooFinanceFetcher()
        self.finnhub = FinnhubFetcher()
        self.newsapi = NewsAPIFetcher()
        self.alphavantage = AlphaVantageFetcher()
        
        self.available_sources = {
            'yahoo': True,
            'finnhub': bool(Config.FINNHUB_API_KEY),
            'newsapi': bool(Config.NEWS_API_KEY),
            'alphavantage': bool(Config.ALPHA_VANTAGE_API_KEY)
        }
        logger.info(f"Available sources: {[k for k, v in self.available_sources.items() if v]}")
    
    @staticmethod
    def _normalize_datetime(dt: datetime) -> datetime:
        """Remove timezone info for comparison"""
        return dt.replace(tzinfo=None) if dt.tzinfo else dt

    def fetch_company_news(self, ticker: str, company_name: str = None, max_items: int = 30, use_cache: bool = True) -> Dict:
        """Fetch all available news for a specific company."""
        cache_key = f"company_{ticker}"
        
        if use_cache and self.cache:
            cached = self.cache.get(cache_key, 'company')
            if cached:
                return {'success': True, 'ticker': ticker, 'articles': [NewsArticle.from_dict(a) for a in cached],
                       'total_count': len(cached), 'from_cache': True, 'sources_used': ['cache']}
        
        all_articles, sources_used, errors = [], [], []
        items_per_source = max(5, max_items // 4)
        
        # Yahoo Finance (always available)
        result = self.yahoo.fetch(ticker, max_items=items_per_source)
        if result.success and result.articles:
            all_articles.extend(result.articles)
            sources_used.append('yahoo_finance')
        elif result.error:
            errors.append(f"Yahoo: {result.error}")
        
        # Finnhub
        if self.available_sources['finnhub']:
            result = self.finnhub.fetch(ticker, max_items=items_per_source)
            if result.success and result.articles:
                all_articles.extend(result.articles)
                sources_used.append('finnhub')
        
        # Alpha Vantage (includes sentiment!)
        if self.available_sources['alphavantage']:
            result = self.alphavantage.fetch(ticker, max_items=items_per_source)
            if result.success and result.articles:
                all_articles.extend(result.articles)
                sources_used.append('alphavantage')
        
        # NewsAPI (search by company name)
        if self.available_sources['newsapi'] and company_name:
            result = self.newsapi.fetch(company_name, max_items=items_per_source)
            if result.success and result.articles:
                for a in result.articles:
                    a.ticker = ticker
                all_articles.extend(result.articles)
                sources_used.append('newsapi')
        
        # Deduplicate
        seen, unique = set(), []
        for article in all_articles:
            key = article.title.lower().strip()[:100]
            if key and key not in seen:
                seen.add(key)
                unique.append(article)
        unique.sort(key=lambda x: self._normalize_datetime(x.published_at), reverse=True)
        unique = unique[:max_items]
        
        if self.cache and unique:
            self.cache.set(cache_key, [a.to_dict() for a in unique], 'company')
        
        return {'success': True, 'ticker': ticker, 'company_name': company_name, 'articles': unique,
               'total_count': len(unique), 'from_cache': False, 'sources_used': sources_used, 'errors': errors or None}
    
    def fetch_industry_news(self, industry: str, max_items: int = 20, use_cache: bool = True) -> Dict:
        """Fetch news related to an industry sector."""
        cache_key = f"industry_{industry.lower().replace(' ', '_')}"
        
        if use_cache and self.cache:
            cached = self.cache.get(cache_key, 'industry')
            if cached:
                return {'success': True, 'industry': industry, 'articles': [NewsArticle.from_dict(a) for a in cached],
                       'total_count': len(cached), 'from_cache': True}
        
        industry_keywords = {
            'technology': 'technology sector stocks tech', 'healthcare': 'healthcare pharmaceutical biotech',
            'finance': 'banking financial services', 'energy': 'energy oil gas renewable',
            'consumer': 'consumer goods retail', 'industrial': 'manufacturing industrial',
            'real estate': 'real estate REIT property', 'materials': 'materials mining commodities'
        }
        query = industry_keywords.get(industry.lower(), industry)
        
        articles = []
        if self.available_sources['newsapi']:
            result = self.newsapi.fetch(query, max_items=max_items, category=ContentCategory.INDUSTRY)
            if result.success:
                for a in result.articles:
                    a.industry = industry
                    a.category = ContentCategory.INDUSTRY
                articles.extend(result.articles)
        
        if self.available_sources['finnhub'] and len(articles) < max_items:
            result = self.finnhub.fetch_market_news('general', max_items - len(articles))
            if result.success:
                articles.extend(result.articles)
        
        seen, unique = set(), []
        for a in articles:
            key = a.title.lower()[:100]
            if key not in seen:
                seen.add(key)
                unique.append(a)
                
        unique.sort(key=lambda x: self._normalize_datetime(x.published_at), reverse=True)
        unique = unique[:max_items]
        
        if self.cache and unique:
            self.cache.set(cache_key, [a.to_dict() for a in unique], 'industry')
        
        return {'success': True, 'industry': industry, 'articles': unique, 'total_count': len(unique), 'from_cache': False}
    
    def fetch_all(self, ticker: str, company_name: str = None, industry: str = None,
                  max_company_news: int = 30, max_industry_news: int = 15) -> Dict:
        """Fetch all data for comprehensive sentiment analysis."""
        result = {
            'ticker': ticker, 'company_name': company_name, 'industry': industry,
            'fetch_time': datetime.now().isoformat(),
            'company_news': None, 'industry_news': None,
            'summary': {'total_items': 0, 'sources_used': [], 'has_sentiment_scores': False}
        }
        
        company_data = self.fetch_company_news(ticker, company_name, max_company_news)
        result['company_news'] = company_data
        result['summary']['total_items'] += company_data.get('total_count', 0)
        result['summary']['sources_used'].extend(company_data.get('sources_used', []))
        if company_data.get('articles'):
            result['summary']['has_sentiment_scores'] = any(a.sentiment_hint is not None for a in company_data['articles'])
        
        if industry:
            industry_data = self.fetch_industry_news(industry, max_industry_news)
            result['industry_news'] = industry_data
            result['summary']['total_items'] += industry_data.get('total_count', 0)
        
        return result
    
    def get_formatted_for_llm(self, ticker: str, company_name: str = None, industry: str = None, max_items: int = 20) -> str:
        """Get data formatted for LLM sentiment analysis."""
        data = self.fetch_all(ticker, company_name, industry, max_company_news=max_items)
        sections = []
        
        if data['company_news'] and data['company_news'].get('articles'):
            sections.append("=== COMPANY NEWS ===")
            for i, a in enumerate(data['company_news']['articles'][:max_items], 1):
                sentiment = f" [Sentiment: {a.sentiment_hint:.2f}]" if a.sentiment_hint else ""
                sections.append(f"\n[{i}] {a.title}\nSource: {a.source} | {a.published_at:%Y-%m-%d}{sentiment}\n{a.content[:400]}...")
        
        if data['industry_news'] and data['industry_news'].get('articles'):
            sections.append("\n\n=== INDUSTRY NEWS ===")
            for i, a in enumerate(data['industry_news']['articles'][:5], 1):
                sections.append(f"\n[{i}] {a.title}\n{a.content[:300]}...")
        
        return "\n".join(sections) if sections else f"No news data found for {ticker}."
    
    def get_available_sources(self) -> Dict[str, bool]:
        return self.available_sources.copy()


# =============================================================================
# Convenience Function
# =============================================================================

def fetch_news_for_sentiment(ticker: str, company_name: str = None, max_news: int = 30) -> List[Dict]:
    """Simple function to fetch news for sentiment analysis."""
    fetcher = SentimentDataFetcher()
    result = fetcher.fetch_company_news(ticker, company_name, max_news)
    return [{'title': a.title, 'content': a.content, 'source': a.source,
             'published_at': a.published_at.isoformat(), 'url': a.url,
             'sentiment_hint': a.sentiment_hint} for a in result.get('articles', [])]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sentiment Data Fetcher')
    parser.add_argument('ticker', help='Stock ticker (e.g., AAPL)')
    parser.add_argument('--company', '-c', help='Company name')
    parser.add_argument('--industry', '-i', help='Industry sector')
    parser.add_argument('--max', '-m', type=int, default=20)
    parser.add_argument('--format', '-f', choices=['json', 'llm'], default='json')
    args = parser.parse_args()
    
    print(f"\n{'='*60}\nFetching sentiment data for {args.ticker}\n{'='*60}")
    
    fetcher = SentimentDataFetcher()
    print("\nAvailable sources:")
    for src, ok in fetcher.get_available_sources().items():
        print(f"  {'✓' if ok else '✗'} {src}")
    
    if args.format == 'llm':
        print("\n" + fetcher.get_formatted_for_llm(args.ticker, args.company, args.industry, args.max))
    else:
        data = fetcher.fetch_all(args.ticker, args.company, args.industry, max_company_news=args.max)
        print(f"\nTotal: {data['summary']['total_items']} items from {data['summary']['sources_used']}")
        print(f"Has sentiment scores: {data['summary']['has_sentiment_scores']}")
        if data['company_news'] and data['company_news'].get('articles'):
            print(f"\nCompany News ({len(data['company_news']['articles'])}):")
            for a in data['company_news']['articles'][:5]:
                score = f" [{a.sentiment_hint:.2f}]" if a.sentiment_hint else ""
                print(f"  • {a.title[:55]}...{score}")