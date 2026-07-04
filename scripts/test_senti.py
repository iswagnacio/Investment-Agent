"""
Simple test for sentiment_data_fetcher.py
Run: python test_fetcher.py
"""

from sentiment import SentimentDataFetcher

# Initialize
fetcher = SentimentDataFetcher()

# Check available sources
print("=" * 50)
print("Available Data Sources:")
print("=" * 50)
for source, available in fetcher.get_available_sources().items():
    status = "✓ Ready" if available else "✗ No API key"
    print(f"  {source}: {status}")

# Test with Apple stock (Yahoo Finance works without API key)
print("\n" + "=" * 50)
print("Fetching news for AAPL (Apple Inc)...")
print("=" * 50)

data = fetcher.fetch_all(
    ticker="AAPL",
    company_name="Apple Inc",
    industry="Technology"
)

print(f"\nTotal items fetched: {data['summary']['total_items']}")
print(f"Sources used: {data['summary']['sources_used']}")
print(f"Has sentiment scores: {data['summary']['has_sentiment_scores']}")

# Show sample articles
if data['company_news'] and data['company_news'].get('articles'):
    print(f"\nCompany News ({len(data['company_news']['articles'])} articles):")
    for i, article in enumerate(data['company_news']['articles'][:3], 1):
        sentiment = f" [Sentiment: {article.sentiment_hint:.2f}]" if article.sentiment_hint else ""
        print(f"\n  [{i}] {article.title[:70]}...{sentiment}")
        print(f"      Source: {article.source} | {article.published_at.strftime('%Y-%m-%d')}")

if data['industry_news'] and data['industry_news'].get('articles'):
    print(f"\nIndustry News ({len(data['industry_news']['articles'])} articles):")
    for i, article in enumerate(data['industry_news']['articles'][:2], 1):
        print(f"\n  [{i}] {article.title[:70]}...")

print("\n" + "=" * 50)
print("Test completed!")
print("=" * 50)