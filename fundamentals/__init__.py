"""
Fundamental Analysis Module
Handles SEC EDGAR data fetching, XBRL parsing, and financial metrics calculation
"""

from .analyzer import IntelligentAnalyzer, IntelligentAnalysis, MetricResult
from .edgar_fetch import EdgarFetcher
from .xbrl_parse import XBRLParser, FinancialStatements
from .metrics import METRICS_CATALOG, Sector, get_sector_from_sic

__all__ = [
    "IntelligentAnalyzer",
    "IntelligentAnalysis",
    "MetricResult",
    "EdgarFetcher",
    "XBRLParser",
    "FinancialStatements",
    "METRICS_CATALOG",
    "Sector",
    "get_sector_from_sic",
]
