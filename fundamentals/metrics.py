"""
Metrics Catalog - Defines all available financial metrics with their formulas.

Each metric includes:
- Human-readable description
- Formula as a lambda (for calculation)
- Required inputs (XBRL concepts needed)
- Sector relevance (which sectors care about this metric)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from enum import Enum


class Sector(Enum):
    """Industry sectors with distinct analysis needs."""
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"  # Banks, insurance
    HEALTHCARE = "healthcare"
    CONSUMER_CYCLICAL = "consumer_cyclical"  # Retail, auto
    CONSUMER_DEFENSIVE = "consumer_defensive"  # Food, household
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    MATERIALS = "materials"
    COMMUNICATION = "communication"
    UNKNOWN = "unknown"


# Map SIC codes to sectors (simplified)
SIC_TO_SECTOR = {
    # Technology
    "3571": Sector.TECHNOLOGY,  # Electronic Computers
    "3572": Sector.TECHNOLOGY,  # Computer Storage
    "3661": Sector.TECHNOLOGY,  # Telephone Equipment
    "3674": Sector.TECHNOLOGY,  # Semiconductors
    "7370": Sector.TECHNOLOGY,  # Computer Programming
    "7371": Sector.TECHNOLOGY,  # Computer Programming Services
    "7372": Sector.TECHNOLOGY,  # Prepackaged Software
    "7373": Sector.TECHNOLOGY,  # Computer Integrated Systems
    "7374": Sector.TECHNOLOGY,  # Computer Processing
    
    # Financials
    "6020": Sector.FINANCIALS,  # Commercial Banks
    "6021": Sector.FINANCIALS,  # National Commercial Banks
    "6022": Sector.FINANCIALS,  # State Commercial Banks
    "6035": Sector.FINANCIALS,  # Savings Institutions
    "6141": Sector.FINANCIALS,  # Personal Credit
    "6211": Sector.FINANCIALS,  # Security Brokers
    "6282": Sector.FINANCIALS,  # Investment Advice
    "6311": Sector.FINANCIALS,  # Life Insurance
    "6331": Sector.FINANCIALS,  # Fire Insurance
    
    # Healthcare
    "2834": Sector.HEALTHCARE,  # Pharmaceutical Preparations
    "2836": Sector.HEALTHCARE,  # Biological Products
    "3841": Sector.HEALTHCARE,  # Surgical Instruments
    "3845": Sector.HEALTHCARE,  # Electromedical Equipment
    "8011": Sector.HEALTHCARE,  # Offices of Doctors
    "8062": Sector.HEALTHCARE,  # General Medical Hospitals
    
    # Consumer Cyclical
    "5311": Sector.CONSUMER_CYCLICAL,  # Department Stores
    "5331": Sector.CONSUMER_CYCLICAL,  # Variety Stores
    "5411": Sector.CONSUMER_CYCLICAL,  # Grocery Stores
    "5812": Sector.CONSUMER_CYCLICAL,  # Eating Places
    "5912": Sector.CONSUMER_CYCLICAL,  # Drug Stores
    "5961": Sector.CONSUMER_CYCLICAL,  # Catalog Retail
    "7011": Sector.CONSUMER_CYCLICAL,  # Hotels
    
    # Consumer Defensive
    "2000": Sector.CONSUMER_DEFENSIVE,  # Food Products
    "2080": Sector.CONSUMER_DEFENSIVE,  # Beverages
    "2111": Sector.CONSUMER_DEFENSIVE,  # Cigarettes
    
    # Energy
    "1311": Sector.ENERGY,  # Crude Petroleum
    "2911": Sector.ENERGY,  # Petroleum Refining
    "4922": Sector.ENERGY,  # Natural Gas Transmission
    "4923": Sector.ENERGY,  # Natural Gas Distribution
    
    # Utilities
    "4911": Sector.UTILITIES,  # Electric Services
    "4924": Sector.UTILITIES,  # Natural Gas Distribution
    "4931": Sector.UTILITIES,  # Electric and Other Services
    
    # Real Estate
    "6500": Sector.REAL_ESTATE,  # Real Estate
    "6510": Sector.REAL_ESTATE,  # Real Estate Operators
    "6798": Sector.REAL_ESTATE,  # REITs
    
    # Communication
    "4812": Sector.COMMUNICATION,  # Radiotelephone
    "4813": Sector.COMMUNICATION,  # Telephone Communications
    "4841": Sector.COMMUNICATION,  # Cable TV
    "7941": Sector.COMMUNICATION,  # Sports Clubs
}


@dataclass
class MetricDefinition:
    """Definition of a financial metric."""
    name: str
    description: str
    formula_description: str
    required_inputs: list[str]  # XBRL concept names needed
    calculate: Callable[[dict[str, float]], Optional[float]]
    unit: str = "ratio"  # ratio, percent, currency, days
    sectors: list[Sector] = field(default_factory=list)  # Empty = all sectors
    priority: int = 1  # 1 = always show, 2 = sector-specific, 3 = nice-to-have


# --- Metric Calculation Functions ---

def safe_divide(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely divide, returning None if invalid."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def safe_subtract(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely subtract, returning None if invalid."""
    if a is None or b is None:
        return None
    return a - b


# --- Metric Definitions Catalog ---

METRICS_CATALOG: dict[str, MetricDefinition] = {
    # ============ UNIVERSAL METRICS (Priority 1) ============
    
    "gross_margin": MetricDefinition(
        name="Gross Margin",
        description="Percentage of revenue retained after cost of goods sold",
        formula_description="gross_profit / revenue",
        required_inputs=["gross_profit", "revenue"],
        calculate=lambda d: safe_divide(d.get("gross_profit"), d.get("revenue")),
        unit="percent",
        priority=1,
    ),
    
    "operating_margin": MetricDefinition(
        name="Operating Margin",
        description="Percentage of revenue retained after operating expenses",
        formula_description="operating_income / revenue",
        required_inputs=["operating_income", "revenue"],
        calculate=lambda d: safe_divide(d.get("operating_income"), d.get("revenue")),
        unit="percent",
        priority=1,
    ),
    
    "net_margin": MetricDefinition(
        name="Net Profit Margin",
        description="Percentage of revenue retained as net income",
        formula_description="net_income / revenue",
        required_inputs=["net_income", "revenue"],
        calculate=lambda d: safe_divide(d.get("net_income"), d.get("revenue")),
        unit="percent",
        priority=1,
    ),
    
    "roe": MetricDefinition(
        name="Return on Equity (ROE)",
        description="How efficiently the company uses shareholder equity to generate profit",
        formula_description="net_income / stockholders_equity",
        required_inputs=["net_income", "stockholders_equity"],
        calculate=lambda d: safe_divide(d.get("net_income"), d.get("stockholders_equity")),
        unit="percent",
        priority=1,
    ),
    
    "roa": MetricDefinition(
        name="Return on Assets (ROA)",
        description="How efficiently the company uses assets to generate profit",
        formula_description="net_income / total_assets",
        required_inputs=["net_income", "total_assets"],
        calculate=lambda d: safe_divide(d.get("net_income"), d.get("total_assets")),
        unit="percent",
        priority=1,
    ),
    
    "debt_to_equity": MetricDefinition(
        name="Debt to Equity",
        description="Financial leverage - how much debt vs equity",
        formula_description="total_debt / stockholders_equity",
        required_inputs=["total_debt", "stockholders_equity"],
        calculate=lambda d: safe_divide(d.get("total_debt"), d.get("stockholders_equity")),
        unit="ratio",
        priority=1,
    ),
    
    "current_ratio": MetricDefinition(
        name="Current Ratio",
        description="Ability to pay short-term obligations",
        formula_description="current_assets / current_liabilities",
        required_inputs=["current_assets", "current_liabilities"],
        calculate=lambda d: safe_divide(d.get("current_assets"), d.get("current_liabilities")),
        unit="ratio",
        priority=1,
    ),
    
    # ============ TECHNOLOGY SECTOR (Priority 2) ============
    
    "rd_ratio": MetricDefinition(
        name="R&D to Revenue",
        description="Investment in research and development relative to sales",
        formula_description="rd_expense / revenue",
        required_inputs=["rd_expense", "revenue"],
        calculate=lambda d: safe_divide(d.get("rd_expense"), d.get("revenue")),
        unit="percent",
        sectors=[Sector.TECHNOLOGY, Sector.HEALTHCARE],
        priority=2,
    ),
    
    "revenue_per_employee": MetricDefinition(
        name="Revenue per Employee",
        description="Productivity measure - revenue generated per employee",
        formula_description="revenue / employee_count",
        required_inputs=["revenue", "employee_count"],
        calculate=lambda d: safe_divide(d.get("revenue"), d.get("employee_count")),
        unit="currency",
        sectors=[Sector.TECHNOLOGY],
        priority=2,
    ),
    
    "rule_of_40": MetricDefinition(
        name="Rule of 40",
        description="SaaS health metric: revenue growth % + profit margin % should exceed 40",
        formula_description="revenue_growth_rate + operating_margin",
        required_inputs=["revenue_growth_rate", "operating_margin"],
        calculate=lambda d: (
            (d.get("revenue_growth_rate") or 0) + (d.get("operating_margin") or 0)
            if d.get("revenue_growth_rate") is not None or d.get("operating_margin") is not None
            else None
        ),
        unit="percent",
        sectors=[Sector.TECHNOLOGY],
        priority=2,
    ),
    
    # ============ FINANCIALS SECTOR (Priority 2) ============
    
    "net_interest_margin": MetricDefinition(
        name="Net Interest Margin (NIM)",
        description="Bank profitability - difference between interest earned and paid",
        formula_description="(interest_income - interest_expense) / average_earning_assets",
        required_inputs=["interest_income", "interest_expense", "total_assets"],
        calculate=lambda d: safe_divide(
            safe_subtract(d.get("interest_income"), d.get("interest_expense")),
            d.get("total_assets")
        ),
        unit="percent",
        sectors=[Sector.FINANCIALS],
        priority=2,
    ),
    
    "efficiency_ratio": MetricDefinition(
        name="Efficiency Ratio",
        description="Bank operating efficiency - lower is better",
        formula_description="non_interest_expense / (net_interest_income + non_interest_income)",
        required_inputs=["non_interest_expense", "net_interest_income", "non_interest_income"],
        calculate=lambda d: safe_divide(
            d.get("non_interest_expense"),
            (d.get("net_interest_income") or 0) + (d.get("non_interest_income") or 0)
            if (d.get("net_interest_income") or d.get("non_interest_income")) else None
        ),
        unit="percent",
        sectors=[Sector.FINANCIALS],
        priority=2,
    ),
    
    "tier1_capital_ratio": MetricDefinition(
        name="Tier 1 Capital Ratio",
        description="Bank capital adequacy - regulatory requirement",
        formula_description="tier1_capital / risk_weighted_assets",
        required_inputs=["tier1_capital", "risk_weighted_assets"],
        calculate=lambda d: safe_divide(d.get("tier1_capital"), d.get("risk_weighted_assets")),
        unit="percent",
        sectors=[Sector.FINANCIALS],
        priority=2,
    ),
    
    # ============ RETAIL / CONSUMER (Priority 2) ============
    
    "inventory_turnover": MetricDefinition(
        name="Inventory Turnover",
        description="How quickly inventory is sold - higher is generally better",
        formula_description="cost_of_revenue / average_inventory",
        required_inputs=["cost_of_revenue", "inventory"],
        calculate=lambda d: safe_divide(d.get("cost_of_revenue"), d.get("inventory")),
        unit="ratio",
        sectors=[Sector.CONSUMER_CYCLICAL, Sector.CONSUMER_DEFENSIVE],
        priority=2,
    ),
    
    "days_inventory": MetricDefinition(
        name="Days Inventory Outstanding",
        description="Average days to sell inventory",
        formula_description="365 / inventory_turnover",
        required_inputs=["cost_of_revenue", "inventory"],
        calculate=lambda d: safe_divide(365, safe_divide(d.get("cost_of_revenue"), d.get("inventory"))),
        unit="days",
        sectors=[Sector.CONSUMER_CYCLICAL, Sector.CONSUMER_DEFENSIVE],
        priority=2,
    ),
    
    "same_store_sales": MetricDefinition(
        name="Same-Store Sales Growth",
        description="Revenue growth from existing stores (excludes new stores)",
        formula_description="Reported directly by company",
        required_inputs=["same_store_sales_growth"],
        calculate=lambda d: d.get("same_store_sales_growth"),
        unit="percent",
        sectors=[Sector.CONSUMER_CYCLICAL],
        priority=2,
    ),
    
    # ============ ENERGY SECTOR (Priority 2) ============
    
    "reserve_replacement_ratio": MetricDefinition(
        name="Reserve Replacement Ratio",
        description="Ability to replace produced reserves with new discoveries",
        formula_description="reserves_added / reserves_produced",
        required_inputs=["reserves_added", "reserves_produced"],
        calculate=lambda d: safe_divide(d.get("reserves_added"), d.get("reserves_produced")),
        unit="ratio",
        sectors=[Sector.ENERGY],
        priority=2,
    ),
    
    "finding_cost": MetricDefinition(
        name="Finding & Development Cost",
        description="Cost to find and develop new reserves per barrel",
        formula_description="exploration_cost / reserves_added",
        required_inputs=["exploration_cost", "reserves_added"],
        calculate=lambda d: safe_divide(d.get("exploration_cost"), d.get("reserves_added")),
        unit="currency",
        sectors=[Sector.ENERGY],
        priority=2,
    ),
    
    # ============ REAL ESTATE / REITS (Priority 2) ============
    
    "ffo": MetricDefinition(
        name="Funds From Operations (FFO)",
        description="REIT cash flow measure - adds back depreciation to net income",
        formula_description="net_income + depreciation - gains_on_sale",
        required_inputs=["net_income", "depreciation", "gains_on_sale"],
        calculate=lambda d: (
            (d.get("net_income") or 0) + 
            (d.get("depreciation") or 0) - 
            (d.get("gains_on_sale") or 0)
        ) if d.get("net_income") is not None else None,
        unit="currency",
        sectors=[Sector.REAL_ESTATE],
        priority=2,
    ),
    
    "occupancy_rate": MetricDefinition(
        name="Occupancy Rate",
        description="Percentage of rentable space that is leased",
        formula_description="Reported directly by company",
        required_inputs=["occupancy_rate"],
        calculate=lambda d: d.get("occupancy_rate"),
        unit="percent",
        sectors=[Sector.REAL_ESTATE],
        priority=2,
    ),
    
    # ============ HEALTHCARE (Priority 2) ============
    
    "pipeline_value": MetricDefinition(
        name="Pipeline NPV",
        description="Net present value of drug pipeline",
        formula_description="Analyst estimate - not from financials",
        required_inputs=["pipeline_npv"],
        calculate=lambda d: d.get("pipeline_npv"),
        unit="currency",
        sectors=[Sector.HEALTHCARE],
        priority=3,
    ),
    
    # ============ UTILITIES (Priority 2) ============
    
    "payout_ratio": MetricDefinition(
        name="Dividend Payout Ratio",
        description="Percentage of earnings paid as dividends",
        formula_description="dividends_paid / net_income",
        required_inputs=["dividends_paid", "net_income"],
        calculate=lambda d: safe_divide(abs(d.get("dividends_paid") or 0), d.get("net_income")),
        unit="percent",
        sectors=[Sector.UTILITIES, Sector.REAL_ESTATE],
        priority=2,
    ),
    
    "regulatory_asset_base": MetricDefinition(
        name="Regulatory Asset Base Growth",
        description="Growth in regulated assets that earn allowed returns",
        formula_description="Reported directly by company",
        required_inputs=["regulatory_asset_base", "regulatory_asset_base_prior"],
        calculate=lambda d: safe_divide(
            safe_subtract(d.get("regulatory_asset_base"), d.get("regulatory_asset_base_prior")),
            d.get("regulatory_asset_base_prior")
        ),
        unit="percent",
        sectors=[Sector.UTILITIES],
        priority=2,
    ),
    
    # ============ VALUATION METRICS (Priority 1, need price) ============
    
    "pe_ratio": MetricDefinition(
        name="P/E Ratio",
        description="Price relative to earnings per share",
        formula_description="stock_price / eps",
        required_inputs=["stock_price", "eps"],
        calculate=lambda d: safe_divide(d.get("stock_price"), d.get("eps")),
        unit="ratio",
        priority=1,
    ),
    
    "pb_ratio": MetricDefinition(
        name="P/B Ratio",
        description="Price relative to book value per share",
        formula_description="stock_price / book_value_per_share",
        required_inputs=["stock_price", "stockholders_equity", "shares_outstanding"],
        calculate=lambda d: safe_divide(
            d.get("stock_price"),
            safe_divide(d.get("stockholders_equity"), d.get("shares_outstanding"))
        ),
        unit="ratio",
        priority=1,
    ),
    
    "ps_ratio": MetricDefinition(
        name="P/S Ratio",
        description="Price relative to revenue per share",
        formula_description="market_cap / revenue",
        required_inputs=["stock_price", "shares_outstanding", "revenue"],
        calculate=lambda d: safe_divide(
            (d.get("stock_price") or 0) * (d.get("shares_outstanding") or 0),
            d.get("revenue")
        ) if d.get("stock_price") and d.get("shares_outstanding") else None,
        unit="ratio",
        priority=1,
    ),
    
    "ev_to_ebitda": MetricDefinition(
        name="EV/EBITDA",
        description="Enterprise value relative to operating earnings",
        formula_description="(market_cap + debt - cash) / ebitda",
        required_inputs=["stock_price", "shares_outstanding", "total_debt", "cash_and_equivalents", "ebitda"],
        calculate=lambda d: safe_divide(
            ((d.get("stock_price") or 0) * (d.get("shares_outstanding") or 0) +
             (d.get("total_debt") or 0) - (d.get("cash_and_equivalents") or 0)),
            d.get("ebitda")
        ) if d.get("stock_price") and d.get("shares_outstanding") else None,
        unit="ratio",
        priority=1,
    ),
    
    "free_cash_flow_yield": MetricDefinition(
        name="Free Cash Flow Yield",
        description="Free cash flow relative to market cap",
        formula_description="free_cash_flow / market_cap",
        required_inputs=["operating_cash_flow", "capital_expenditures", "stock_price", "shares_outstanding"],
        calculate=lambda d: safe_divide(
            (d.get("operating_cash_flow") or 0) - abs(d.get("capital_expenditures") or 0),
            (d.get("stock_price") or 0) * (d.get("shares_outstanding") or 0)
        ) if d.get("stock_price") and d.get("shares_outstanding") else None,
        unit="percent",
        priority=2,
    ),
}


def get_sector_from_sic(sic_code: str) -> Sector:
    """Map SIC code to sector."""
    return SIC_TO_SECTOR.get(sic_code, Sector.UNKNOWN)


def get_metrics_for_sector(sector: Sector, include_universal: bool = True) -> list[str]:
    """Get list of relevant metric names for a sector."""
    metrics = []
    
    for name, defn in METRICS_CATALOG.items():
        # Include if universal (empty sectors list) or sector matches
        if not defn.sectors or sector in defn.sectors:
            if include_universal or defn.sectors:  # Skip universal if not wanted
                metrics.append(name)
    
    # Sort by priority
    metrics.sort(key=lambda m: METRICS_CATALOG[m].priority)
    return metrics


def get_all_required_inputs(metric_names: list[str]) -> set[str]:
    """Get all XBRL inputs needed for a list of metrics."""
    inputs = set()
    for name in metric_names:
        if name in METRICS_CATALOG:
            inputs.update(METRICS_CATALOG[name].required_inputs)
    return inputs