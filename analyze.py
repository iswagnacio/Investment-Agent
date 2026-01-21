#!/usr/bin/env python3
"""
Main Entry Point for Investment Analysis Platform
Multi-agent system powered by LangGraph and Anthropic Claude
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import analyze_stock
from config import config


def main():
    """Main entry point for the investment analysis system"""

    parser = argparse.ArgumentParser(
        description="Multi-Agent Investment Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Apple stock
  python analyze.py AAPL

  # Analyze with current stock price
  python analyze.py TSLA --price 250.50

  # Focus on specific areas
  python analyze.py MSFT --focus profitability growth
        """
    )

    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol to analyze"
    )

    parser.add_argument(
        "--price",
        type=float,
        help="Current stock price (optional, for valuation metrics)",
        default=None
    )

    parser.add_argument(
        "--focus",
        nargs="+",
        help="Focus areas for analysis (e.g., profitability growth debt)",
        default=None
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Validate configuration
    if not config.validate():
        print("\nError: Configuration validation failed.")
        print("Please ensure ANTHROPIC_API_KEY is set in your .env file.")
        return 1

    # Print header
    print(f"\n{'='*70}")
    print(f"  MULTI-AGENT INVESTMENT ANALYSIS SYSTEM")
    print(f"{'='*70}")
    print(f"\nAnalyzing: {args.ticker.upper()}")
    if args.price:
        print(f"Stock Price: ${args.price:.2f}")
    if args.focus:
        print(f"Focus Areas: {', '.join(args.focus)}")
    print()

    try:
        # Run the analysis
        result = analyze_stock(
            ticker=args.ticker,
            stock_price=args.price,
            focus_areas=args.focus
        )

        # Print the synthesis report
        if result.get("synthesis_report"):
            print(result["synthesis_report"])
        else:
            print("\nNo synthesis report generated.")

        # Print any errors
        if result.get("errors"):
            print("\n" + "="*70)
            print("ERRORS ENCOUNTERED:")
            print("="*70)
            for error in result["errors"]:
                print(f"  ⚠  {error}")

        # Print timing information
        if result.get("analysis_start_time") and result.get("analysis_end_time"):
            duration = result["analysis_end_time"] - result["analysis_start_time"]
            print(f"\nAnalysis completed in {duration.total_seconds():.1f} seconds")

        return 0

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        return 130

    except Exception as e:
        print(f"\n\nError: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
