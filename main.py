#!/usr/bin/env python3
"""
Console interface for the AI Stock Analysis Bot
Provides batch processing capabilities for multiple stocks
"""

import sys
import argparse
from datetime import datetime
import pandas as pd

from data_retrieval import DataRetrieval
from technical_analysis import TechnicalAnalysis
from fundamental_analysis import FundamentalAnalysis
from recommendation import RecommendationEngine
from database import Database
from utils import format_currency, format_percentage, validate_ticker

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🤖 AI Stock Analysis Bot - Console Interface")
    print("=" * 60)
    print()

def analyze_stock(ticker, components, period='1y', min_ta_score=50, include_fa=True):
    """Analyze a single stock"""
    db, data_retrieval, technical_analysis, fundamental_analysis, recommendation_engine = components
    
    try:
        print(f"📊 Analyzing {ticker}...")
        
        # Get stock data
        stock_data = data_retrieval.get_stock_data(ticker, period)
        if stock_data.empty:
            print(f"❌ No data available for {ticker}")
            return None
        
        # Technical analysis
        ta_results = technical_analysis.analyze_stock(stock_data, ticker)
        print(f"   TA Score: {ta_results['score']:.1f}")
        
        # Filter by TA score
        if ta_results['score'] < min_ta_score:
            print(f"   ❌ TA score below threshold ({min_ta_score})")
            return None
        
        # Fundamental analysis if enabled
        fa_results = None
        if include_fa:
            fa_results = fundamental_analysis.analyze_stock(ticker)
            if fa_results:
                print(f"   FA Score: {fa_results['score']:.1f}")
        
        # Get recommendation
        recommendation = recommendation_engine.get_recommendation(
            ticker, ta_results, fa_results
        )
        
        print(f"   Final Score: {recommendation['final_score']:.1f}")
        print(f"   Recommendation: {recommendation['action']}")
        print()
        
        return {
            'ticker': ticker,
            'data': stock_data,
            'ta_results': ta_results,
            'fa_results': fa_results,
            'recommendation': recommendation
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {str(e)}")
        return None

def print_summary(results):
    """Print analysis summary"""
    if not results:
        print("No stocks passed the analysis criteria.")
        return
    
    print("\n" + "=" * 80)
    print("📈 ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Sort by final score
    results.sort(key=lambda x: x['recommendation']['final_score'], reverse=True)
    
    print(f"{'Rank':<5} {'Ticker':<8} {'TA Score':<10} {'FA Score':<10} {'Final':<10} {'Action':<12} {'Price':<12}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        ticker = result['ticker']
        ta_score = result['ta_results']['score']
        fa_score = result['fa_results']['score'] if result['fa_results'] else 0
        final_score = result['recommendation']['final_score']
        action = result['recommendation']['action']
        price = result['data']['Close'].iloc[-1]
        
        print(f"{i:<5} {ticker:<8} {ta_score:<10.1f} {fa_score:<10.1f} {final_score:<10.1f} {action:<12} {format_currency(price):<12}")

def print_detailed_analysis(results):
    """Print detailed analysis for each stock"""
    print("\n" + "=" * 80)
    print("📊 DETAILED ANALYSIS")
    print("=" * 80)
    
    for result in results:
        ticker = result['ticker']
        data = result['data']
        ta_results = result['ta_results']
        fa_results = result['fa_results']
        recommendation = result['recommendation']
        
        print(f"\n🔍 {ticker} - {recommendation['action']}")
        print("-" * 40)
        
        # Price info
        current_price = data['Close'].iloc[-1]
        print(f"Current Price: {format_currency(current_price)}")
        
        # Technical indicators
        print(f"RSI: {ta_results['indicators'].get('RSI', [0])[-1]:.1f}")
        print(f"MACD: {ta_results['indicators'].get('MACD', [0])[-1]:.3f}")
        
        # Fundamental metrics
        if fa_results:
            metrics = fa_results['metrics']
            print(f"P/E Ratio: {metrics.get('pe_ratio', 'N/A')}")
            print(f"ROE: {format_percentage(metrics.get('roe', 0))}")
            print(f"Market Cap: {metrics.get('market_cap', 'N/A')}")
        
        # Patterns
        if ta_results['patterns']:
            print("Detected Patterns:")
            for pattern in ta_results['patterns']:
                print(f"  • {pattern}")
        
        # Reasoning
        print(f"AI Analysis: {recommendation['reasoning']}")

def main():
    parser = argparse.ArgumentParser(description='AI Stock Analysis Bot - Console Interface')
    parser.add_argument('tickers', nargs='*', help='Stock tickers to analyze')
    parser.add_argument('--period', default='1y', choices=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                       help='Time period for analysis (default: 1y)')
    parser.add_argument('--min-ta-score', type=int, default=50,
                       help='Minimum technical analysis score (default: 50)')
    parser.add_argument('--no-fa', action='store_true',
                       help='Skip fundamental analysis')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed analysis')
    parser.add_argument('--file', help='Read tickers from file (one per line)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Get tickers
    tickers = []
    if args.file:
        try:
            with open(args.file, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            return 1
    else:
        tickers = [ticker.upper() for ticker in args.tickers]
    
    if not tickers:
        print("❌ No tickers provided. Use --help for usage information.")
        return 1
    
    # Validate tickers
    valid_tickers = []
    for ticker in tickers:
        if validate_ticker(ticker):
            valid_tickers.append(ticker)
        else:
            print(f"⚠️  Invalid ticker: {ticker}")
    
    if not valid_tickers:
        print("❌ No valid tickers provided.")
        return 1
    
    # Initialize components
    print("🔧 Initializing components...")
    db = Database()
    data_retrieval = DataRetrieval(db)
    technical_analysis = TechnicalAnalysis()
    fundamental_analysis = FundamentalAnalysis()
    recommendation_engine = RecommendationEngine()
    
    components = (db, data_retrieval, technical_analysis, fundamental_analysis, recommendation_engine)
    
    # Analyze stocks
    print(f"🚀 Analyzing {len(valid_tickers)} stocks...")
    print(f"Parameters: Period={args.period}, Min TA Score={args.min_ta_score}, FA={'No' if args.no_fa else 'Yes'}")
    print()
    
    results = []
    for ticker in valid_tickers:
        result = analyze_stock(
            ticker, 
            components, 
            period=args.period,
            min_ta_score=args.min_ta_score,
            include_fa=not args.no_fa
        )
        if result:
            results.append(result)
    
    # Print results
    print_summary(results)
    
    if args.detailed:
        print_detailed_analysis(results)
    
    print(f"\n✅ Analysis complete. Processed {len(results)} stocks.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
