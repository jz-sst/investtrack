#!/usr/bin/env python3
"""
Live Analysis Demo - Real-time stock analysis with current market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def perform_live_analysis(ticker="AAPL"):
    """Perform comprehensive live analysis on a stock"""
    print(f"🔍 Live Analysis for {ticker}")
    print("="*50)
    
    # Get live data
    stock = yf.Ticker(ticker)
    info = stock.info
    data = stock.history(period='1y')
    
    if data.empty:
        print(f"❌ No data available for {ticker}")
        return None
    
    # Company Information
    print(f"📊 Company: {info.get('longName', ticker)}")
    print(f"💰 Current Price: ${info.get('currentPrice', data['Close'].iloc[-1]):.2f}")
    print(f"📈 Market Cap: ${info.get('marketCap', 0):,}")
    print(f"📊 Volume: {info.get('volume', 0):,}")
    print(f"📈 P/E Ratio: {info.get('trailingPE', 'N/A')}")
    
    # Technical Analysis
    print("\n🔧 Technical Analysis:")
    
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # Current values
    current = data.iloc[-1]
    
    print(f"   20-day SMA: ${current['SMA_20']:.2f}")
    print(f"   50-day SMA: ${current['SMA_50']:.2f}")
    print(f"   RSI: {current['RSI']:.2f}")
    print(f"   MACD: {current['MACD']:.4f}")
    print(f"   Signal: {current['Signal']:.4f}")
    
    # Generate signals
    signals = []
    score = 0
    
    # Trend Analysis
    if current['Close'] > current['SMA_20'] > current['SMA_50']:
        signals.append("✅ Strong Uptrend")
        score += 30
    elif current['Close'] > current['SMA_20']:
        signals.append("🟡 Weak Uptrend")
        score += 15
    elif current['Close'] < current['SMA_20'] < current['SMA_50']:
        signals.append("❌ Strong Downtrend")
        score -= 30
    else:
        signals.append("⚠️ Sideways/Mixed")
        score += 5
    
    # RSI Analysis
    if 30 <= current['RSI'] <= 70:
        signals.append("✅ RSI Neutral Zone")
        score += 10
    elif current['RSI'] < 30:
        signals.append("🟢 RSI Oversold - Buy Signal")
        score += 20
    else:
        signals.append("⚠️ RSI Overbought")
        score -= 10
    
    # MACD Analysis
    if current['MACD'] > current['Signal']:
        signals.append("🟢 MACD Bullish")
        score += 15
    else:
        signals.append("❌ MACD Bearish")
        score -= 15
    
    # Bollinger Bands
    if current['Close'] < current['BB_Lower']:
        signals.append("🟢 Below Lower BB - Buy Signal")
        score += 20
    elif current['Close'] > current['BB_Upper']:
        signals.append("⚠️ Above Upper BB - Sell Signal")
        score -= 20
    else:
        signals.append("✅ Within BB Range")
        score += 5
    
    # Volume Analysis
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    if current['Volume'] > avg_volume * 1.5:
        signals.append("🟢 High Volume - Strong Signal")
        score += 10
    elif current['Volume'] < avg_volume * 0.5:
        signals.append("⚠️ Low Volume - Weak Signal")
        score -= 5
    
    print("\n📊 Technical Signals:")
    for signal in signals:
        print(f"   {signal}")
    
    # Final Recommendation
    print(f"\n🎯 Technical Score: {score}/100")
    
    if score >= 70:
        recommendation = "STRONG BUY"
        confidence = "High"
    elif score >= 40:
        recommendation = "BUY"
        confidence = "Medium"
    elif score >= 10:
        recommendation = "HOLD"
        confidence = "Medium"
    elif score >= -20:
        recommendation = "WEAK SELL"
        confidence = "Low"
    else:
        recommendation = "SELL"
        confidence = "High"
    
    print(f"🎯 Recommendation: {recommendation}")
    print(f"📊 Confidence: {confidence}")
    
    # Fundamental Analysis
    print("\n💼 Fundamental Analysis:")
    
    fundamental_score = 0
    
    # P/E Ratio
    pe_ratio = info.get('trailingPE', 0)
    if pe_ratio and 10 <= pe_ratio <= 25:
        print(f"   ✅ P/E Ratio: {pe_ratio:.2f} (Good)")
        fundamental_score += 20
    elif pe_ratio and pe_ratio > 25:
        print(f"   ⚠️ P/E Ratio: {pe_ratio:.2f} (High)")
        fundamental_score += 5
    elif pe_ratio:
        print(f"   🟢 P/E Ratio: {pe_ratio:.2f} (Low)")
        fundamental_score += 15
    
    # Profit Margins
    profit_margin = info.get('profitMargins', 0)
    if profit_margin and profit_margin > 0.15:
        print(f"   ✅ Profit Margin: {profit_margin:.2%} (Excellent)")
        fundamental_score += 20
    elif profit_margin and profit_margin > 0.05:
        print(f"   🟡 Profit Margin: {profit_margin:.2%} (Good)")
        fundamental_score += 10
    elif profit_margin:
        print(f"   ❌ Profit Margin: {profit_margin:.2%} (Poor)")
        fundamental_score -= 10
    
    # ROE
    roe = info.get('returnOnEquity', 0)
    if roe and roe > 0.15:
        print(f"   ✅ ROE: {roe:.2%} (Excellent)")
        fundamental_score += 20
    elif roe and roe > 0.05:
        print(f"   🟡 ROE: {roe:.2%} (Good)")
        fundamental_score += 10
    elif roe:
        print(f"   ❌ ROE: {roe:.2%} (Poor)")
        fundamental_score -= 10
    
    # Debt to Equity
    debt_to_equity = info.get('debtToEquity', 0)
    if debt_to_equity and debt_to_equity < 50:
        print(f"   ✅ Debt/Equity: {debt_to_equity:.2f} (Low)")
        fundamental_score += 15
    elif debt_to_equity and debt_to_equity < 100:
        print(f"   🟡 Debt/Equity: {debt_to_equity:.2f} (Moderate)")
        fundamental_score += 5
    elif debt_to_equity:
        print(f"   ❌ Debt/Equity: {debt_to_equity:.2f} (High)")
        fundamental_score -= 15
    
    print(f"\n🎯 Fundamental Score: {fundamental_score}/100")
    
    # Combined Analysis
    combined_score = (score * 0.7) + (fundamental_score * 0.3)
    print(f"\n🎯 Combined Score: {combined_score:.1f}/100")
    
    if combined_score >= 70:
        final_recommendation = "STRONG BUY"
        final_confidence = "High"
    elif combined_score >= 50:
        final_recommendation = "BUY"
        final_confidence = "Medium"
    elif combined_score >= 30:
        final_recommendation = "HOLD"
        final_confidence = "Medium"
    elif combined_score >= 10:
        final_recommendation = "WEAK SELL"
        final_confidence = "Low"
    else:
        final_recommendation = "SELL"
        final_confidence = "High"
    
    print(f"🎯 Final Recommendation: {final_recommendation}")
    print(f"📊 Final Confidence: {final_confidence}")
    
    # Risk Assessment
    print("\n⚠️ Risk Assessment:")
    risk_factors = []
    
    if current['RSI'] > 70:
        risk_factors.append("Overbought conditions")
    if current['Volume'] < avg_volume * 0.5:
        risk_factors.append("Low trading volume")
    if pe_ratio and pe_ratio > 30:
        risk_factors.append("High P/E ratio")
    if debt_to_equity and debt_to_equity > 100:
        risk_factors.append("High debt levels")
    
    if risk_factors:
        for risk in risk_factors:
            print(f"   ⚠️ {risk}")
    else:
        print("   ✅ No major risk factors identified")
    
    # Current market data timestamp
    print(f"\n🕐 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Data Points: {len(data)} days")
    print(f"📈 Latest Data: {data.index[-1].strftime('%Y-%m-%d')}")
    
    return {
        'ticker': ticker,
        'technical_score': score,
        'fundamental_score': fundamental_score,
        'combined_score': combined_score,
        'recommendation': final_recommendation,
        'confidence': final_confidence,
        'signals': signals,
        'risk_factors': risk_factors,
        'current_price': current['Close'],
        'analysis_time': datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Test with multiple stocks
    test_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    print("🚀 Live Stock Analysis Demo")
    print("=" * 80)
    
    for ticker in test_stocks:
        result = perform_live_analysis(ticker)
        print("\n" + "="*80 + "\n")
        
        if len(test_stocks) > 1:
            input("Press Enter to analyze next stock...")