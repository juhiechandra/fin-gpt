import yfinance as yf
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from api.logger import api_logger, error_logger

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = "0WNQ0H05ZSWX1X2G"


class StockDataFetcher:
    """
    A comprehensive class for fetching and analyzing stock market data
    """

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key or ALPHA_VANTAGE_API_KEY

    def search_stock_symbols(self, query: str) -> List[Dict]:
        """
        Search for stock symbols based on company name or ticker
        """
        try:
            # Using yfinance to search for tickers
            # This is a simple implementation - you could enhance it with more APIs
            ticker = yf.Ticker(query.upper())
            info = ticker.info

            if info and "symbol" in info:
                return [
                    {
                        "symbol": info.get("symbol", query.upper()),
                        "name": info.get("longName", "Unknown"),
                        "sector": info.get("sector", "Unknown"),
                        "industry": info.get("industry", "Unknown"),
                        "exchange": info.get("exchange", "Unknown"),
                    }
                ]
            else:
                # Fallback: return the query as symbol
                return [
                    {
                        "symbol": query.upper(),
                        "name": f"Stock: {query.upper()}",
                        "sector": "Unknown",
                        "industry": "Unknown",
                        "exchange": "Unknown",
                    }
                ]

        except Exception as e:
            error_logger.error(
                f"Error searching stock symbols: {str(e)}", exc_info=True
            )
            return [
                {
                    "symbol": query.upper(),
                    "name": f"Stock: {query.upper()}",
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "exchange": "Unknown",
                }
            ]

    def get_stock_data(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> Dict:
        """
        Fetch comprehensive stock data for a given symbol
        """
        try:
            ticker = yf.Ticker(symbol)

            # Get historical data
            hist_data = ticker.history(period=period, interval=interval)

            if hist_data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Get stock info
            info = ticker.info

            # Get current price and basic metrics
            current_price = hist_data["Close"].iloc[-1] if not hist_data.empty else None
            previous_close = (
                hist_data["Close"].iloc[-2] if len(hist_data) > 1 else current_price
            )

            change = (
                current_price - previous_close
                if current_price and previous_close
                else 0
            )
            change_percent = (change / previous_close * 100) if previous_close else 0

            # Calculate additional metrics
            high_52w = hist_data["High"].max() if len(hist_data) > 50 else current_price
            low_52w = hist_data["Low"].min() if len(hist_data) > 50 else current_price

            # Calculate moving averages
            ma_20 = (
                hist_data["Close"].rolling(window=20).mean().iloc[-1]
                if len(hist_data) >= 20
                else None
            )
            ma_50 = (
                hist_data["Close"].rolling(window=50).mean().iloc[-1]
                if len(hist_data) >= 50
                else None
            )
            ma_200 = (
                hist_data["Close"].rolling(window=200).mean().iloc[-1]
                if len(hist_data) >= 200
                else None
            )

            # Calculate RSI
            rsi = (
                self.calculate_rsi(hist_data["Close"]) if len(hist_data) >= 14 else None
            )

            # Volume analysis
            avg_volume = hist_data["Volume"].mean() if not hist_data.empty else 0
            current_volume = hist_data["Volume"].iloc[-1] if not hist_data.empty else 0

            return {
                "symbol": symbol.upper(),
                "company_name": info.get("longName", symbol.upper()),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "current_price": float(current_price) if current_price else 0,
                "change": float(change),
                "change_percent": float(change_percent),
                "volume": int(current_volume),
                "avg_volume": int(avg_volume),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", None),
                "dividend_yield": info.get("dividendYield", None),
                "high_52w": float(high_52w) if high_52w else 0,
                "low_52w": float(low_52w) if low_52w else 0,
                "ma_20": float(ma_20) if ma_20 else None,
                "ma_50": float(ma_50) if ma_50 else None,
                "ma_200": float(ma_200) if ma_200 else None,
                "rsi": float(rsi) if rsi else None,
                "beta": info.get("beta", None),
                "eps": info.get("trailingEps", None),
                "revenue": info.get("totalRevenue", None),
                "profit_margin": info.get("profitMargins", None),
                "debt_to_equity": info.get("debtToEquity", None),
                "historical_data": hist_data.to_dict("records"),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            error_logger.error(
                f"Error fetching stock data for {symbol}: {str(e)}", exc_info=True
            )
            raise

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index (RSI)
        """
        try:
            if len(prices) < period + 1:
                return None

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1]
        except Exception:
            return None

    def get_market_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        Fetch recent news for a stock symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            formatted_news = []
            for article in news[:limit]:
                formatted_news.append(
                    {
                        "title": article.get("title", ""),
                        "link": article.get("link", ""),
                        "publisher": article.get("publisher", ""),
                        "published": (
                            datetime.fromtimestamp(
                                article.get("providerPublishTime", 0)
                            ).isoformat()
                            if article.get("providerPublishTime")
                            else None
                        ),
                        "summary": article.get("summary", ""),
                    }
                )

            return formatted_news

        except Exception as e:
            error_logger.error(
                f"Error fetching news for {symbol}: {str(e)}", exc_info=True
            )
            return []

    def create_stock_chart(
        self, symbol: str, data: Dict, chart_type: str = "candlestick"
    ) -> str:
        """
        Create interactive stock charts using Plotly
        """
        try:
            # Convert historical data back to DataFrame
            df = pd.DataFrame(data["historical_data"])
            if df.empty:
                return ""

            # Ensure Date column is datetime
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            else:
                df.reset_index(inplace=True)
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                else:
                    df["Date"] = pd.date_range(
                        start="2023-01-01", periods=len(df), freq="D"
                    )

            # Create subplots
            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    f"{symbol} Stock Price",
                    "Volume",
                    "Technical Indicators",
                ),
                vertical_spacing=0.1,
                row_heights=[0.6, 0.2, 0.2],
            )

            if chart_type == "candlestick":
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name=symbol,
                    ),
                    row=1,
                    col=1,
                )
            else:
                # Line chart
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df["Close"],
                        mode="lines",
                        name=f"{symbol} Close Price",
                        line=dict(color="blue", width=2),
                    ),
                    row=1,
                    col=1,
                )

            # Add moving averages if available
            if len(df) >= 20:
                ma_20 = df["Close"].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=ma_20,
                        mode="lines",
                        name="MA 20",
                        line=dict(color="orange", width=1),
                    ),
                    row=1,
                    col=1,
                )

            if len(df) >= 50:
                ma_50 = df["Close"].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=ma_50,
                        mode="lines",
                        name="MA 50",
                        line=dict(color="red", width=1),
                    ),
                    row=1,
                    col=1,
                )

            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=df["Date"],
                    y=df["Volume"],
                    name="Volume",
                    marker_color="lightblue",
                ),
                row=2,
                col=1,
            )

            # RSI chart
            if len(df) >= 14:
                rsi_data = self.calculate_rsi_series(df["Close"])
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=rsi_data,
                        mode="lines",
                        name="RSI",
                        line=dict(color="purple", width=1),
                    ),
                    row=3,
                    col=1,
                )

                # Add RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # Update layout
            fig.update_layout(
                title=f'{symbol} - {data.get("company_name", symbol)} Stock Analysis',
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                template="plotly_white",
            )

            # Update y-axis titles
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

            return fig.to_json()

        except Exception as e:
            error_logger.error(
                f"Error creating chart for {symbol}: {str(e)}", exc_info=True
            )
            return ""

    def calculate_rsi_series(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI for entire series
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except Exception:
            return pd.Series([50] * len(prices))

    def analyze_stock_trends(self, data: Dict) -> Dict:
        """
        Perform technical analysis and provide insights
        """
        try:
            insights = {
                "overall_trend": "neutral",
                "signals": [],
                "risk_level": "medium",
                "recommendation": "hold",
            }

            current_price = data.get("current_price", 0)
            ma_20 = data.get("ma_20")
            ma_50 = data.get("ma_50")
            ma_200 = data.get("ma_200")
            rsi = data.get("rsi")
            change_percent = data.get("change_percent", 0)

            # Trend analysis
            if ma_20 and ma_50 and ma_200:
                if current_price > ma_20 > ma_50 > ma_200:
                    insights["overall_trend"] = "bullish"
                    insights["signals"].append(
                        "Strong uptrend - price above all moving averages"
                    )
                elif current_price < ma_20 < ma_50 < ma_200:
                    insights["overall_trend"] = "bearish"
                    insights["signals"].append(
                        "Strong downtrend - price below all moving averages"
                    )

            # RSI analysis
            if rsi:
                if rsi > 70:
                    insights["signals"].append(
                        "RSI indicates overbought conditions (>70)"
                    )
                    insights["risk_level"] = "high"
                elif rsi < 30:
                    insights["signals"].append(
                        "RSI indicates oversold conditions (<30)"
                    )
                    insights["recommendation"] = "buy"
                elif 40 <= rsi <= 60:
                    insights["signals"].append(
                        "RSI in neutral zone - balanced momentum"
                    )

            # Price change analysis
            if change_percent > 5:
                insights["signals"].append(
                    f"Strong positive momentum (+{change_percent:.2f}%)"
                )
            elif change_percent < -5:
                insights["signals"].append(
                    f"Strong negative momentum ({change_percent:.2f}%)"
                )

            # Volume analysis
            current_volume = data.get("volume", 0)
            avg_volume = data.get("avg_volume", 1)

            if current_volume > avg_volume * 1.5:
                insights["signals"].append("High volume activity - increased interest")
            elif current_volume < avg_volume * 0.5:
                insights["signals"].append("Low volume - limited trading interest")

            # Generate recommendation
            bullish_signals = len(
                [
                    s
                    for s in insights["signals"]
                    if any(
                        word in s.lower()
                        for word in ["strong uptrend", "oversold", "positive momentum"]
                    )
                ]
            )
            bearish_signals = len(
                [
                    s
                    for s in insights["signals"]
                    if any(
                        word in s.lower()
                        for word in ["downtrend", "overbought", "negative momentum"]
                    )
                ]
            )

            if bullish_signals > bearish_signals:
                insights["recommendation"] = "buy"
            elif bearish_signals > bullish_signals:
                insights["recommendation"] = "sell"
            else:
                insights["recommendation"] = "hold"

            return insights

        except Exception as e:
            error_logger.error(f"Error analyzing stock trends: {str(e)}", exc_info=True)
            return {
                "overall_trend": "neutral",
                "signals": ["Analysis unavailable"],
                "risk_level": "unknown",
                "recommendation": "hold",
            }

    def get_market_overview(self) -> Dict:
        """
        Get overview of major market indices
        """
        try:
            indices = {
                "^GSPC": "S&P 500",
                "^DJI": "Dow Jones",
                "^IXIC": "NASDAQ",
                "^RUT": "Russell 2000",
            }

            market_data = {}

            for symbol, name in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")

                    if not hist.empty:
                        current = hist["Close"].iloc[-1]
                        previous = hist["Close"].iloc[-2] if len(hist) > 1 else current
                        change = current - previous
                        change_percent = (change / previous * 100) if previous else 0

                        market_data[symbol] = {
                            "name": name,
                            "price": float(current),
                            "change": float(change),
                            "change_percent": float(change_percent),
                        }
                except Exception:
                    continue

            return market_data

        except Exception as e:
            error_logger.error(
                f"Error fetching market overview: {str(e)}", exc_info=True
            )
            return {}
