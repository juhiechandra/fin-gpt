from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import ollama
import yfinance as yf
import time
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ollama Finance Chat",
    description="A simple Ollama-based finance chatbot with automatic stock data fetching",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    processing_time: float
    session_id: str
    stock_data: Optional[List[Dict[str, Any]]] = None


# Ollama configuration
FINANCE_MODEL = "jjansen/adapt-finance-llama2-7b:latest"  # Main finance model


# Stock data functions
def get_stock_price(symbol: str) -> float:
    """
    Get the current stock price for a given symbol

    Args:
        symbol: The stock symbol (e.g., AAPL, GOOGL, TSLA)

    Returns:
        float: The current stock price
    """
    try:
        ticker = yf.Ticker(symbol.upper())

        # Try to get price from info first
        info = ticker.info
        if info and "regularMarketPrice" in info and info["regularMarketPrice"]:
            return float(info["regularMarketPrice"])
        elif info and "currentPrice" in info and info["currentPrice"]:
            return float(info["currentPrice"])

        # Fallback to fast_info
        fast_info = ticker.fast_info
        if hasattr(fast_info, "last_price") and fast_info.last_price:
            return float(fast_info.last_price)

        # Final fallback to history
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

        raise ValueError(f"No price data found for {symbol}")

    except Exception as e:
        logger.error(f"Error fetching stock price for {symbol}: {str(e)}")
        raise Exception(f"Could not fetch stock price for {symbol}: {str(e)}")


def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive stock information for a given symbol

    Args:
        symbol: The stock symbol (e.g., AAPL, GOOGL, TSLA)

    Returns:
        dict: Comprehensive stock information
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        # Get current price
        current_price = None
        if info and "regularMarketPrice" in info:
            current_price = info["regularMarketPrice"]
        elif info and "currentPrice" in info:
            current_price = info["currentPrice"]

        if not current_price:
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])

        return {
            "symbol": symbol.upper(),
            "company_name": info.get("longName", "Unknown"),
            "current_price": current_price,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "price_change": info.get("regularMarketChange"),
            "price_change_percent": info.get("regularMarketChangePercent"),
            "volume": info.get("regularMarketVolume"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }

    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        raise Exception(f"Could not fetch stock information for {symbol}: {str(e)}")


# Stock functions available for direct calling
stock_functions = {
    "get_stock_price": get_stock_price,
    "get_stock_info": get_stock_info,
}


def check_ollama_connection() -> bool:
    """Check if Ollama server is running and accessible"""
    try:
        ollama.list()
        return True
    except Exception as e:
        logger.error(f"Ollama connection failed: {str(e)}")
        return False


def extract_stock_symbols(message: str) -> list:
    """Extract multiple stock symbols from user message"""
    import re

    # Common stock symbols (add more as needed)
    known_symbols = {
        "apple": "AAPL",
        "tesla": "TSLA",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "netflix": "NFLX",
        "spotify": "SPOT",
        "uber": "UBER",
        "disney": "DIS",
        "coca cola": "KO",
        "pepsi": "PEP",
        "walmart": "WMT",
        "target": "TGT",
        "ibm": "IBM",
        "intel": "INTC",
        "amd": "AMD",
        "salesforce": "CRM",
        "oracle": "ORCL",
        "adobe": "ADBE",
        "paypal": "PYPL",
        "visa": "V",
        "mastercard": "MA",
        "jpmorgan": "JPM",
        "goldman sachs": "GS",
        "berkshire": "BRK-A",
        "johnson": "JNJ",
        "pfizer": "PFE",
        "moderna": "MRNA",
        "boeing": "BA",
        "ge": "GE",
        "ford": "F",
        "gm": "GM",
        "general motors": "GM",
    }

    message_lower = message.lower()
    found_symbols = set()

    # First, look for explicit stock symbols (3-5 uppercase letters)
    symbol_pattern = r"\b([A-Z]{1,5})\b"
    symbols = re.findall(symbol_pattern, message)
    for symbol in symbols:
        # Filter out common words that might match the pattern
        if symbol not in [
            "AND",
            "OR",
            "THE",
            "FOR",
            "WITH",
            "FROM",
            "TO",
            "BY",
            "OF",
            "IN",
            "ON",
            "AT",
            "IS",
            "ARE",
            "WAS",
            "WERE",
        ]:
            found_symbols.add(symbol)

    # Then look for company names
    for company, symbol in known_symbols.items():
        if company in message_lower:
            found_symbols.add(symbol)

    return list(found_symbols)


def extract_stock_symbol(message: str) -> str:
    """Extract single stock symbol from user message (for backward compatibility)"""
    symbols = extract_stock_symbols(message)
    return symbols[0] if symbols else None


def is_stock_query(message: str) -> bool:
    """Check if the message is asking for stock information"""
    stock_keywords = [
        "stock",
        "price",
        "share",
        "ticker",
        "market",
        "trading",
        "stock price",
        "current price",
        "stock info",
        "stock data",
        "quote",
        "shares",
        "equity",
        "NYSE",
        "NASDAQ",
    ]

    message_lower = message.lower()
    has_stock_keyword = any(
        keyword.lower() in message_lower for keyword in stock_keywords
    )
    has_stock_symbol = extract_stock_symbol(message) is not None

    return has_stock_keyword or has_stock_symbol


@app.on_event("startup")
async def startup_event():
    """Check Ollama connection on startup"""
    if not check_ollama_connection():
        logger.warning(
            "Ollama server is not accessible. Please ensure Ollama is running."
        )
    else:
        logger.info("Ollama server connection verified")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_status = check_ollama_connection()
    return {
        "status": "healthy" if ollama_status else "degraded",
        "ollama_connected": ollama_status,
        "timestamp": time.time(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Simple chat endpoint with automatic stock data fetching
    """
    try:
        start_time = time.time()

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        logger.info(f"Processing chat request for session: {session_id}")
        logger.info(f"User message: {request.message[:100]}...")

        # Check Ollama connection
        if not check_ollama_connection():
            raise HTTPException(
                status_code=503, detail="Ollama service is not available"
            )

        # Prepare the message for the model
        enhanced_message = request.message
        structured_stock_data = None

        # Check if this is a stock-related query and automatically fetch data
        if is_stock_query(request.message):
            symbols = extract_stock_symbols(request.message)
            if symbols:
                logger.info(f"Stock query detected for symbols: {symbols}")
                stock_data_parts = []
                structured_stock_data = []

                # Check if it's a comparison query (multiple symbols)
                is_comparison = len(symbols) > 1 or any(
                    word in request.message.lower()
                    for word in ["compare", "comparison", "vs", "versus", "against"]
                )

                try:
                    for symbol in symbols:
                        # Get structured stock data for chart
                        try:
                            stock_info = get_stock_info(symbol)
                            current_price = stock_info.get("current_price")
                            if current_price:
                                structured_stock_data.append(
                                    {
                                        "symbol": symbol.upper(),
                                        "price": float(current_price),
                                        "change": 0,  # Could calculate change if needed
                                        "company_name": stock_info.get(
                                            "company_name", symbol
                                        ),
                                    }
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to get structured data for {symbol}: {e}"
                            )

                        # Generate text for AI context
                        if (
                            any(
                                word in request.message.lower()
                                for word in [
                                    "detailed",
                                    "information",
                                    "info",
                                    "analysis",
                                    "company",
                                ]
                            )
                            and not is_comparison
                        ):
                            # Detailed info for single stock
                            stock_info = get_stock_info(symbol)
                            stock_data = (
                                f"\n\nDetailed stock information for {symbol}:\n"
                            )
                            stock_data += (
                                f"Company: {stock_info.get('company_name', 'N/A')}\n"
                            )
                            stock_data += f"Current Price: ${stock_info.get('current_price', 'N/A')}\n"
                            stock_data += (
                                f"Market Cap: ${stock_info.get('market_cap', 'N/A'):,}\n"
                                if stock_info.get("market_cap")
                                else f"Market Cap: N/A\n"
                            )
                            stock_data += (
                                f"P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}\n"
                            )
                            stock_data += f"Dividend Yield: {stock_info.get('dividend_yield', 'N/A')}\n"
                            stock_data += f"Sector: {stock_info.get('sector', 'N/A')}\n"
                            stock_data += (
                                f"Industry: {stock_info.get('industry', 'N/A')}\n"
                            )
                            stock_data += f"52-Week High: ${stock_info.get('fifty_two_week_high', 'N/A')}\n"
                            stock_data += f"52-Week Low: ${stock_info.get('fifty_two_week_low', 'N/A')}\n"
                            stock_data_parts.append(stock_data)
                        else:
                            # Price info for comparison or simple queries
                            stock_price = get_stock_price(symbol)
                            if is_comparison:
                                stock_data_parts.append(f"{symbol}: ${stock_price}")
                            else:
                                stock_data_parts.append(
                                    f"\n\nCurrent stock price for {symbol}: ${stock_price}"
                                )

                    if is_comparison and len(symbols) > 1:
                        stock_data = f"\n\nStock price comparison:\n" + "\n".join(
                            stock_data_parts
                        )
                    else:
                        stock_data = "\n".join(stock_data_parts)

                    enhanced_message = request.message + stock_data
                    logger.info(f"Added stock data for {len(symbols)} symbols")

                except Exception as e:
                    logger.warning(
                        f"Failed to get stock data for symbols {symbols}: {e}"
                    )
                    enhanced_message = (
                        request.message
                        + f"\n\nNote: Unable to retrieve current stock data for some symbols."
                    )
            else:
                logger.info("Stock query detected but no symbols found")

        # Use finance model for all queries
        finance_response = ollama.chat(
            model=FINANCE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable financial advisor specializing in stock market analysis, investment strategies, and financial planning. When provided with current stock data in the user's message, use it to give accurate and helpful analysis. Always be helpful and provide insights based on the data provided.",
                },
                {"role": "user", "content": enhanced_message},
            ],
        )

        final_response = (
            finance_response.message.content or "I couldn't process your request."
        )

        processing_time = time.time() - start_time

        logger.info(f"Chat request completed in {processing_time:.2f} seconds")

        return ChatResponse(
            response=final_response,
            processing_time=processing_time,
            session_id=session_id,
            stock_data=structured_stock_data,
        )

    except Exception as e:
        error_msg = f"Error processing chat request: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ollama Finance Chat API",
        "version": "1.0.0",
        "endpoints": {"chat": "/chat", "health": "/health"},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
