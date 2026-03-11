"""MCP server exposing stock and recipe RAG tools."""

import os
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from recipes.rag.rag_service import RAGService

mcp = FastMCP("Agent Experiment MCP Server")

CSV_FILE_PATH = "../data/stocks_data.csv"
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Create and cache the RAG service instance."""
    global _rag_service
    if _rag_service is None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        query_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        db_url = os.getenv("MONGO_DB_URL")
        collection = os.getenv("MONGO_COLLECTION_NAME_RECIPES")

        if not db_url:
            raise ValueError("Missing required environment variable: MONGO_DB_URL")

        _rag_service = RAGService(
            embedding_model=embedding_model,
            query_model=query_model,
            db_url=db_url,
            collection=collection,
        )
    return _rag_service


def get_price_from_csv(symbol: str) -> Optional[float]:
    try:
        if not os.path.exists(CSV_FILE_PATH):
            return None

        df = pd.read_csv(CSV_FILE_PATH)

        # Convert symbol column to uppercase for case-insensitive matching
        df["symbol"] = df["symbol"].str.upper()
        symbol = symbol.upper()

        # Find the stock in the CSV
        stock_row = df[df["symbol"] == symbol]

        if not stock_row.empty:
            return float(stock_row["price"].iloc[0])
        else:
            return None

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def get_stock_price_with_fallback(symbol: str) -> tuple[Optional[float], str]:
    # Try yfinance first
    try:
        ticker = yf.Ticker(symbol)

        # Get today's data (may be empty if market is closed)
        data = ticker.history(period="1d")

        if not data.empty:
            price = data["Close"].iloc[-1]
            return price, "yfinance"
        else:
            # Try using regular market price from ticker info
            info = ticker.info
            price = info.get("regularMarketPrice")

            if price is not None:
                return price, "yfinance"

    except Exception as e:
        print(f"yfinance error for {symbol}: {e}")
        pass

    # Fallback to CSV
    csv_price = get_price_from_csv(symbol)
    if csv_price is not None:
        return csv_price, "csv"

    return None, "none"


@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """
    Retrieve the latest available price for one stock ticker symbol.

    Use this tool when the user asks for the current/latest price of a single
    stock or wants a quick price lookup for one symbol.

    Do not use this tool for recipe questions or side-by-side comparisons of
    two symbols (use compare_stocks for comparisons).

    Parameters:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        A human-readable price result indicating source (Yahoo Finance or local fallback).
    """
    price, source = get_stock_price_with_fallback(symbol)

    if price is not None:
        source_text = (
            " (from Yahoo Finance)" if source == "yfinance" else " (from local data)"
        )
        return f"The current price of {symbol} is ${price:.2f}{source_text}"
    else:
        return (
            f"Could not retrieve price for {symbol} from either Yahoo Finance or local data. "
            f"Please ensure the symbol is correct and that local data file '{CSV_FILE_PATH}' "
            f"exists with the required format."
        )


@mcp.tool()
def compare_stocks(symbol1: str, symbol2: str) -> str:
    """
    Compare the latest available prices of two stock ticker symbols.

    Use this tool when the user explicitly asks to compare two stocks (higher,
    lower, same) or asks which of two symbols is more expensive.

    Do not use this tool for single-symbol price lookups (use get_stock_price)
    or for non-finance requests.

    Parameters:
        symbol1: First stock ticker symbol
        symbol2: Second stock ticker symbol

    Returns:
        A human-readable comparison including source hints for each symbol.
    """
    # Get prices for both symbols
    price1, source1 = get_stock_price_with_fallback(symbol1)
    price2, source2 = get_stock_price_with_fallback(symbol2)

    if price1 is None:
        return f"Could not retrieve price for {symbol1} from either Yahoo Finance or local data."

    if price2 is None:
        return f"Could not retrieve price for {symbol2} from either Yahoo Finance or local data."

    # Create source information
    source1_text = " (YF)" if source1 == "yfinance" else " (local)"
    source2_text = " (YF)" if source2 == "yfinance" else " (local)"

    if price1 > price2:
        return f"{symbol1} (${price1:.2f}{source1_text}) is higher than {symbol2} (${price2:.2f}{source2_text})."
    elif price1 < price2:
        return f"{symbol1} (${price1:.2f}{source1_text}) is lower than {symbol2} (${price2:.2f}{source2_text})."
    else:
        return f"Both {symbol1} and {symbol2} have the same price (${price1:.2f})."


@mcp.tool()
def answer_recipe_query(query: str) -> str:
    """Answer recipe-focused questions using the recipe RAG pipeline.

    Use this tool when the user asks about recipes, ingredients, substitutions,
    preparation steps, nutrition-related recipe guidance, or meal ideas grounded
    in the recipe knowledge base.

    Do not use this tool for stock/finance questions, generic chit-chat, or
    requests unrelated to recipes.

    Parameters:
        query: Natural-language recipe question or request.

    Returns:
        Grounded recipe answer generated by retrieval + LLM synthesis.
    """

    if not query.strip():
        return "Query must not be empty."

    try:
        rag_service = get_rag_service()
        answer = rag_service.generate_response(query)
        return answer or "RAG returned an empty answer."
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        return f"RAG generation failed: {exc}"


if __name__ == "__main__":
    load_dotenv()
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "sse":
        try:
            mcp.run(transport="sse")
        except TypeError:
            print(
                "This MCP version does not accept explicit transport selection. "
                "Falling back to default mcp.run() transport."
            )
            mcp.run()
    else:
        mcp.run()
