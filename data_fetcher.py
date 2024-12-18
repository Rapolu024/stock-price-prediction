import os
import logging
import yfinance as yf
import finnhub
import csv
import pandas as pd  # Import pandas for removing duplicates
from datetime import datetime, timedelta
from config import FINNHUB_API_KEY
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Finnhub client
if not FINNHUB_API_KEY:
    logger.error("Finnhub API key is not set. Exiting.")
    exit(1)

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


def remove_mismatched_data(stock_csv, news_csv):
    """
    Removes rows from stock_csv if there is no corresponding news data,
    and removes rows from news_csv if there is no corresponding stock data.
    """
    try:
        # Load the data from CSV files into DataFrames
        stock_data = pd.read_csv(stock_csv)
        news_data = pd.read_csv(news_csv)

        # Find stock data without corresponding news data
        stock_symbols_dates = set(
            stock_data[["Symbol", "Date"]].itertuples(index=False, name=None)
        )
        news_symbols_dates = set(
            news_data[["Symbol", "Date"]].itertuples(index=False, name=None)
        )

        # Remove rows from stock_data where there is no corresponding news data
        stock_data_filtered = stock_data[
            stock_data[["Symbol", "Date"]].apply(tuple, axis=1).isin(news_symbols_dates)
        ]

        # Remove rows from news_data where there is no corresponding stock data
        news_data_filtered = news_data[
            news_data[["Symbol", "Date"]].apply(tuple, axis=1).isin(stock_symbols_dates)
        ]

        # Save the filtered data back to CSV
        stock_data_filtered.to_csv(stock_csv, index=False)
        news_data_filtered.to_csv(news_csv, index=False)

        logger.info(f"Removed mismatched data from {stock_csv} and {news_csv}.")

    except Exception as e:
        logger.error(f"Failed to remove mismatched data: {e}")


def remove_duplicates_from_csv(file_path, key_columns):
    """Remove duplicate rows from a CSV file based on key columns."""
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Drop duplicates based on the specified key columns
        before_count = len(df)
        df = df.drop_duplicates(subset=key_columns, keep="first")
        after_count = len(df)

        # Write the cleaned DataFrame back to the same file
        df.to_csv(file_path, index=False)

        logger.info(
            f"Removed {before_count - after_count} duplicate rows from '{file_path}'."
        )
    except Exception as e:
        logger.error(f"Failed to remove duplicates from '{file_path}': {e}")


def fetch_and_store_data(
    symbols,
    start_date,
    end_date,
    period="1d",
    stock_csv="stock_data.csv",
    news_csv="news_data.csv",
):
    """
    Fetch stock price and news data for given symbols and store them in CSV files.
    """
    try:
        os.makedirs(os.path.dirname(stock_csv), exist_ok=True)
        os.makedirs(os.path.dirname(news_csv), exist_ok=True)

        if not os.path.exists(stock_csv):
            with open(stock_csv, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "Date",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "Symbol",
                    ],
                )
                writer.writeheader()

        if not os.path.exists(news_csv):
            with open(news_csv, mode="w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "Symbol",
                        "Date",
                        "Headline",
                        "Summary",
                        "Source",
                        "URL",
                    ],
                )
                writer.writeheader()

        for symbol_index in range(len(symbols)):
            symbol = symbols[symbol_index]
            logger.info(f"Processing symbol: {symbol}")
            if symbol_index % 5 == 0:
                logger.info("Waits for 60 seconds to counter the api calls")
                time.sleep(60)

            try:
                stock_data = yf.Ticker(symbol).history(
                    start=start_date, end=end_date, interval=period
                )
                if stock_data.empty:
                    logger.warning(f"No stock data found for {symbol}.")
                    continue

                stock_data.reset_index(inplace=True)
                stock_data["Symbol"] = symbol
                stock_data = stock_data[
                    ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
                ]

                stock_data["Open"] = pd.to_numeric(stock_data["Open"], errors="coerce")
                stock_data["Close"] = pd.to_numeric(
                    stock_data["Close"], errors="coerce"
                )
                stock_data.drop_duplicates(subset=["Date", "Symbol"], inplace=True)

                with open(stock_csv, mode="a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=[
                            "Date",
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Volume",
                            "Symbol",
                        ],
                    )
                    for _, row in stock_data.iterrows():
                        writer.writerow(
                            {
                                "Date": row["Date"].strftime("%Y-%m-%d"),
                                "Open": row["Open"],
                                "High": row["High"],
                                "Low": row["Low"],
                                "Close": row["Close"],
                                "Volume": row["Volume"],
                                "Symbol": symbol,
                            }
                        )
                logger.info(f"Stock data for {symbol} saved to {stock_csv}.")
            except Exception as stock_err:
                logger.error(f"Error fetching stock data for {symbol}: {stock_err}")

            # Fetch and write news data in chunks for the symbol
            try:
                # Convert string date to datetime objects
                start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

                # Date range for splitting requests
                current_start_date = start_date_dt
                current_end_date = min(
                    current_start_date + timedelta(days=30), end_date_dt
                )

                # Write news data in chunks
                while current_start_date < end_date_dt:
                    start_date_str = current_start_date.strftime("%Y-%m-%d")
                    end_date_str = current_end_date.strftime("%Y-%m-%d")

                    # Fetch news data for the date range
                    news = finnhub_client.company_news(
                        symbol, _from=start_date_str, to=end_date_str
                    )

                    if news:
                        formatted_news = [
                            {
                                "Symbol": symbol,
                                "Date": datetime.fromtimestamp(
                                    article["datetime"]
                                ).strftime("%Y-%m-%d"),
                                "Headline": article.get("headline", "N/A"),
                                "Summary": article.get("summary", "N/A"),
                                "Source": article.get("source", "N/A"),
                                "URL": article.get("url", "N/A"),
                            }
                            for article in news
                        ]

                        # Write news data to CSV
                        with open(
                            news_csv, mode="a", newline="", encoding="utf-8"
                        ) as csvfile:
                            writer = csv.DictWriter(
                                csvfile,
                                fieldnames=[
                                    "Symbol",
                                    "Date",
                                    "Headline",
                                    "Summary",
                                    "Source",
                                    "URL",
                                ],
                            )
                            writer.writerows(formatted_news)
                    else:
                        logger.info(
                            f"No news data found for {symbol} between {start_date_str} and {end_date_str}."
                        )

                    # Update date range for the next loop iteration
                    current_start_date = current_end_date + timedelta(days=1)
                    current_end_date = min(
                        current_start_date + timedelta(days=30), end_date_dt
                    )
                logger.info(f"News data for {symbol} saved to {news_csv}.")

            except Exception as news_err:
                logger.error(f"Error fetching news data for {symbol}: {news_err}")

        # Remove duplicates from stock and news CSV files after writing data
        remove_duplicates_from_csv(
            stock_csv, key_columns=["Date", "Symbol"]
        )
        remove_duplicates_from_csv(
            news_csv, key_columns=["Symbol", "Date"]
        )
        # After fetching and saving stock and news data, call this function to clean mismatched data
        remove_mismatched_data(
            stock_csv="output/stock_data.csv", news_csv="output/news_data.csv"
        )

    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")


# # Example usage
# if __name__ == "__main__":
#     symbols_list = ["AAPL", "GOOGL", "TSLA"]
#     start_date = "2022-01-01"
#     end_date = "2024-11-22"
#     stock_output_csv = "output/stock_data.csv"
#     news_output_csv = "output/news_data.csv"

#     fetch_and_store_data(
#         symbols=symbols_list,
#         start_date=start_date,
#         end_date=end_date,
#         period="1d",
#         stock_csv=stock_output_csv,
#         news_csv=news_output_csv
#     )
