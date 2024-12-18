import os
import csv
import logging
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
from data_fetcher import fetch_and_store_data
import torch
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

warnings.filterwarnings("ignore")


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the tokenizer and model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the sentiment analysis pipeline with the model and tokenizer
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def analyze_sentiments_and_store(symbol):
    try:
        # Ensure stock and news data exist
        if not os.path.exists(STOCK_FILE) or not os.path.exists(NEWS_FILE):
            logger.error(
                "Stock or News data files are missing. Ensure they are generated before running sentiment analysis."
            )
            return

        # Load stock and news data
        stock_data = pd.read_csv(STOCK_FILE)
        news_data = pd.read_csv(NEWS_FILE)

        # Filter data for the given symbol
        stock_data = stock_data[stock_data["Symbol"] == symbol]
        news_data = news_data[news_data["Symbol"] == symbol]

        # Clean 'Open' and 'Close' columns: Remove non-numeric values and handle NaN values
        stock_data["Open"] = pd.to_numeric(stock_data["Open"], errors="coerce")
        stock_data["Close"] = pd.to_numeric(stock_data["Close"], errors="coerce")
        stock_data = stock_data.dropna(
            subset=["Open", "Close"]
        )  # Drop rows with missing Open or Close prices

        results = []

        # Iterate over each unique date in the stock data
        for date in stock_data["Date"].unique():
            daily_stock = stock_data[stock_data["Date"] == date]
            daily_news = news_data[news_data["Date"] == date]

            # Skip if there is no stock or news data for the current date
            if daily_stock.empty or daily_news.empty:
                continue

            # Extract stock data for the date
            open_price = daily_stock["Open"].iloc[0]
            close_price = daily_stock["Close"].iloc[0]
            price_change = close_price - open_price

            # Analyze sentiment for all news articles of the day
            sentiments = []
            avg_sentiment = None
            for _, row in daily_news.iterrows():
                article_text = row["Summary"]

                # Skip if article is empty or NaN
                if pd.isna(article_text) or not article_text.strip():
                    continue

                try:
                    # Tokenize the article text
                    inputs = tokenizer(
                        article_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True,
                    )

                    # Use the model to predict sentiment
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prediction = torch.argmax(logits, dim=-1).item()

                    # Map sentiment labels to numeric values
                    if prediction == 1:  # positive sentiment
                        score = 1
                    elif prediction == 0:  # negative sentiment
                        score = -1
                    else:  # neutral or undefined sentiment
                        score = 0

                    sentiments.append(score)
                    avg_sentiment = sum(sentiments) / len(
                        sentiments
                    )  # Update average sentiment

                except Exception as e:
                    logger.error(
                        f"Error analyzing sentiment for article: {article_text}, Error: {e}"
                    )
                    # If error occurs, use the average sentiment from previous successful records
                    if avg_sentiment is not None:
                        sentiments.append(
                            avg_sentiment
                        )  # Apply the last known sentiment

            if not sentiments:
                continue

            # Calculate the average sentiment for the day
            avg_sentiment = sum(sentiments) / len(sentiments)

            # Adjust sentiment based on stock price movement
            if price_change < 0:
                avg_sentiment -= 1  # Decrease sentiment score if stock dropped
            elif price_change > 0:
                avg_sentiment += 1  # Increase sentiment score if stock rose

            results.append({"Symbol": symbol, "Date": date, "Sentiment": avg_sentiment})

        # Convert results into DataFrame and handle NaN values
        if results:
            senti_df = pd.DataFrame(results)

            # Drop rows with NaN values in Sentiment column
            senti_df = senti_df.dropna(subset=["Sentiment"])

            # Save results to senti.csv
            senti_df.to_csv(
                SENTI_FILE, mode="a", header=not os.path.exists(SENTI_FILE), index=False
            )
            logger.info(f"Sentiments stored for {symbol} in {SENTI_FILE}.")
        else:
            logger.warning(f"No sentiment data to store for {symbol}.")

    except Exception as e:
        logger.error(f"Error during sentiment analysis for {symbol}: {e}")


# Function to train and predict
def train_and_predict(symbol):
    try:
        if not os.path.exists(STOCK_FILE) or not os.path.exists(SENTI_FILE):
            logger.error("Required data files are missing.")
            return

        # Load the data
        stock_data = pd.read_csv(STOCK_FILE)
        senti_data = pd.read_csv(SENTI_FILE)

        # Filter data for the given symbol
        stock_data = stock_data[stock_data["Symbol"] == symbol]
        senti_data = senti_data[senti_data["Symbol"] == symbol]

        # Merge the datasets based on Symbol and Date
        combined_data = pd.merge(stock_data, senti_data, on=["Symbol", "Date"])

        # Feature columns and target columns
        X = combined_data[["Open", "Close", "Volume", "Sentiment"]]
        y_open = combined_data["Close"].shift(
            -1
        )  
        # Next day's open (which is today's close)
        y_close = combined_data["Close"].shift(-1)  # Next day's close (to be predicted)

        # Drop the last row that contains NaN after shifting
        combined_data = combined_data.iloc[:-1]
        y_open = y_open.dropna()
        y_close = y_close.dropna()

        # Ensure that X has the same length as y_open and y_close
        X = X.iloc[:-1]

        # Train-test split (80% training, 20% testing)
        X_train, X_test, y_open_train, y_open_test, y_close_train, y_close_test = (
            train_test_split(X, y_open, y_close, test_size=0.2, random_state=42)
        )

        # Initialize the Random Forest Regressors
        open_model = RandomForestRegressor(n_estimators=100, random_state=42)
        close_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the models
        open_model.fit(X_train, y_open_train)
        close_model.fit(X_train, y_close_train)

        # Make predictions on the test set
        open_predictions = open_model.predict(X_test)
        close_predictions = close_model.predict(X_test)

        # Initialize a list to store results for each symbol
        metrics_results = []

        # Calculate accuracy - Mean Absolute Error (MAE) and R^2 Score
        open_mae = mean_absolute_error(y_open_test, open_predictions)
        close_mae = mean_absolute_error(y_close_test, close_predictions)
        open_r2 = r2_score(y_open_test, open_predictions)
        close_r2 = r2_score(y_close_test, close_predictions)

        # Log the results
        logger.info(f"{symbol} Open price prediction MAE: {open_mae}")
        logger.info(f"{symbol} Close price prediction MAE: {close_mae}")
        logger.info(f"{symbol} Open price prediction R^2: {open_r2}")
        logger.info(f"{symbol} Close price prediction R^2: {close_r2}")

        # Append metrics to the results list
        metrics_results.append(
            {
                "Symbol": symbol,
                "Open MAE": open_mae,
                "Close MAE": close_mae,
                "Open R2": open_r2,
                "Close R2": close_r2,
            }
        )

        # Write metrics to a CSV file
        METRICS_FILE = "metrics.csv"
        try:
            with open(METRICS_FILE, mode="a", newline="") as file:
                fieldnames = ["Symbol", "Open MAE", "Close MAE", "Open R2", "Close R2"]
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write header
                if os.stat(METRICS_FILE).st_size == 0:
                    writer.writeheader()

                # Write all rows
                for result in metrics_results:
                    writer.writerow(result)

            logger.info(f"Metrics saved in {METRICS_FILE}.")

        except Exception as e:
            logger.error(f"Error writing metrics to CSV: {e}")

        # Now predict for the next day's Open and Close prices
        predictions = []
        for _, row in combined_data.iloc[
            -1:
        ].iterrows():  # Only use the last row for prediction
            # Today's closing price is used as the predicted opening price for tomorrow
            next_open = row[
                "Close"
            ]  # Use today's closing price as the next day's opening
            next_close = close_model.predict(
                pd.DataFrame([row[["Open", "Close", "Volume", "Sentiment"]]])
            )[0]
            predictions.append(
                {
                    "Symbol": row["Symbol"],
                    "Date": row["Date"],
                    "Today Close": row["Close"],
                    "Next Day Open": next_open,
                    "Next Day Close": next_close,
                }
            )

        # Save predictions to CSV
        pd.DataFrame(predictions).to_csv(
            PREDICTIONS_FILE,
            mode="a",
            header=not os.path.exists(PREDICTIONS_FILE),
            index=False,
        )
        logger.info(f"Predictions saved for {symbol} in {PREDICTIONS_FILE}.")

    except Exception as e:
        logger.error(f"Error in training and prediction for {symbol}: {e}")


def process_symbol(symbol):
    analyze_sentiments_and_store(symbol)
    train_and_predict(symbol)


if __name__ == "__main__":

    # File paths
    SYMBOL_FILE = "symbols.txt"
    STOCK_FILE = "output/stock_data.csv"
    NEWS_FILE = "output/news_data.csv"
    SENTI_FILE = "senti.csv"
    PREDICTIONS_FILE = "next_day_predictions.csv"

    # Load symbols
    with open(SYMBOL_FILE, "r") as f:
        symbols = [symbol.strip() for symbol in f.readlines()]
    
    fetch_and_store_data(
            symbols=symbols,
            start_date="2022-01-01",
            end_date=datetime.now().strftime('%Y-%m-%d'),
            period="1d",
            stock_csv=STOCK_FILE,
            news_csv=NEWS_FILE
        )

    # Load stock and news data
    stock_data = pd.read_csv(STOCK_FILE)
    news_data = pd.read_csv(NEWS_FILE)

    with ThreadPoolExecutor(max_workers=5) as executor:
        for symbol in symbols:
            executor.submit(process_symbol, symbol)
            time.sleep(10)
            logger.info(
                "System goes sleep for 100 seconds to resolve the conflicts between the loading and predictions"
            )