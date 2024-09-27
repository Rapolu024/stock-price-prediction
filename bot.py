import csv
import os
from data_fetcher import get_stock_data, get_all_stock_symbols
from summarizer import summarize_text
from sentiment import analyze_sentiment
from decision import should_buy_stock

class StockBot:
    def __init__(self, symbols_file='stocks.txt', training_file='training_data.csv'):
        self.symbols = get_all_stock_symbols(symbols_file)
        self.training_file = training_file
        self.init_csv_file()  # Initialize CSV if not already present
        self.fetch_and_store_data()

    def init_csv_file(self):
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.training_file):
            with open(self.training_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Symbol", "Stock Price", "News Summary", "Sentiment", "Decision"])

    def fetch_and_store_data(self):
        for symbol in self.symbols:
            stock_price, news = get_stock_data(symbol)
            if news:  # Check if there's news available
                summaries = summarize_text(news)
                sentiments = analyze_sentiment(summaries)
                decision = should_buy_stock(stock_price, sentiments)
                # Store data in the CSV file
                self.store_training_data(symbol, stock_price, summaries, sentiments, decision)

    def store_training_data(self, symbol, stock_price, summaries, sentiments, decision):
        # Append data to CSV file
        with open(self.training_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for summary, sentiment in zip(summaries, sentiments):
                writer.writerow([symbol, stock_price, summary, sentiment, decision])

    def answer(self, question):
        # This part of the code will allow answering questions based on data in CSV
        with open(self.training_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'price' in question.lower() and row['Symbol'] in question:
                    return f"The current stock price of {row['Symbol']} is {row['Stock Price']}."
                elif 'news' in question.lower() and row['Symbol'] in question:
                    return f"The latest news summary for {row['Symbol']} is: {row['News Summary']}"
                elif 'sentiment' in question.lower() and row['Symbol'] in question:
                    return f"The sentiment analysis for {row['Symbol']} is: {row['Sentiment']}."
                elif 'buy' in question.lower() and row['Symbol'] in question:
                    return f"The decision for {row['Symbol']} is: {row['Decision']}."
        return "I'm sorry, I don't understand the question."

if __name__ == "__main__":
    bot = StockBot()
    
    # Sample interaction
    print(bot.answer("What is the current stock price for AAPL?"))
    print(bot.answer("What is the latest news for MSFT?"))
    print(bot.answer("Should I buy GOOGL?"))
