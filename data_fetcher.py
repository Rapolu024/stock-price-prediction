import yfinance as yf

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    stock_price = stock.history(period="1d")['Close'][0]  # Get current stock price
    news = stock.news  # Fetch latest news articles
    return stock_price, news

def get_all_stock_symbols(filename='stocks.txt'):
    with open(filename, 'r') as file:
        symbols = [line.strip() for line in file if line.strip()]
    return symbols

if __name__ == "__main__":
    symbols = get_all_stock_symbols()
    for symbol in symbols:
        stock_price, news = get_stock_data(symbol)
        print(f"Stock Price for {symbol}:", stock_price)
        print(f"News for {symbol}:", news)
