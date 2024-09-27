def should_buy_stock(stock_price, sentiments):
    avg_sentiment = sum([1 if sentiment['label'] == 'POSITIVE' else -1 for sentiment in sentiments]) / len(sentiments)
    
    # Example decision: Buy if sentiment is positive and stock price is below $150
    if avg_sentiment > 0 and float(stock_price) < 150:
        return "Buy the stock"
    else:
        return "Don't buy the stock"

if __name__ == "__main__":
    stock_price = 145  # Replace with actual stock price
    sentiments = [{'label': 'POSITIVE'}, {'label': 'NEGATIVE'}, {'label': 'POSITIVE'}]  # Replace with actual sentiments
    decision = should_buy_stock(stock_price, sentiments)
    print("Decision:", decision)
