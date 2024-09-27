from transformers import pipeline

def analyze_sentiment(summaries):
    sentiment_analyzer = pipeline('sentiment-analysis')
    sentiments = [sentiment_analyzer(summary)[0] for summary in summaries]
    return sentiments

if __name__ == "__main__":
    summaries = ["This is a summarized article."]  # Replace with actual summaries
    sentiments = analyze_sentiment(summaries)
    print("Sentiments:", sentiments)
