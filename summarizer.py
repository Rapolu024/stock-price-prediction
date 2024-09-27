from transformers import pipeline

def summarize_text(news_articles):
    summarizer = pipeline('summarization')

    # Summarize either the 'summary' or 'title' if available, and handle missing keys
    summaries = []
    for article in news_articles:
        # Use 'summary' if available; otherwise, fallback to 'title' or other text fields
        text_to_summarize = article.get('summary') or article.get('title') or article.get('description', "")
        
        if text_to_summarize:
            # Perform the summarization
            summary = summarizer(text_to_summarize, max_length=100, min_length=100, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        else:
            summaries.append("No content available to summarize.")

    return summaries

if __name__ == "__main__":
    # Example structure of news_articles (replace this with the actual scraped data from `yfinance`)
    news_articles = [
        {"title": "Apple releases new iPhone with innovative features."},
        {"description": "Microsoft announces cloud-based products."},
        {"summary": "Long news article about a company's quarterly earnings."},
        {"title": "A short article without much detail."}
    ]

    summaries = summarize_text(news_articles)
    print("Summarized Articles:", summaries)
