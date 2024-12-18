
# Stock Market Prediction Using Sentiment Analysis and Random Forest Regressor

This project aims to predict stock prices by integrating **sentiment analysis** of financial news articles with historical stock data. By utilizing the powerful BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis and a **Random Forest Regressor** for prediction, the goal is to predict the next day's stock price movements more accurately than traditional methods.


## Project Overview
Accurately predicting stock prices has always been a major challenge in the financial market due to the dynamic and volatile nature of stock price movements. Traditional methods such as **technical analysis** and **fundamental analysis** often overlook the influence of news, rumors, and external events. This project aims to combine historical stock data with **sentiment analysis** from financial news articles to enhance prediction accuracy.

### The Steps:
1. **Sentiment Analysis using BERT**: The BERT model is fine-tuned to classify financial news articles as positive, negative, or neutral, providing sentiment scores for each article.
2. **Data Preprocessing**: Historical stock data and sentiment scores are cleaned, transformed, and aligned based on date.
3. **Feature Engineering**: Features such as stock **open price**, **close price**, **volume**, and **sentiment** are created to train the model.
4. **Model Training**: The **Random Forest Regressor** is used to predict the next day’s stock prices based on the selected features.
5. **Model Evaluation**: The model is evaluated using **Mean Absolute Error (MAE)** and **R-squared (R²)** scores to assess its prediction accuracy.

## Technologies Used

- **Python** (Programming Language)
- **Pandas** (Data manipulation and analysis)
- **Scikit-learn** (Machine learning algorithms and utilities)
- **Transformers** (Hugging Face library for BERT)
- **Matplotlib / Seaborn** (Data visualization)
- **NumPy** (Numerical computing)
- **Nltk** (Natural Language Processing tasks)

## Dataset
The dataset used in this project consists of:
1. **Stock Market Data**: Daily stock data, including open, close, and volume, obtained from platforms like Yahoo Finance and Finnhub.
2. **Financial News Articles**: A collection of financial news articles related to the stock market, sourced from news APIs like Finnhub. Sentiment analysis is performed on these articles to extract sentiment scores (positive, negative, or neutral).

The dataset is preprocessed by:
- Handling missing values
- Normalizing the stock data
- Tokenizing and vectorizing the news text for sentiment analysis

## Installation Guide

To run this project, you'll need Python 3.6 or above and the following dependencies.

### Install the required libraries:
```bash
pip install pandas scikit-learn transformers nltk numpy matplotlib seaborn
```

### Clone the repository:
```bash
git clone git@github.com:Rapolu024/stock-price-prediction.git
cd stock-price-prediction
```

## How It Works

1. **Sentiment Analysis**:
    - The BERT model is fine-tuned on a labeled dataset of financial news articles to generate sentiment scores (positive, negative, or neutral).
    - These sentiment scores reflect the market sentiment, which can impact stock price movements.
  
2. **Feature Engineering**:
    - Historical stock data (open, close, volume) and sentiment scores are selected as features.
    - These features are used to train the **Random Forest Regressor**.

3. **Model Training**:
    - The **Random Forest Regressor** is trained on a training dataset, using the selected features to predict stock prices (open and close) for the next day.

4. **Model Evaluation**:
    - The model’s performance is evaluated using **Mean Absolute Error (MAE)** and **R-squared (R²)** to assess its prediction accuracy.

## Model Training

The **Random Forest Regressor** is chosen due to its robustness in handling complex datasets and its ability to avoid overfitting. The model is trained using:
- A subset of historical stock market data
- Sentiment scores from financial news articles

The training set is used to teach the model to make accurate predictions for the next day's stock prices, while the remaining data is used for validation and testing.

## Model Evaluation

Model performance is assessed using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average absolute error between predicted and actual values.
- **R-squared (R²)**: Indicates the proportion of variance explained by the model.

These metrics help determine how well the model is generalizing to unseen data.

## Usage

1. Clone the repository.
2. Install dependencies.
3. Load the dataset and train the model.
4. Evaluate the model using test data.

```bash
python stock_market.py
```

This will run the training process, evaluate the model, and output performance metrics.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

