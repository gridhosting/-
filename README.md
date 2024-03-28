# -
智能投资利用大数据分析和机器学习算法,为投资者提供实时的市场洞察和个性化的投资建议,帮助投资者做出更明智的投资决策。
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta

def collect_data(stock_symbol, start_date, end_date):
    """
    Collects historical stock data from Yahoo Finance.
    
    Parameters:
    - stock_symbol: The ticker symbol of the stock (e.g., 'AAPL' for Apple Inc.)
    - start_date: Start date of the historical data (format: 'YYYY-MM-DD')
    - end_date: End date of the historical data (format: 'YYYY-MM-DD')
    
    Returns:
    - Pandas DataFrame with historical stock data
    """
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

def feature_engineering(data):
    """
    Creates new features for the machine learning model.
    
    Parameters:
    - data: Pandas DataFrame with the stock data
    
    Returns:
    - Modified DataFrame with new features
    """
    # Simple feature: Moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    # You can add more sophisticated features here
    return data.dropna()

def train_model(data):
    """
    Trains a machine learning model to predict future stock prices.
    
    Parameters:
    - data: Pandas DataFrame with stock data and features
    
    Returns:
    - Trained model
    """
    # Feature and target variable
    X = data[['MA_5', 'MA_10']]
    y = data['Close']
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model: Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse}")
    
    return model

def predict_and_advise(model, recent_data):
    """
    Uses the model to predict the next day's closing price and provides basic investment advice.
    
    Parameters:
    - model: Trained machine learning model
    - recent_data: Most recent stock data used for making a prediction
    
    Returns:
    - Prediction for the next day's closing price and investment advice
    """
    next_day_prediction = model.predict([recent_data])
    print(f"Predicted next day closing price: {next_day_prediction[0]}")
    
    # Basic advice logic
    if recent_data[-1] < next_day_prediction:
        advice = "Considering buying, expected to rise."
    else:
        advice = "Considering selling, expected to fall."
    
    return next_day_prediction[0], advice

if __name__ == "__main__":
    stock_symbol = "AAPL" # Example stock symbol
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = collect_data(stock_symbol, start_date, end_date)
    data_with_features = feature_engineering(data)
    model = train_model(data_with_features)
    
    # Assuming 'recent_data' is the latest available data
    recent_data = data_with_features.iloc[-1][['MA_5', 'MA_10']].values
    prediction, advice = predict_and_advise(model, recent_data)
    print(f"Advice: {advice}")
