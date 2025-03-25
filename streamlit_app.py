# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Function to download historical stock data
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess data
def preprocess_data(data):
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    return data

# Function to create features
def create_features(data):
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    return data

# Function to calculate daily volatility
def calculate_volatility(data):
    data['Volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(20)
    return data

# Function to classify risk
def classify_risk(volatility):
    threshold = 0.05  # Adjust the threshold as needed
    return 'High' if volatility > threshold else 'Low'

# Function to train the model
def train_model(data):
    X = data[['Year', 'Month', 'Day']]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Function to make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Function to display results
def display_results(predictions, y_test):
    results = pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions})
    st.write(results)

# Function to plot live line chart
def plot_live_line_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Live Stock Price Movement', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Main function
def main():
    st.title("Stock Market Prediction App")
    st.sidebar.header("User Input")
    
    ticker = st.sidebar.text_input("Enter Ticker Symbol", 'AAPL')
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))
    
    st.sidebar.subheader("Nifty 50 Companies")
    nifty50_companies = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'INFY.NS',
        'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'MARUTI.NS',
        'ASIANPAINT.NS', 'TECHM.NS', 'ULTRACEMCO.NS', 'AXISBANK.NS', 'ITC.NS',
        'BAJAJFINSV.NS', 'SUNPHARMA.NS', 'BAJFINANCE.NS', 'TITAN.NS', 'BAJAJ-AUTO.NS',
        'LT.NS', 'NESTLEIND.NS', 'ONGC.NS', 'NTPC.NS', 'UPL.NS',
        'POWERGRID.NS', 'SBIN.NS', 'IOC.NS', 'JSWSTEEL.NS', 'HCLTECH.NS',
        'HEROMOTOCO.NS', 'DRREDDY.NS', 'COALINDIA.NS', 'INDUSINDBK.NS', 'BRITANNIA.NS',
        'SHREECEM.NS', 'WIPRO.NS', 'BHARTIINFRA.NS', 'DIVISLAB.NS', 'GRASIM.NS',
        'CIPLA.NS', 'RECLTD.NS', 'ADANIPORTS.NS', 'IOC.NS', 'HINDALCO.NS'
    ]
    
    selected_company = st.sidebar.selectbox("Select a Company", nifty50_companies)
    
    data = download_data(selected_company, start_date, end_date)
    
    if not data.empty:
        data = preprocess_data(data)
        data = create_features(data)
        data = calculate_volatility(data)
        
        st.subheader("Stock Data")
        st.write(data)

        if st.button("Train Model"):
            model, X_test, y_test = train_model(data)
            predictions = make_predictions(model, X_test)
            display_results(predictions, y_test)

            # Only plot after predictions
            plot_live_line_chart(data)

        # Display risk classification
        risk_classification = classify_risk(data['Volatility'].iloc[-1])
        st.sidebar.subheader("Risk Classification")
        st.sidebar.write(f"The risk of {selected_company} is classified as {risk_classification}-risk.")
    else:
        st.sidebar.error("No data available for the selected ticker and date range.")

# Run the main function
if __name__ == "__main__":
    main()