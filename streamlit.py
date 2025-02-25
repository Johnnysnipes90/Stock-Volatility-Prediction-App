import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Data import AlphaVantageAPI, SQLRepository
from model import GarchModel
from config import settings

# Page Configuration
st.set_page_config(
    page_title='📈 Stock Volatility Predictor',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Title and Introduction
st.title('📈 Volatility Prediction App')
st.markdown("""
    This app allows you to **fetch historical stock data**, **train a GARCH model**, 
    and **predict future stock volatility**. Get started by entering a stock ticker!
""")

# Database connection
connection = sqlite3.connect('volatility_data.db')
repo = SQLRepository(connection)

# Sidebar - User Inputs
st.sidebar.header('📂 Input Parameters')
ticker = st.sidebar.text_input('Ticker Symbol', value='AAPL').upper()
n_observations = st.sidebar.number_input('Number of Observations', min_value=1, value=500)
p = st.sidebar.slider('Lag Order of Innovation (p)', min_value=1, max_value=10, value=1)
q = st.sidebar.slider('Lag Order of Volatility (q)', min_value=1, max_value=10, value=1)
horizon = st.sidebar.slider('Prediction Horizon (Days)', min_value=1, max_value=30, value=5)
use_new_data = st.sidebar.checkbox('Use New Data from AlphaVantage', value=True)

# Initialize Session State Variables
if 'model' not in st.session_state:
    st.session_state.model = None

# Prediction
st.sidebar.subheader('🔮 Predict Volatility')
if st.sidebar.button('Predict Volatility'):
    with st.spinner('Fetching data and training model...'):
        model = GarchModel(ticker=ticker, repo=repo, use_new_data=use_new_data)
        model.wrangle_data(n_observations=n_observations)
        model.fit(p=p, q=q)
        prediction = model.predict_volatility(horizon=horizon)

        # Display prediction results
        st.success('✅ Prediction Complete!')
        st.subheader(f'📊 Volatility Prediction for {ticker}')
        st.write(prediction)
        
        # Visualization of the prediction
        prediction_df = pd.DataFrame(list(prediction.items()), columns=['Date', 'Predicted Volatility'])
        prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])
        prediction_df.set_index('Date', inplace=True)
        st.line_chart(prediction_df)

        # Improved Volatility Trend Visualization
        st.subheader('📈 Volatility Trend Visualization')
        fig, ax = plt.subplots()
        ax.plot(prediction_df.index, prediction_df['Predicted Volatility'], label='Predicted Volatility', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.legend()
        
        # Enhance date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# Close the database connection
connection.close()

# Footer
st.markdown('---')
st.caption('Developed with ❤️ by Olalemi John')
