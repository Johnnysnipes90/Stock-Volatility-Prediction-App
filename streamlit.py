import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
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

# Custom Styling
st.markdown("""
    <style>
    body {background-color: #f4f4f4;}
    .stSidebar {background-color: #2c3e50; color: white;}
    </style>
    """, unsafe_allow_html=True)

# Title and Introduction
st.title('📈 Stock Volatility Prediction App')
st.markdown("""
    <p style='font-size:18px;'>
    This app allows you to <b>fetch historical stock data</b>, <b>train a GARCH model</b>,
    and <b>predict future stock volatility</b>. Get started by entering a stock ticker!
    </p>
    """, unsafe_allow_html=True)

# Database connection
connection = sqlite3.connect('volatility_data.db')
repo = SQLRepository(connection)

# Sidebar - User Inputs
st.sidebar.header('📂 Input Parameters')
ticker = st.sidebar.text_input('Ticker Symbol', value='AAPL').upper()
n_observations = st.sidebar.number_input('Number of Observations', min_value=50, value=500)
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
        
        # Store results
        st.session_state.model = model
        st.session_state.prediction = prediction

        # Display prediction results
        st.success('✅ Prediction Complete!')
        st.subheader(f'📊 Volatility Prediction for {ticker}')
        st.write(prediction)
        
        # Visualization with Plotly
        prediction_df = pd.DataFrame(list(prediction.items()), columns=['Date', 'Predicted Volatility'])
        prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])
        prediction_df.set_index('Date', inplace=True)

        fig = px.line(prediction_df, x=prediction_df.index, y='Predicted Volatility',
                      title='Predicted Volatility Trend', line_shape='spline', markers=True,
                      labels={'x': 'Date', 'y': 'Volatility'})
        fig.update_layout(template='plotly_dark', xaxis_title='Date', yaxis_title='Volatility',
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True))
        st.plotly_chart(fig, use_container_width=True)

        # Additional Volatility Trend Visualization (Reduced Size)
        st.subheader('📉 Volatility Trend Visualization')
        fig2, ax = plt.subplots(figsize=(4, 3))
        ax.plot(prediction_df.index, prediction_df['Predicted Volatility'], label='Predicted Volatility', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)

# Close the database connection
connection.close()

# Footer
st.markdown('---')
st.caption('Developed with ❤️ by Olalemi John')