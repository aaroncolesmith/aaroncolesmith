from fbprophet import Prophet
from yahooquery import Ticker
import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go
import random

foo = ['a', 'b', 'c', 'd', 'e']
print(random.choice(foo))

def app():
    st.title('Stock Predictions')
    stock_list = ['SPY','TSLA','AAPL','NKE','GME']
    initial_stock = random.choice(stock_list)

    with st.form("input_form"):
        st.write("Pick a stock and see a prediction")
        ticker = st.input_text("Stock Ticker", value=initial_stock)
        years_back = st.number_input("Years Back",min_value=.5, max_value=10, value=2.5, step=0.25)
        years_fwd = st.number_input("Years to Predict",min_value=.5, max_value=10, value=1, step=0.25)
        submit_button = st.form_submit_button(label="Go")

    if submit_button:
        #draw graph
        tickers = Ticker(ticker, asynchronous=True)
        end = datetime.date.today() + datetime.timedelta(days=1)
        start = datetime.date.today() - datetime.timedelta(days=(365*years_back))
        df = tickers.history(start=start, end=end).reset_index()

        df=df[['date','close']]
        df.columns=['ds','y']

        m = Prophet(daily_seasonality = True)
        m.fit(df) 

        fut = m.make_future_dataframe(periods=(int(round(365*years_fwd,0))))
        pred = m.predict(fut)
        d2=df
        d2['ds'] = pd.to_datetime(d2['ds'])
        d2=pd.merge(pred,d2,how='left')

        d2=d2[d2.ds.dt.dayofweek < 5]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d2.ds,
                                y=d2.yhat_lower,
                                opacity=.3,
                                name='Low Threshold',
                                line_color='rgba(75, 0, 130, .33)'))
        fig.add_trace(go.Scatter(x=d2.ds,
                                y=d2.yhat_upper,
                                fill='tonexty',
                                opacity=.3,
                                name='High Threshold',
                                fillcolor='rgba(75, 0, 130, .25)',
                                line_color='rgba(75, 0, 130, .33)'))
        fig.add_trace(go.Scatter(x=d2.ds,
                                y=d2.y,
                                name='Actual Stock Price',
                                mode='markers',
                                marker_color='#626EF6',
                                marker=dict(size=4,
                                            opacity=.5,
                                            line=dict(width=1,
                                                        color='#1320B2'))))
        fig.add_trace(go.Scatter(x=d2.ds,
                                y=d2.yhat,
                                name='Predicted Stock Price',
                                mode='lines',
                                marker_color='#626EF6',
                                marker_line_color='#1320B2',
                                marker_line_width=1.5,
                                opacity=0.75
                                )
                    )
        fig.update_layout(title='Predicted Stock Price Over Time - '+ticker)
        fig.update_yaxes(title='Stock Price')
        fig.update_xaxes(title='Date')

        st.plotly_chart(fig)

if __name__ == "__main__":
    #execute
    app()

