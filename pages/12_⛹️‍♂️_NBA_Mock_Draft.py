import pandas as pd
import plotly_express as px
import streamlit as st

def load_data():
    df=pd.read_parquet('https://github.com/aaroncolesmith/nba_draft_db/blob/main/mock_draft_db.parquet?raw=true', engine='pyarrow')
    return df

def mocks_over_time(df):
    df = df.sort_values(['date','draft_order'],ascending=True)
    fig=px.scatter(df,
            x='date',
            y='draft_order',
            color='player',
            title='Mock Drafts Over Time'
            hover_data=['team','source'])
    fig.update_layout(
        template="plotly_white",

    )
    fig.update_traces(showlegend=True,
                    mode='lines+markers',
                    opacity=.5,
                    textposition='top center',
                    marker=dict(size=8,
                                opacity=1,
                                line=dict(width=1,
                                            color='DarkSlateGrey')
                                )
                    )

    st.plotly_chart(fig, use_container_width=True)

def app():
    df = load_data()
    st.title('NBA Mock Draft Database')
    st.write('This is a database of NBA mock drafts for the 2022 NBA Draft')

    mocks_over_time(df)
