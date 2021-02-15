import pandas as pd
import streamlit as st
import plotly_express as px


@st.cache(suppress_st_warning=True)
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/nfl_mock_draft_db.csv')
    return df


def app():
    st.title('NFL Mock Draft Database')
    st.write('By scraping the results of multiple NFL Mock Drafts, can we start to see trends that will help understand how teams will draft?')
    df=load_data()
    d=df.groupby(['team','team_img','player']).agg({'pick':['min','mean','median','size']}).reset_index()
    d.columns=['team','team_img','player','min_pick','avg_pick','median_pick','cnt']
    fig=px.scatter(d,
          x='cnt',
          y='avg_pick',
           color='team',
           title='# of Times a Player is Mocked to a Given Pick / Team',
          hover_data=['player'])
    fig.update_xaxes(title='# of Occurences')
    fig.update_yaxes(title='Avg. Draft Pick')
    fig.update_traces(mode='markers',
                      marker=dict(size=8,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)
