import pandas as pd
import plotly_express as px
import streamlit as st

def load_data():
    df=pd.read_parquet('https://github.com/aaroncolesmith/nba_draft_db/blob/main/mock_draft_db.parquet?raw=true', engine='pyarrow')
    return df

def get_color_map():
  df=pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/bovada/master/color_map.csv')
  trans_df = df[['team','primary_color']].set_index('team').T
  color_map=trans_df.to_dict('index')['primary_color']

  # word_freq.update({'before': 23})
  return color_map

def mocks_over_time(df):
    df = df.sort_values(['date','draft_order'],ascending=True)
    fig=px.scatter(df,
            x='date',
            y='draft_order',
            color='player',
            title='Mock Drafts Over Time',
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

def player_team_combo(df):
    dviz=df.loc[df.team!='None'].groupby(['team','player']).agg(times_picked=('draft_order','size'),
                                  avg_pick=('draft_order','mean')).reset_index()
    fig=px.bar(dviz,
        x=dviz['team']+' - '+dviz['player'],
        y='times_picked',
        color_discrete_map=color_map,
        color='team',
        title='Team and Player Drafted Combos')   
    fig.update_layout(
        template="plotly_white",

    )
    fig.update_xaxes(categoryorder='total descending',
                    title='Team / Player') 
    st.plotly_chart(fig, use_container_width=True)


color_map = get_color_map()

def app():
    df = load_data()
    st.title('NBA Mock Draft Database')
    st.write('This is a database of NBA mock drafts for the 2022 NBA Draft')

    mocks_over_time(df)
