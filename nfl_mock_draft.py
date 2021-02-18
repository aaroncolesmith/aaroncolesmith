import pandas as pd
import streamlit as st
import plotly_express as px
import numpy as np

# st.set_page_config(layout="wide" )


@st.cache(suppress_st_warning=True)
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/nfl_mock_draft_db.csv')
    return df

def ga(event_category, event_action, event_label):
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec='+event_category+'&ea='+event_action+'&el='+event_label+'">',unsafe_allow_html=True)

def app():
    st.image('./images/nfl_draft.jpeg', use_column_width=True)
    st.title('NFL Mock Draft Database')

    st.write('By scraping the results of multiple NFL Mock Drafts, can we start to see trends that will help understand how teams will draft?')
    ga('NFL Mock Draft Database', 'Page Load', 'Page Load')
    df=load_data()

    d1=pd.DataFrame()
    d2=pd.DataFrame()
    for i in df.player.unique():
      d1 = pd.concat([d1, df.loc[df.player == i].iloc[0:5]])
      d2 = pd.concat([d2, df.loc[df.player == i].iloc[5:10]])

    d3=pd.merge(d1.groupby(['player']).agg({'pick':'mean'}).reset_index(), d2.groupby(['player']).agg({'pick':'mean'}).reset_index(), how='left', suffixes=('_0','_1'), left_on='player', right_on='player')
    d3['chg'] = d3['pick_0'] - d3['pick_1']
    d3['pct_chg']=d3['chg'] / d3['pick_1']
    d3=d3.sort_values('chg',ascending=True)


    col1, col2 = st.beta_columns(2)
    col1.success("### Players Rising :fire:")
    for i, r in d3.head(5).iterrows():
        col1.write(r['player']+' - trending ' + str(round(abs(r['chg']),2)) + ' picks earlier')
        # col1.write(r['player'] + ' | Avg. Pick changed ' + str(round(r['chg'],2)) +' picks')
    # col1.write(d3.head(5))


    col2.warning("### Players Falling 🧊")
    for i, r in d3.tail(5).iterrows():
        col2.write(r['player'] + ' - trending ' + str(round(r['chg'],2)) +' picks later')


    option = st.radio('View all or most recent mock drafts?',('All','Most Recent'))
    if option == 'All':
        d2 = df

    if option == 'Most Recent':
        num=st.number_input('How many recent mock drafts?', min_value=1, max_value=50, value=10)
        df_latest=pd.DataFrame()
        for i in df.player.unique():
          df_latest = pd.concat([df_latest, df.loc[df.player == i].head(num)])
          d2=df_latest

    d=d2.groupby(['team','team_img','player']).agg({'pick':['min','mean','median','size']}).reset_index()
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

    d=d.sort_values('avg_pick',ascending=True)
    fig=px.scatter(d,
          x='player',
          y='avg_pick',
           size='cnt',
          color='team',
          height=600,
          title='Avg. Pick Placement by Player / Team')
    fig.update_xaxes(title='Player')
    fig.update_yaxes(title='Avg. Draft Position')
    st.plotly_chart(fig, use_container_width=True)

    fig=px.bar(df.groupby(['team','player']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False).head(15),
       x=df.groupby(['team','player']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False).head(15).team + ' - '+df.groupby(['team','player']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False).head(15).player,
       y='cnt',
       title='Most Common Team - Player Pairings')
    fig.update_xaxes(title='Team & Player Pairing')
    fig.update_yaxes(title='Count')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(d2.loc[d2.player.isin(d2.groupby('player').agg({'pick':'mean'}).reset_index().sort_values('pick',ascending=True).head(15)['player'])], x="player", y="pick", points="all", hover_data=['team','date','source'], title='Distribution of Draft Position by Player', width=1600)
    fig.update_xaxes(title='Player')
    fig.update_yaxes(title='Draft Position')
    st.plotly_chart(fig, use_container_width=True)


    a=d2.team.unique()
    a=np.insert(a,0,'')

    team=st.selectbox('Select a team -', a)


    if len(team) > 0:
        ga('NFL Mock Draft Database', 'Team Select', team)
        fig=px.bar(d2.loc[df.team == team].groupby('player').size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False).head(10),
           x='player',
           y='cnt',
           title='Number of Times a Player is Mocked to '+team)
        fig.update_xaxes(title='Player')
        fig.update_yaxes(title='Number of Occurences')
        st.plotly_chart(fig, use_container_width=True)
