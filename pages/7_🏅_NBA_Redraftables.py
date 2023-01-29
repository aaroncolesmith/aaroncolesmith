import pandas as pd
import numpy as np
import plotly_express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from datetime import datetime
import requests
import streamlit as st

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog'
    )

def track_google_analytics_event(event_category, event_action, event_label):
    tracking_id = 'UA-18433914-1'
    clientid_str = str(datetime.now())
    tracking_url = 'https://www.google-analytics.com/collect?v=1&t=event&tid='+tracking_id+'&cid='+clientid_str+'&ec='+event_category+'&ea='+event_action+'&el='+event_label+'&aip=1'
    requests.post(tracking_url)


def load_data():

    df = pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/nba-redraftables/master/data/nba_draft_data.csv')

    return df

def redraft_data(year, df):
    d = df.loc[df.Year == year.astype('int')].sort_values('WSPS',ascending=False)
    d = d.reset_index(drop=True)
    d = d.reset_index(drop=False)
    d = d.rename(columns={'index':'Redraft'})
    d['Redraft'] = d['Redraft']+1

    d.loc[(d.Pk - d.Redraft) < 0,'Pick_Analysis'] = 'Bad Pick'
    d.loc[(d.Pk - d.Redraft) >= 0,'Pick_Analysis'] = 'Good Pick'

    return d

def draft_scatter(year, df):
    d=redraft_data(year, df)

    good=d.loc[d.Pick_Analysis == 'Good Pick'].index.size
    bad=d.loc[d.Pick_Analysis == 'Bad Pick'].index.size
    WSPS=round(d.WSPS.sum(),1)


    fig=px.scatter(d,
               x='Pk',
               y='Redraft',
               hover_data=['Player','Tm','College','WS','WSPS'],
               color='Pick_Analysis',
               title = 'Redraft for NBA Draft - Year '+ str(year) +
                   ' <br>'+ str(good) + ' Good Picks - '+ str(bad) + ' Bad Picks <br>'+
                   str(WSPS) +' Total WSPS for Draft')

    fig.update_traces(mode='markers',
                      marker=dict(size=12,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    fig.update_layout(
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})


    fig.data[0]['marker'].update(color='#00CC99') #Good Pick
    fig.data[1]['marker'].update(color='#CC0000') #Bad Pick

    st.plotly_chart(fig)

def redraft_bar(year, df):
    d = redraft_data(year, df)
    fig=px.bar(
        d.loc[d.WSPS != 0].sort_values('WSPS',ascending=True),
           y='Player',
           x='WSPS',
               title='Win Shares Per Season (WSPS) by Player',
               hover_data=['Pk','Redraft','Tm','College'],
           orientation='h'
          )

    fig.update_yaxes(tickfont=dict(size=8))
    fig.update_xaxes(tickfont=dict(size=8))

    fig.update_layout(height=800)

    st.plotly_chart(fig)


def app():
    st.write("""
    # Welcome to NBA Redraftables!
    Hindsight is 20/20 -- with year's of statistical data, how would each year be redrafted? Select a year to see!""")
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=nba_redraft&ea=page_load">',unsafe_allow_html=True)
    # track_google_analytics_event('nba_redraftables', 'main', 'page_load')
    df = load_data()
    year_list = df.Year.unique().astype('str')
    year_list=np.insert(year_list,0,'')
    year = st.selectbox('Select a year to view the draft - ',year_list,0)
    if len(year) > 0:
        st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=nba_redraft&ea='+str(year)+'">',unsafe_allow_html=True)
        # track_google_analytics_event('nba_redraftables', 'year', str(year))
        draft_scatter(year, df)
        redraft_bar(year, df)

if __name__ == "__main__":
    #execute
    app()
