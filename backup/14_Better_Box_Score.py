import pandas as pd
import streamlit as st
import plotly_express as px
import datetime
import requests
from bs4 import BeautifulSoup


def app():
    st.markdown('# Better Box Score')
    date_filter=st.date_input("Games for date:",
        datetime.date.today(),
        # min_value=pd.to_datetime(df.start_time).dt.date.min(),
        # max_value=pd.to_datetime(df.start_time).dt.date.max()
        )
    date_filter=date_filter.strftime('%Y-%m-%d')

    if date_filter == datetime.date.today().strftime('%Y-%m-%d'):
        url='https://plaintextsports.com/'
    else:
        url = f'https://plaintextsports.com/all/{date_filter}/'
    st.write(url)
    r=requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    url_list=[]
    league_list=[]
    away_team_list=[]
    home_team_list=[]

    url_list = [a['href'] for a in soup.find_all('a', href=True) if '+----' in str(a)]
    league_list = [str(a['href']).split('/')[1].replace('-', ' ').title() if str(a['href']).split('/')[1] in ['premier-league','champions-league'] else str(a['href']).split('/')[1].upper() for a in soup.find_all('a', href=True) if '+----' in str(a)]
    away_team_list = [str(a['href']).split('/')[-1].split('-')[0].replace('#', '').upper() for a in soup.find_all('a', href=True) if '+----' in str(a)]
    home_team_list = [str(a['href']).split('/')[-1].split('-')[1].replace('#', '').upper() for a in soup.find_all('a', href=True) if '+----' in str(a)]

    d=pd.DataFrame({
        'url':url_list,
        'league':league_list,
        'away':away_team_list,
        'home':home_team_list
    })
    d['game_string'] = d['league'] + ' - ' +d['away'] + ' @ ' + d['home']

    game_list = d['game_string'].tolist()
    game_select=st.selectbox('Select a game -', game_list)

    st.write(d.head(3))

    game_url = d.query("game_string == @game_select")['url'].values[0]
    game_url = f'https://plaintextsports.com{game_url}'
    st.write(game_url)
    r=requests.get(game_url)
    soup = BeautifulSoup(r.content, 'html.parser')
    st.write(r.text)

if __name__ == "__main__":
    #execute
    app()
