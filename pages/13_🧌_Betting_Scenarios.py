import pandas as pd
import streamlit as st
import numpy as np
import plotly_express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
pio.templates.default = "simple_white"

import colorsys

@st.cache(suppress_st_warning=True,ttl=3600)
def load_data_soccer():
    df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/df_soccer.parquet?raw=true', engine='pyarrow')
    df=update_df(df)
    return df

@st.cache(suppress_st_warning=True,ttl=3600)
def load_data_cbb():
    df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/df_cbb.parquet?raw=true', engine='pyarrow')
    df=update_df(df)
    return df

@st.cache(suppress_st_warning=True,ttl=3600)
def load_data_nba():
    df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/df_nba.parquet?raw=true', engine='pyarrow')
    df=update_df(df)
    return df

@st.cache(suppress_st_warning=True)
def load_df_teams():
    df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/teams_db.parquet?raw=true', engine='pyarrow')
    return df

def update_df(df):
    df.columns = [x.lower() for x in df.columns]
    df.columns = [x.replace('.','_') for x in df.columns]

    df=df.sort_values(['start_time','date_scraped'],ascending=[True,True]).reset_index(drop=True) 

    df['ml_home_p'] = df['ml_home'].apply(get_prob)
    df['ml_away_p'] = df['ml_away'].apply(get_prob)

    for col in ['ml_away','ml_home','draw','total_over_money','ml_away_p','ml_home_p']:
        df[col]=pd.to_numeric(df[col])

    # Convert datetime column to pandas datetime object
    df['game_time'] = pd.to_datetime(df['start_time'])

    # Convert to desired timezone (PT)
    df['game_time'] = df['game_time'].dt.tz_convert('US/Pacific')

    # Format datetime column as string in desired format
    df['game_time'] = df['game_time'].dt.strftime('%Y-%m-%d %I:%M%p (%Z)')

    df['game_title'] = pd.to_datetime(df['start_time']).dt.strftime('%m/%d') + ' - ' + df['away_team'] +' @ ' + df['home_team']


    return df

def get_prob(a):
    odds = 0
    if a < 0:
        odds = (-a)/(-a + 100)
    else:
        odds = 100/(100+a)

    return odds

def fav_payout(ml):
  return 100/abs(ml)

def dog_payout(ml):
  return ml/100

def get_different_hex_colors(hex_color1, hex_color2):
    # Convert the input colors to RGB values
    rgb_color1 = tuple(int(hex_color1[i:i+2], 16) for i in (1, 3, 5))
    rgb_color2 = tuple(int(hex_color2[i:i+2], 16) for i in (1, 3, 5))
    
    # Convert the RGB colors to HSV values
    hsv_color1 = colorsys.rgb_to_hsv(*rgb_color1)
    hsv_color2 = colorsys.rgb_to_hsv(*rgb_color2)
    
    # Calculate the difference in hue between the input colors
    hue_diff = abs(hsv_color1[0] - hsv_color2[0])
    
    # Calculate the hue values for the five new colors
    new_hue1 = (hsv_color1[0] + 0.2) % 1.0
    new_hue2 = (hsv_color1[0] + 0.4) % 1.0
    new_hue3 = (hsv_color1[0] + 0.6) % 1.0
    new_hue4 = (hsv_color1[0] + 0.8) % 1.0
    new_hue5 = (hsv_color1[0] + hue_diff / 2) % 1.0
    
    # Convert the new HSV values to RGB values
    rgb_color3 = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(new_hue1, 0.8, 0.8))
    rgb_color4 = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(new_hue2, 0.8, 0.8))
    rgb_color5 = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(new_hue3, 0.8, 0.8))
    rgb_color6 = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(new_hue4, 0.8, 0.8))
    rgb_color7 = tuple(round(x * 255) for x in colorsys.hsv_to_rgb(new_hue5, 0.8, 0.8))
    
    # Convert the RGB values to hex codes
    hex_color3 = '#' + ''.join(f'{x:02x}' for x in rgb_color3)
    hex_color4 = '#' + ''.join(f'{x:02x}' for x in rgb_color4)
    hex_color5 = '#' + ''.join(f'{x:02x}' for x in rgb_color5)
    hex_color6 = '#' + ''.join(f'{x:02x}' for x in rgb_color6)
    hex_color7 = '#' + ''.join(f'{x:02x}' for x in rgb_color7)
    
    return [hex_color3, hex_color4, hex_color5, hex_color6, hex_color7]


def plot_game(df,df_teams,game_id):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
 
    home_team_color='#'+df_teams.loc[df_teams.team_id==df.query('id==@game_id')['home_team_id'].min()]['team_primary_color'].values[0]
    away_team_color='#'+df_teams.loc[df_teams.team_id==df.query('id==@game_id')['away_team_id'].min()]['team_primary_color'].values[0]

    colors=get_different_hex_colors(home_team_color, away_team_color)


    fig.add_trace(
        go.Scatter(
            x=df.query('id==@game_id').date_scraped,
            y=df.query('id==@game_id').ml_away,
            mode="markers+lines",
            name=df.query('id==@game_id')['away_team'].min() + ' Money Line (Away)',
            marker=dict(
                color=away_team_color, size=10, line=dict(color=away_team_color, width=2),opacity=.75
            ),
            line = dict(width=4, dash='dash'),
            opacity=0.6,
            customdata=df.query('id==@game_id'),
            hovertemplate=df.query('id==@game_id')['away_team'].min() + ' Money Line %{y} <br>Status: %{customdata[1]}<extra></extra>',

        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.query('id==@game_id').date_scraped,
            y=df.query('id==@game_id').ml_away_p,
            mode="lines",
            name=df.query('id==@game_id')['away_team'].min() + ' Probability (Away)',
            marker=dict(
                color=away_team_color, size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
            ),
            # marker_symbol='diamond',
            line = dict(width=4),
            opacity=0.6,
            hovertemplate=df.query('id==@game_id')['away_team'].min() + ' Probability %{y}<extra></extra>',
        ),
        secondary_y=True,
    )


    fig.add_trace(
        go.Scatter(
            x=df.query('id==@game_id').date_scraped,
            y=df.query('id==@game_id').ml_home,
            mode="markers+lines",
            name=df.query('id==@game_id')['home_team'].min() + ' Money Line (Home)',
            marker=dict(
                color=home_team_color, size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
            ),
            line = dict(width=4, dash='dash'),
            opacity=0.6,
            hovertemplate=df.query('id==@game_id')['home_team'].min() + ' Money Line %{y}<extra></extra>',
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.query('id==@game_id').date_scraped,
            y=df.query('id==@game_id').ml_home_p,
            mode="lines",
            name=df.query('id==@game_id')['home_team'].min() + ' Probability (Home)',
            marker=dict(
                color=home_team_color, size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
            ),
            # marker_symbol='diamond',
            line = dict(width=4),
            opacity=0.6,
            hovertemplate=df.query('id==@game_id')['home_team'].min() + ' Probability %{y}<extra></extra>',
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=df.query('id==@game_id').date_scraped,
            y=df.query('id==@game_id').draw,
            mode="lines",
            name='Money Line - Tie',
            marker=dict(
                color=colors[0], size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
            ),
            # marker_symbol='diamond',
            line = dict(width=4),
            opacity=0.6,
            hovertemplate='Money Line Tie - %{y}<extra></extra>'
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.query('id==@game_id').date_scraped,
            y=df.query('id==@game_id').total,
            mode="lines",
            name='Over / Under',
            marker=dict(
                color=colors[1], size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
            ),
            # marker_symbol='diamond',
            line = dict(width=4),
            opacity=0.6,
            hovertemplate='Over / Under - %{y}<extra></extra>'
        ),
        secondary_y=False,
    )


    if df.query('id==@game_id').total_over_money.max() > 0:
        fig.add_trace(
            go.Scatter(
                x=df.query('id==@game_id').date_scraped,
                y=df.query('id==@game_id').total_over_money,
                mode="lines",
                name='Total Over Money',
                marker=dict(
                    color=colors[2], size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
                ),
                # marker_symbol='diamond',
                line = dict(width=4),
                opacity=0.6,
                hovertemplate='Total Over Money - %{y}<extra></extra>'
            ),
            secondary_y=False,
        )


    if df.query('id==@game_id').ml_home_money.max() > 0:
        fig.add_trace(
            go.Scatter(
                x=df.query('id==@game_id').date_scraped,
                y=df.query('id==@game_id').ml_home_money,
                mode="lines",
                name='ML Home Money',
                marker=dict(
                    color=colors[3], size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
                ),
                # marker_symbol='diamond',
                line = dict(width=4),
                opacity=0.6,
                hovertemplate='ML Home Money - %{y}<extra></extra>'
            ),
            secondary_y=False,
        )

    if df.query('id==@game_id').ml_away_money.max() > 0:
        fig.add_trace(
            go.Scatter(
                x=df.query('id==@game_id').date_scraped,
                y=df.query('id==@game_id').ml_away_money,
                mode="lines",
                name='ML Away Money',
                marker=dict(
                    color=colors[4], size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
                ),
                # marker_symbol='diamond',
                line = dict(width=4),
                opacity=0.6,
                hovertemplate='ML Away Money - %{y}<extra></extra>'
            ),
            secondary_y=False,
        )


    fig.update_layout(hovermode="x", 
                    # template="simple_white",
                    height=600,
                    title=df.query('id==@game_id')['game_title'].min(),
                    legend=dict(yanchor="bottom", 
                        y=-0.35, 
                        xanchor="left", 
                        x=0.0,
                        orientation="h")
                    )

    fig.update_traces(textposition="top right", 
                    mode="lines", 
                    line_shape="spline",
                    opacity=.75,
                        marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                        line = dict(width=4)
                    )

    fig.update_yaxes(title='Win Probability', tickformat=',.1%', secondary_y=True)
    fig.update_yaxes(title='',secondary_y=False)

    st.plotly_chart(fig,use_container_width=False)


##DEFINING SIMULATION FUNCTIONS
def betting_home_above_threshold(d4, pct_chg_threshold):
    payout='ml_home'
    result_title='betting_line_chg_home_result'
    total_title='betting_line_chg_home_total'
    # pct_chg_threshold=.02

    ## Scenario 1 -- Bet 1 dollar on home team money line
    d4.loc[
        (d4.boxscore_total_home_points > d4.boxscore_total_away_points)
        & (d4[payout] < 0)
        & (d4.ml_home_change > pct_chg_threshold),
        result_title,
    ] = fav_payout(d4[payout])

    d4.loc[
        (d4.boxscore_total_home_points > d4.boxscore_total_away_points)
        & (d4[payout] > 0)
        & (d4.ml_home_change > pct_chg_threshold),
        result_title,
    ] = dog_payout(d4[payout])

    d4.loc[
        (d4.boxscore_total_home_points <= d4.boxscore_total_away_points)
        & (d4.ml_home_change > pct_chg_threshold),
        result_title,
    ] = -1
    d4[result_title] = d4[result_title].fillna(0)
    d4[total_title] = d4[result_title].cumsum()
    
    return d4

def betting_away_above_threshold(d4, pct_chg_threshold):

    payout='ml_away'
    result_title='betting_line_chg_away_result'
    total_title='betting_line_chg_away_total'
    # pct_chg_threshold=.1

    ## Scenario 1 -- Bet 1 dollar on away team money line
    d4.loc[
        (d4.boxscore_total_away_points > d4.boxscore_total_home_points)
        & (d4[payout] < 0)
        & (d4.ml_away_change > pct_chg_threshold),
        result_title,
    ] = fav_payout(d4[payout])

    d4.loc[
        (d4.boxscore_total_away_points > d4.boxscore_total_home_points)
        & (d4[payout] > 0)
        & (d4.ml_away_change > pct_chg_threshold),
        result_title,
    ] = dog_payout(d4[payout])

    d4.loc[
        (d4.boxscore_total_away_points <= d4.boxscore_total_home_points)
        & (d4.ml_away_change > pct_chg_threshold),
        result_title,
    ] = -1
    d4[result_title] = d4[result_title].fillna(0)
    d4[total_title] = d4[result_title].cumsum()

    return d4

def betting_both_above_threshold(d4, pct_chg_threshold):

    ## Betting either home or away if they meet the threshold
    payout='ml_away'
    result_title='betting_line_chg_both_result'
    total_title='betting_line_chg_both_total'
    # pct_chg_threshold=.1

    ## Scenario 1 -- Bet 1 dollar on away team money line
    d4.loc[
        (d4.boxscore_total_away_points > d4.boxscore_total_home_points)
        & (d4[payout] < 0)
        & (d4.ml_away_change > pct_chg_threshold),
        result_title,
    ] = fav_payout(d4[payout])

    d4.loc[
        (d4.boxscore_total_away_points > d4.boxscore_total_home_points)
        & (d4[payout] > 0)
        & (d4.ml_away_change > pct_chg_threshold),
        result_title,
    ] = dog_payout(d4[payout])

    d4.loc[
        (d4.boxscore_total_away_points <= d4.boxscore_total_home_points)
        & (d4.ml_away_change > pct_chg_threshold),
        result_title,
    ] = -1

    payout='ml_home'
    d4.loc[
        (d4.boxscore_total_home_points > d4.boxscore_total_away_points)
        & (d4[payout] < 0)
        & (d4.ml_home_change > pct_chg_threshold),
        result_title,
    ] = fav_payout(d4[payout])

    d4.loc[
        (d4.boxscore_total_home_points > d4.boxscore_total_away_points)
        & (d4[payout] > 0)
        & (d4.ml_home_change > pct_chg_threshold),
        result_title,
    ] = dog_payout(d4[payout])

    d4.loc[
        (d4.boxscore_total_home_points <= d4.boxscore_total_away_points)
        & (d4.ml_home_change > pct_chg_threshold),
        result_title,
    ] = -1
    d4[result_title] = d4[result_title].fillna(0)
    d4[total_title] = d4[result_title].cumsum()

    return d4

def betting_all_home(d4):
  d4.loc[
      (d4.boxscore_total_home_points > d4.boxscore_total_away_points)
      & (d4.ml_home < 0),
      "betting_all_home_result",
  ] = fav_payout(d4.ml_home)
  d4.loc[
      (d4.boxscore_total_home_points > d4.boxscore_total_away_points)
      & (d4.ml_home > 0),
      "betting_all_home_result",
  ] = dog_payout(d4.ml_home)
  d4["betting_all_home_result"] = d4["betting_all_home_result"].fillna(-1)
  d4["betting_all_home_total"] = d4["betting_all_home_result"].cumsum()
  return d4

def betting_all_away(d4):
  d4.loc[
      (d4.boxscore_total_home_points < d4.boxscore_total_away_points)
      & (d4.ml_away < 0),
      "betting_all_away_result",
  ] = fav_payout(d4.ml_away)
  d4.loc[
      (d4.boxscore_total_home_points < d4.boxscore_total_away_points)
      & (d4.ml_away > 0),
      "betting_all_away_result",
  ] = dog_payout(d4.ml_away)
  d4["betting_all_away_result"] = d4["betting_all_away_result"].fillna(-1)
  d4["betting_all_away_total"] = d4["betting_all_away_result"].cumsum()
  return d4


def app():


    st.title('Betting Scenarios')
    df_teams = load_df_teams()

    data_select = st.sidebar.radio(
            "Which sport?",
            ('Soccer','CBB','NBA')
    )
    if data_select == 'Soccer':
        df = load_data_soccer()
    if data_select == 'CBB':
        df = load_data_cbb()
    if data_select == 'NBA':
        df = load_data_nba()

    date_filter=st.date_input("Games for date:",
        datetime.date.today(),
        min_value=pd.to_datetime(df.start_time).dt.date.min(),
        max_value=pd.to_datetime(df.start_time).dt.date.max())

    d3=pd.concat([pd.merge(df, df.groupby(['id'])['date_scraped'].max(),on=['id','date_scraped']).reset_index(drop=True),pd.merge(df, df.groupby(['id'])['date_scraped'].min(),on=['id','date_scraped']).reset_index(drop=True)])
    d3=d3.drop_duplicates(subset=d3.columns.to_list()[:-1]).reset_index(drop=True).sort_values(['id','date_scraped'],ascending=[True,True])
    d3['ml_home_change']=d3.groupby('id')['ml_home_p'].apply(lambda x: x.diff() / x.shift().abs())
    d3['ml_away_change']=d3.groupby('id')['ml_away_p'].apply(lambda x: x.diff() / x.shift().abs())


    df_todays_games = pd.merge(d3, d3.groupby(["id"])["date_scraped"].max(), on=["id", "date_scraped"])
    df_todays_games = df_todays_games.loc[pd.to_datetime(df_todays_games['start_time']).dt.tz_convert('US/Pacific').dt.date==date_filter].sort_values("start_time", ascending=True).reset_index(drop=True)[['id','game_time','home_team','away_team','ml_home_p','ml_away_p','ml_home_change','ml_away_change']]
    if df_todays_games.index.size > 0:
        
        ## FORMATTING THE PERCENTAGE NUMBERS TO LOOK BETTER -- BUT THIS MESSES UP THE SORT
        # for col in ['ml_home_p','ml_away_p']:
        #     df_todays_games[col]=df_todays_games[col].apply(lambda x: '{:.2%}'.format(x))

        # for col in ['ml_home_change','ml_away_change']:
            # df_todays_games[col]=df_todays_games[col].apply(lambda x: '{:+.2%}'.format(x))

        df_todays_games.columns=['ID','Game Time','Home','Away','Home Probability','Away Probability','Home Probability % Change','Away Probability % Change']

        st.dataframe(df_todays_games)

        ## plot today's first game as well as selector for any game
        # game_dict = dict(
        #         zip(
        #             df.loc[pd.to_datetime(df['start_time']).dt.tz_convert('US/Pacific').dt.date==date_filter]['game_title'].unique(),
        #             df.loc[pd.to_datetime(df['start_time']).dt.tz_convert('US/Pacific').dt.date==date_filter]['game_id'].unique()
        #             )
        #             )

        # group by game_id and game_title, and get unique values
        df_unique = df.loc[pd.to_datetime(df['start_time']).dt.tz_convert('US/Pacific').dt.date==date_filter].groupby(['game_id', 'game_title']).agg({'start_time': 'min'}).reset_index()

        # sort by start_time
        df_unique = df_unique.sort_values(by='start_time')

        # create dictionary from the values
        game_dict = {row['game_title']: row['game_id'] for _, row in df_unique[['game_title', 'game_id']].iterrows()}
        
        game_select = st.selectbox('Select a game to view the odds over time: ', game_dict.keys())

        game_select_id = game_dict[game_select]

        plot_game(df,df_teams,game_select_id)
    else:
        st.write('No games on selected date 😔')

    st.markdown('Upcoming Games Meeting Pct Change Threshold')
    pct_chg_threshold = st.number_input('Pct Change Threshold',value=.05)

    df_threshold_games_h = d3.query("status == 'scheduled' & ml_home_change > @pct_chg_threshold")[['id','game_time','home_team','away_team','ml_home_p','ml_home','ml_home_change']]
    df_threshold_games_h.columns=['ID','Game Time','Team','Opponent','Probability','Money Line','Probability % Change']
    df_threshold_games_h['Home / Away'] = 'Home'

    df_threshold_games_a = d3.query("status == 'scheduled' & ml_away_change > @pct_chg_threshold")[['id','game_time','away_team','home_team','ml_away_p','ml_away','ml_away_change']]
    df_threshold_games_a.columns=['ID','Game Time','Team','Opponent','Probability','Money Line','Probability % Change']
    df_threshold_games_a['Home / Away'] = 'Away'

    df_threshold_games = pd.concat([df_threshold_games_h,df_threshold_games_a]).sort_values('Game Time',ascending=True).reset_index(drop=True)

    del df_threshold_games_h
    del df_threshold_games_a

    st.dataframe(df_threshold_games)

    initial_game_id=df_threshold_games.head(1)['ID'].min()
    game_id = st.text_input('Input Game ID',value=initial_game_id)

    game_id = int(game_id)
    plot_game(df,df_teams,game_id)

    
    d4=d3.query('status=="complete"').sort_values('start_time',ascending=True).copy()
    d4['start_time'] = pd.to_datetime(d4['start_time'])

    # start_date = pd.to_datetime('today') - pd.Timedelta(days=30)
    # start_date = start_date.date()

    start_date=st.date_input("How far back to test betting scenarios:",
        pd.to_datetime('today') - pd.Timedelta(days=30),
        min_value=pd.to_datetime(d4.start_time).dt.date.min(),
        max_value=pd.to_datetime(d4.start_time).dt.date.max())

    d4 = d4.loc[d4.start_time.dt.date > start_date].reset_index(drop=True)

    for col in d4.columns:
        if col.startswith('betting_'):
            del d4[col]


    d4 = betting_home_above_threshold(d4, pct_chg_threshold)
    d4 = betting_away_above_threshold(d4, pct_chg_threshold)
    d4 = betting_both_above_threshold(d4, pct_chg_threshold)
    d4=betting_all_home(d4)

    d4['score'] = d4['boxscore_total_away_points'].astype('int').astype('str') + '-'+d4['boxscore_total_home_points'].astype('int').astype('str')

    betting_cols_result = [col for col in d4.columns if col.startswith('betting_') and col.endswith('_result')]
    betting_cols_total = [col for col in d4.columns if col.startswith('betting_') and col.endswith('_total')]
    betting_dict = {result_col: total_col for result_col, total_col in zip(betting_cols_result, betting_cols_total)}


    fig=go.Figure()
    for key, value in betting_dict.items():
        fig.add_trace(go.Scatter(
            x=d4.loc[d4[key]!=0].start_time,
            y=d4.loc[d4[key]!=0][value],
            mode='markers+lines',
            name=key.replace('_result','').replace('_',' ').title(),
            customdata=d4.loc[d4[key]!=0][['id','game_time','game_title','score',key]],
            hovertemplate='<b>'+key.replace('_result','').replace('_',' ').title()+': %{y:.1f}</b><br>Date: %{customdata[1]}<br>%{customdata[2]}<br>ID: %{customdata[0]} | Score: %{customdata[3]} | Bet Result: %{customdata[4]:.2}<extra></extra>'
        )
        )

    fig.update_layout(
        # hovermode="x", 
                    # template="simple_white",
                    height=600,
                    # title=df.query('id==@game_id')['game_title'].min(),
                    legend=dict(yanchor="bottom", 
                        y=-0.35, 
                        xanchor="left", 
                        x=0.0,
                        orientation="h")
                    )

    fig.update_traces(textposition="top right", 
                    mode="markers+lines", 
                    line_shape="spline",
                    opacity=.75,
                        marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                        line = dict(width=4)
                    )
    st.plotly_chart(fig,use_container_width=True)


    fig=px.scatter(d4.query("betting_line_chg_home_result != 0 | betting_line_chg_away_result != 0 | betting_line_chg_both_total != 0"),
            x='start_time',
            y=['betting_line_chg_home_total',
                'betting_line_chg_away_total',
                'betting_line_chg_both_total'],
            hover_data=['home_team','boxscore_total_home_points','away_team','boxscore_total_away_points',
                        'id',
                        #  'total_over_money','total_under_money',
                        'ml_home','ml_away','betting_line_chg_home_result',
                        'ml_home_change'])
    # fig.update_layout(hovermode="x")

    fig.update_layout(
        template="plotly_white"
    )


    fig.update_layout( 
                    template="plotly_white",
                    height=600)

    fig.update_traces(textposition="top right", 
                    mode="lines+markers", 
                    # line_shape="spline",
                    opacity=.75,
                        marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                        line = dict(width=4),
                    # hovertemplate=('%{x}<br>$ Amt: %{y:$,.2s} <br>GEOs: %{customdata[0]} / Countries: %{customdata[1]}')
                    )

    st.plotly_chart(fig,use_container_width=True)







    c1,c2=st.columns(2)
    c1.subheader('Betting Home team')
    c1.write('Number of wins: '+str(d4.query("betting_line_chg_home_result > 0").index.size))
    c1.write('Number of losses: '+str(d4.query("betting_line_chg_home_result < 0").index.size))
    c1.write('Money won: '+str(
        round(d4.query("betting_line_chg_home_result > 0").betting_line_chg_home_result.sum(),2)
        )
        )
    c1.write('Money lost: '+str(
        round(d4.query("betting_line_chg_home_result < 0").betting_line_chg_home_result.sum(),2)
        ))

    if d4.query("betting_line_chg_home_result > 0").betting_line_chg_home_result.sum() + d4.query("betting_line_chg_home_result < 0").betting_line_chg_home_result.sum() >= 0:
        c1.success('Net Results: '+str(
            round(d4.query("betting_line_chg_home_result > 0").betting_line_chg_home_result.sum() + d4.query("betting_line_chg_home_result < 0").betting_line_chg_home_result.sum(),2)
        ))
    if d4.query("betting_line_chg_home_result > 0").betting_line_chg_home_result.sum() + d4.query("betting_line_chg_home_result < 0").betting_line_chg_home_result.sum() < 0:
        c1.warning('Net Results: '+str(
            round(d4.query("betting_line_chg_home_result > 0").betting_line_chg_home_result.sum() + d4.query("betting_line_chg_home_result < 0").betting_line_chg_home_result.sum(),2)
        ))


    c2.subheader('Betting Away team')
    c2.write('Number of wins: '+str(d4.query("betting_line_chg_away_result > 0").index.size))
    c2.write('Number of losses: '+str(d4.query("betting_line_chg_away_result < 0").index.size))
    c2.write('Money won: '+str(
        round(d4.query("betting_line_chg_away_result > 0").betting_line_chg_away_result.sum(),2)
        )
        )
    c2.write('Money lost: '+str(
        round(d4.query("betting_line_chg_away_result < 0").betting_line_chg_away_result.sum(),2)
        ))

    if d4.query("betting_line_chg_away_result > 0").betting_line_chg_away_result.sum() + d4.query("betting_line_chg_away_result < 0").betting_line_chg_away_result.sum() >= 0:
        c2.success('Net Results: '+str(
            round(d4.query("betting_line_chg_away_result > 0").betting_line_chg_away_result.sum() + d4.query("betting_line_chg_away_result < 0").betting_line_chg_away_result.sum(),2)
        ))
    if d4.query("betting_line_chg_away_result > 0").betting_line_chg_away_result.sum() + d4.query("betting_line_chg_away_result < 0").betting_line_chg_away_result.sum() < 0:
        c2.warning('Net Results: '+str(
            round(d4.query("betting_line_chg_away_result > 0").betting_line_chg_away_result.sum() + d4.query("betting_line_chg_away_result < 0").betting_line_chg_away_result.sum(),2)
        ))


    st.write(df.query('id==@game_id'))

if __name__ == "__main__":
    #execute
    app()