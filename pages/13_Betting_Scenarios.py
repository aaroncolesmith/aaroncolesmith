import pandas as pd
import streamlit as st
import numpy as np
import plotly_express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
pio.templates.default = "simple_white"

@st.cache(suppress_st_warning=True)
def load_data_soccer():
    df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/df_soccer.parquet?raw=true', engine='pyarrow')
    df=update_df(df)
    return df

@st.cache(suppress_st_warning=True)
def load_data_cbb():
    df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/df_cbb.parquet?raw=true', engine='pyarrow')
    df=update_df(df)
    return df

@st.cache(suppress_st_warning=True)
def load_data_nba():
    df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/df_nba.parquet?raw=true', engine='pyarrow')
    df=update_df(df)
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





def app():


    st.title('Betting Scenarios')
    st.markdown('TBD')

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

    d3=pd.concat([pd.merge(df, df.groupby(['id'])['date_scraped'].max(),on=['id','date_scraped']).reset_index(drop=True),pd.merge(df, df.groupby(['id'])['date_scraped'].min(),on=['id','date_scraped']).reset_index(drop=True)])
    d3=d3.drop_duplicates(subset=d3.columns.to_list()[:-1]).reset_index(drop=True).sort_values(['id','date_scraped'],ascending=[True,True])
    d3['ml_home_change']=d3.groupby('id')['ml_home_p'].apply(lambda x: x.diff() / x.shift().abs())
    d3['ml_away_change']=d3.groupby('id')['ml_away_p'].apply(lambda x: x.diff() / x.shift().abs())

    df_todays_games = pd.merge(d3, d3.groupby(["id"])["date_scraped"].max(), on=["id", "date_scraped"])[pd.to_datetime(d3["start_time"]).dt.date == datetime.date.today()].sort_values("start_time", ascending=True).reset_index(drop=True)[['id','game_time','home_team','away_team','ml_home_p','ml_away_p','ml_home_change','ml_away_change']]

    for col in ['ml_home_p','ml_away_p']:
        df_todays_games[col]=df_todays_games[col].apply(lambda x: '{:.2%}'.format(x))

    for col in ['ml_home_change','ml_away_change']:
        df_todays_games[col]=df_todays_games[col].apply(lambda x: '{:+.2%}'.format(x))

    df_todays_games.columns=['ID','Game Time','Home','Away','Home Probability','Away Probability','Home Probability % Change','Away Probability % Change']

    st.write('Today\'s Games')
    st.dataframe(df_todays_games)



    st.markdown('Home Teams in Upcoming Games Meeting Pct Change Threshold')
    pct_chg_threshold = st.number_input('Pct Change Threshold',value=.05)
    st.dataframe(d3.query("status == 'scheduled' & ml_home_change > @pct_chg_threshold")[['id','start_time','league_name','home_team','away_team','ml_home_p','ml_home_change']].sort_values('start_time',ascending=True))
    initial_game_id=d3.query("status == 'scheduled' & ml_home_change > @pct_chg_threshold")[['id','start_time','league_name','home_team','away_team','ml_home_p','ml_home_change']].sort_values('start_time',ascending=True).head(1)['id'].min()
    game_id = st.text_input('Input Game ID',value=initial_game_id)

    game_id = int(game_id)


    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.query('id==@game_id').date_scraped,
            y=df.query('id==@game_id').ml_away,
            mode="markers+lines",
            name=df.query('id==@game_id')['away_team'].min() + ' Money Line (Away)',
            marker=dict(
                color="SpringGreen", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
            mode="lines+markers",
            name=df.query('id==@game_id')['away_team'].min() + ' Probability (Away)',
            marker=dict(
                color="SpringGreen", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
                color="SpringGreen", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
            mode="lines+markers",
            name=df.query('id==@game_id')['home_team'].min() + ' Probability (Home)',
            marker=dict(
                color="SpringGreen", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
            mode="lines+markers",
            name='Money Line - Tie',
            marker=dict(
                color="Blue", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
            mode="lines+markers",
            name='Over / Under',
            marker=dict(
                color="Blue", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
                mode="lines+markers",
                name='Total Over Money',
                marker=dict(
                    color="Blue", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
                mode="lines+markers",
                name='ML Home Money',
                marker=dict(
                    color="Blue", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
                mode="lines+markers",
                name='ML Away Money',
                marker=dict(
                    color="Blue", size=10, line=dict(color="DarkSlateGrey", width=2),opacity=.75
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
                    mode="lines+markers", 
                    line_shape="spline",
                    opacity=.75,
                        marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                        line = dict(width=4)
                    )

    fig.update_yaxes(title='Win Probability', tickformat=',.1%', secondary_y=True)
    fig.update_yaxes(title='',secondary_y=False)

    st.plotly_chart(fig,use_container_width=False)









    fig=px.scatter(df.query('id==@game_id'),
            x='date_scraped',
            y=['ml_away','ml_home','draw','total',
            'total_over_money','ml_home_money','ml_away_money',
            
            'ml_away_p','ml_home_p'],
            hover_data=['home_team','away_team','status']
    )

    fig.update_traces(
        # texttemplate = "%{text:}",
                    textposition='top right',
                    mode='lines+markers+text',
        line_shape='spline')
    fig.update_layout(
        template="plotly_white"
    )


    fig.update_layout(hovermode="x", 
                    template="plotly_white",
                    height=600)

    fig.update_traces(textposition="top right", 
                    mode="lines+markers", 
                    line_shape="spline",
                    opacity=.75,
                        marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                        line = dict(width=4),
                    # hovertemplate=('%{x}<br>$ Amt: %{y:$,.2s} <br>GEOs: %{customdata[0]} / Countries: %{customdata[1]}')
                    )

    st.plotly_chart(fig,use_container_width=True)

    d4=d3.query('status=="complete"').sort_values('start_time',ascending=True).copy()

    for col in d4.columns:
        if col.startswith('betting_'):
            del d4[col]


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


    fig=px.scatter(d4.query("betting_line_chg_home_result != 0 | betting_line_chg_away_result != 0"),
            x='start_time',
            y=['betting_line_chg_home_total',
                'betting_line_chg_away_total'],
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