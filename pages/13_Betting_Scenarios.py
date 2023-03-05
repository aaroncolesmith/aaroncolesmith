import pandas as pd
import streamlit as st
import numpy as np
import plotly_express as px
import plotly.io as pio
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


def update_df(df):
    df.columns = [x.lower() for x in df.columns]
    df.columns = [x.replace('.','_') for x in df.columns]

    df=df.sort_values(['start_time','date_scraped'],ascending=[True,True]).reset_index(drop=True) 

    df['ml_home_p'] = df['ml_home'].apply(get_prob)
    df['ml_away_p'] = df['ml_away'].apply(get_prob)

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
            "Soccer of CBB?",
            ('Soccer','CBB')
    )
    if data_select == 'Soccer':
        df = load_data_soccer()
    if data_select == 'CBB':
        df = load_data_cbb()

    d3=pd.concat([pd.merge(df, df.groupby(['id'])['date_scraped'].max(),on=['id','date_scraped']).reset_index(drop=True),pd.merge(df, df.groupby(['id'])['date_scraped'].min(),on=['id','date_scraped']).reset_index(drop=True)])
    d3=d3.drop_duplicates(subset=d3.columns.to_list()[:-1]).reset_index(drop=True).sort_values(['id','date_scraped'],ascending=[True,True])
    d3['ml_home_change']=d3.groupby('id')['ml_home_p'].apply(lambda x: x.diff() / x.shift().abs())
    d3['ml_away_change']=d3.groupby('id')['ml_away_p'].apply(lambda x: x.diff() / x.shift().abs())

    st.markdown('Home Teams in Upcoming Games Meeting Pct Change Threshold')
    # pct_chg_threshold=.02

    pct_chg_threshold = st.number_input('Pct Change Threshold',value=.05)
    st.dataframe(d3.query("status == 'scheduled' & ml_home_change > @pct_chg_threshold")[['id','start_time','league_name','home_team','away_team','ml_home_p','ml_home_change']].sort_values('start_time',ascending=True))

    game_id = st.text_input('Input Game ID',value=189729)

    game_id = int(game_id)

    fig=px.scatter(df.query('id==@game_id'),
            x='date_scraped',
             y=['ml_away','ml_home','draw','total_over_money','ml_away_p','ml_home_p'],
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
    pct_chg_threshold=.02

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
    pct_chg_threshold=.1

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

if __name__ == "__main__":
    #execute
    app()