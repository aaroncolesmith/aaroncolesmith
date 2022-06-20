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


def avg_pick_by_player(df):
    tmp=df.loc[df.team != 'None'].groupby(['player','team']).size().to_frame('times_picked').reset_index()
    tmp['times_picked_by_team'] = tmp['team'] + ' - ' + tmp['times_picked'].astype('str')
    tmp = tmp.sort_values('times_picked',ascending=False)

    fig=px.scatter(
    pd.merge(pd.merge(pd.merge(df.groupby(['player']).agg(
        times_picked=('draft_order','size'),
        avg_pick=('draft_order','mean'),
        big_board=('bigboard_order','mean')
    ).sort_values('avg_pick',ascending='True').reset_index(),
    df.loc[df.comparisons!=''].groupby(['player','comparisons']).size().reset_index().groupby(['player']).agg(comparisons=('comparisons',lambda x:','.join(x))).reset_index(),
    how='inner'),
    tmp.groupby(['player']).agg(times_picked_by_team = ('times_picked_by_team',lambda x:'<br>'.join(x))).reset_index(),
    how='inner'),
    df.loc[df.source == 'The Ringer'].groupby(['player']).first().reset_index()[['player','position','height','weight','age','school','year']],
    how='inner'),
                x='player',
                y='avg_pick',
                color='player',
                title='Avg Draft Position by Player',
                hover_data=['times_picked_by_team','comparisons','position','height','weight','age','school','year'])
    fig.update_traces(mode='markers',
                    marker=dict(size=12,
                                opacity=.75,
                                line=dict(width=1,
                                            color='DarkSlateGrey')))

    custom_template = '<b>%{x}</b><br><b>Height / Weight:</b> %{customdata[3]}/%{customdata[4]}<br><b>School:</b> %{customdata[6]}<br><b>Avg. Draft Position: </b>%{y:.4}<br><br><b>Player Comparisons:</b> %{customdata[1]}<br><br><b>Times Picked by Team:</b><br>%{customdata[0]}'
    fig.update_traces(hovertemplate=custom_template)

    fig.update_layout(
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


color_map = get_color_map()

def app():
    df = load_data()
    st.title('NBA Mock Draft Database')
    st.write('This is a database of NBA mock drafts for the 2022 NBA Draft')

    mocks_over_time(df)
    avg_pick_by_player(df)
    player_team_combo(df)


if __name__ == "__main__":
    #execute
    app()