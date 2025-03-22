import pandas as pd
import streamlit as st
import requests
import datetime
from pyvis.network import Network
from pyvis import network as net
import streamlit.components.v1 as components
import networkx as nx
import plotly_express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
import numpy as np
from IPython.core.display import HTML

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog',
    layout='wide'
    )

def update_colors(fig):
    fig.for_each_trace(lambda trace: trace.update(marker_color='#FB4F14') if trace.name == "Cincinnati Bengals" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#006778') if trace.name == "Jacksonville Jaguars" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#008E97') if trace.name == "Miami Dolphins" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#A71930') if trace.name == "Atlanta Falcons" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#125740') if trace.name == "New York Jets" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#97233F') if trace.name == "Arizona Cardinals" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#0076B6') if trace.name == "Detroit Lions" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#AA0000') if trace.name == "San Francisco 49Ers" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#241773') if trace.name == "Baltimore Ravens" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#C60C30') if trace.name == "Buffalo Bills" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#0085CA') if trace.name == "Carolina Panthers" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#C83803') if trace.name == "Chicago Bears" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#041E42') if trace.name == "Dallas Cowboys" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#FB4F14') if trace.name == "Denver Broncos" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#203731') if trace.name == "Green Bay Packers" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#03202F') if trace.name == "Houston Texans" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#FF3C00') if trace.name == "Cleveland Browns" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#002C5F') if trace.name == "Indianapolis Colts" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#E31837') if trace.name == "Kansas City Chiefs" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#0080C6') if trace.name == "Los Angeles Chargers" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#003594') if trace.name == "Los Angeles Rams" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#4F2683') if trace.name == "Minnesota Vikings" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#002244') if trace.name == "New England Patriots" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#D3BC8D') if trace.name == "New Orleans Saints" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#0B2265') if trace.name == "New York Giants" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#A5ACAF') if trace.name == "Las Vegas Raiders" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#004C54') if trace.name == "Philadelphia Eagles" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#FFB612') if trace.name == "Pittsburgh Steelers" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#69BE28') if trace.name == "Seattle Seahawks" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#D50A0A') if trace.name == "Tampa Bay Buccaneers" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#4B92DB') if trace.name == "Tennessee Titans" else ())
    fig.for_each_trace(lambda trace: trace.update(marker_color='#773141') if trace.name == "Washington Football Team" else ())

    return fig


def generate_gradient(start_color, end_color, num_colors):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb_color):
        return '#{0:02X}{1:02X}{2:02X}'.format(*rgb_color)

    def calculate_opposite_color(color1_rgb, color2_rgb):
        return rgb_to_hex((
            255 - int((color1_rgb[0] + color2_rgb[0]) / 2),
            255 - int((color1_rgb[1] + color2_rgb[1]) / 2),
            255 - int((color1_rgb[2] + color2_rgb[2]) / 2)
        ))

    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)

    if num_colors <= 10:
        color_gradient = [
            rgb_to_hex((
                int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (num_colors - 1)),
                int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (num_colors - 1)),
                int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (num_colors - 1))
            ))
            for i in range(num_colors)
        ]
    else:
        additional_color = calculate_opposite_color(start_rgb, end_rgb)
        additional_rgb = hex_to_rgb(additional_color)

        color_gradient = [
            rgb_to_hex((
                int(start_rgb[0] + (additional_rgb[0] - start_rgb[0]) * i / (num_colors - 1)),
                int(start_rgb[1] + (additional_rgb[1] - start_rgb[1]) * i / (num_colors - 1)),
                int(start_rgb[2] + (additional_rgb[2] - start_rgb[2]) * i / (num_colors - 1))
            ))
            for i in range(num_colors // 2)
        ]

        try:
            color_gradient._append(additional_color)
        except:
            color_gradient.append(additional_color)

        color_gradient.extend([
            rgb_to_hex((
                int(additional_rgb[0] + (end_rgb[0] - additional_rgb[0]) * i / (num_colors // 2 - 1)),
                int(additional_rgb[1] + (end_rgb[1] - additional_rgb[1]) * i / (num_colors // 2 - 1)),
                int(additional_rgb[2] + (end_rgb[2] - additional_rgb[2]) * i / (num_colors // 2 - 1))
            ))
            for i in range(num_colors // 2)
        ])

    return color_gradient



def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#0085CA','#03202F','#97233F','#002C5F','#69BE28','#0076B6','#A5ACAF','#A71930','#0B162A','#125740']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp =  list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        
    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    for i in range(len(cat_cols)-1):
        if i==0:
            sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            sourceTargetDf.columns = ['source','target','count']
        else:
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            tempDf.columns = ['source','target','count']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
        
    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node = dict(
          pad = 10,
          thickness = 10,
          line = dict(
            color = "black",
            width = 0.2
          ),
          label = labelList,
          color = colorList
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['count']
        )
      )
    
    layout =  dict(
        title = title,
        font = dict(
          size = 8
        )
    )
       
    fig = dict(data=[data], layout=layout)
    return fig
  


def add_sheets_data(df):
    df_img=df.groupby(['team'])['team_img'].agg(pd.Series.mode).to_frame('team_img').reset_index()
    team_img_dict = dict(zip(df_img['team'], df_img['team_img']))

    df_player_details=df.groupby(['player'])['player_details'].agg(pd.Series.mode).to_frame('player_details').reset_index()
    player_details_dict = dict(zip(df_player_details['player'], df_player_details['player_details']))

    url='https://docs.google.com/spreadsheets/d/1Fq_DdIsiMKe3tTkUj4eb9X5dlzAPnrRzkHvpNpJKnpk/gviz/tq?tqx=out:csv&gid=685837892'
    df_sheet=pd.read_csv(url, on_bad_lines='skip')
    for col in df_sheet.columns:
        if 'Unnamed' in col:
            del df_sheet[col]
    df_sheet['team_img']=df_sheet['team'].map(team_img_dict)
    df_sheet['player_details']=df_sheet['player'].map(player_details_dict)
    df_sheet['date'] = pd.to_datetime(df_sheet['date'])

    df=pd.concat([df,df_sheet]).sort_values(['date','source'],ascending=[True,True]).reset_index(drop=True)

    df['pick']=pd.to_numeric(df['pick'])

    df['team_pick'] = 'Pick '+ df['pick'].astype('str').replace('\.0', '', regex=True) + ' - ' +df['team']
    df=df.loc[~df.team_pick.str.contains('/Colleges')]
    df=df.loc[df.source_key!='tankathon-2024?date=2024-06-12']

    df['date'] = pd.to_datetime(df['date'])

    return df



def app():
    # st.markdown("<h1 style='text-align: center; color: black;'>NFL Mock Draft Database</h1>", unsafe_allow_html=True)
    # st.markdown("<h4 style='text-align: center; color: black;'>Taking a look at a number of public NFL mock drafts to identify trends and relationships</h4>", unsafe_allow_html=True)

    st.title('NFL Mock Draft Database')
    
    req = requests.get('https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/last_updated.txt')
    # last_update = (datetime.datetime.utcnow() - pd.to_datetime(req.text)).total_seconds()
    
    # if last_update <60:
    #     st.write('Last update: '+str(round(last_update,2))+' seconds ago')
    # elif last_update <3600:
    #     st.write('Last update: '+str(round(last_update/60,2))+' minutes ago')
    # else:
    #     st.write('Last update: '+str(round(last_update/3600,2))+' hours ago')

    st.markdown('Taking a look at a number of public NFL mock drafts to identify trends and relationships')

    draft_year = st.selectbox('Draft Year?', 
        ('2025', '2024', '2023', '2022', '2022 - Most Recent', '2021'))

    # Define a dictionary for draft year to URL mapping
    url_mapping = {
        # '2025': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db_2025.csv',
        '2025': 'https://raw.githubusercontent.com/aaroncolesmith/data_nfl_mock_draft/main/data/new_nfl_mock_draft_db_2025.csv',
        '2024': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db_2024.csv',
        '2023': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db_2023.csv',
        '2022': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db_2022.csv',
        '2021': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db.csv'
    }

    # Load the appropriate data based on the selected year
    df = pd.read_csv(url_mapping.get(draft_year, ''))
    df=add_sheets_data(df)
    # Apply specific modifications based on the selected year
    if draft_year == '2023':
        # Drop a bad mock draft for 2023
        df = df.drop(df.query("source == 'TWSN' & date == '2023-04-21'").index).reset_index(drop=True)

    if draft_year == '2022 - Most Recent':
        # Load the 2022 draft and limit rows to the most recent
        df = df.head(1023)

    d=pd.merge(df.iloc[0:500].groupby('player').agg({'pick':'mean','player_details':'size'}).reset_index(),
             df.iloc[501:1000].groupby('player').agg({'pick':'mean','player_details':'size'}).reset_index(),
             left_on='player',
             right_on='player',
             suffixes=('_recent','_before')
    )
    d=d.loc[d.player_details_recent>=5]
    d['chg']=d['pick_recent'] - d['pick_before']
    d['pct_chg'] = (d['pick_recent'] - d['pick_before'])/d['pick_before']
    d.sort_values('chg',ascending=True)

    col1, col2 = st.columns(2)
    col1.success("### Players Rising :fire:")
    for i, r in d.sort_values('chg',ascending=True).head(5).iterrows():
        col1.write(r['player']+' - trending ' + str(round(abs(r['chg']),2)) + ' picks earlier')


    col2.warning("### Players Falling 🧊")
    for i, r in d.sort_values('chg',ascending=False).head(5).iterrows():
        col2.write(r['player'] + ' - trending ' + str(round(r['chg'],2)) +' picks later')

    del d



    st.divider()

    df['source_key'] = df['source'].str.lower().replace(' ','_') + '_'+df['date'].astype('str')
    t1,t2,t3,t4,t5,t6 = st.tabs(['Avg Player Rank','Times Player Mocked to Team','Filter by Team','Filter by Player','Filter by Mock Draft','Consensus Mock Draft'])

    with t1:
        st.markdown('### Avg Pick Over Time by Player')

        c1,c2 = st.columns(2)
        min_date = c1.date_input("Minimum Date for Avg", 
                                 value = df.date.min(),
                                 min_value = df.date.min(),
                                 max_value = df.date.max())
        
        top_n = c2.slider("Top n players?", 2, 50, 25)

        adjustment_num = 35

        df_avg_pick_all = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].groupby(['player']).agg(
            avg_pick=('pick','mean'),
            pick_total=('pick','sum'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values(['avg_pick'],ascending=[True]).reset_index()
        df_avg_pick_all['mocks_not_picked'] = df_avg_pick_all['times_picked'].max() - df_avg_pick_all['times_picked']
        df_avg_pick_all['adjusted_avg_pick'] = (df_avg_pick_all['pick_total'] + (adjustment_num * df_avg_pick_all['mocks_not_picked'])) / df_avg_pick_all['times_picked'].max()

        # Group by player and date to get daily stats
        df_avg_pick_daily = df.groupby(['player', 'date']).agg(
            avg_pick=('pick', 'mean'),
            pick_total=('pick', 'sum'),
            times_picked=('source_key', lambda x: x.nunique())
        ).sort_values(['avg_pick'], ascending=[True]).reset_index()


        df_avg_pick_daily = pd.merge(df_avg_pick_daily,
                                     df_avg_pick_daily.groupby(['date']).agg(total_mocks_for_day=('times_picked','max')).reset_index()
                                     )

        # Calculate mocks where the player wasn't picked
        df_avg_pick_daily['mocks_not_picked'] = df_avg_pick_daily['total_mocks_for_day'] - df_avg_pick_daily['times_picked']

        # Calculate adjusted average pick based on mocks not picked
        df_avg_pick_daily['adjusted_avg_pick'] = (df_avg_pick_daily['pick_total'] + (adjustment_num * df_avg_pick_daily['mocks_not_picked'])) / df_avg_pick_daily['total_mocks_for_day']

        # Optional: Sort the dataframe by player and date for clarity
        df_avg_pick_daily = df_avg_pick_daily.sort_values(['player', 'date'])

        top_players=df_avg_pick_all.sort_values(['adjusted_avg_pick'],ascending=[True]).reset_index().head(top_n)['player'].tolist()
        
        df_d1 = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)]
        df_d1 = df_d1.loc[df_d1.player.isin(top_players)]


        d1=df_d1.groupby(['player','date']).agg(
            avg_pick=('pick','mean'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values(['date','player'],ascending=True).reset_index()


        d2=df_d1.groupby(['player','team','date']).agg(
            avg_pick=('pick','mean'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values(['date','player'],ascending=True).reset_index()

        d2['team_picks'] = d2['team'] + ' - ' + d2['times_picked'].astype('str').replace('\.0', '', regex=True)
        d2=d2.groupby(['player','date']).agg(
            
                                        team_picks=('team_picks', lambda x: ', '.join(x))
        ).reset_index()

        d1 = pd.merge(d1,
                    d2,
                    )

        d1 = pd.merge(d1,
                      df_avg_pick_daily[['player','date','adjusted_avg_pick','mocks_not_picked']],
                      how='left')


        start_color='#FF6600'
        end_color='#55E0FF'
        color_list = generate_gradient(start_color, end_color, top_n)
        c1,c2=st.columns(2)
        fig = px.scatter(
            d1,
            x="date",
            y="adjusted_avg_pick",
            hover_data=['team_picks','avg_pick','times_picked','mocks_not_picked'],
            color="player",
            # color_discrete_sequence=color_list,
            render_mode='svg',)
        fig.update_traces(
            mode="lines+markers",
            line_shape="spline",
            line=dict(width=4),
            marker=dict(
                size=8,
                opacity=0.6,
                line=dict(width=1, color="DarkSlateGrey"),
            ),
        )
        fig.update_layout(
            font_family="Futura",
            font_color="black",
            # title="Avg Pick Over Time",
            template='simple_white'
        )

        c1.plotly_chart(fig,use_container_width=True)
        
        fig = px.box(df_d1, 
                x="player", 
                y="pick", 
                points="all", 
                hover_data=['team','date','source'], 
                title='Distribution of Draft Position by Player',
                template='plotly_white', 
                width=1600)
        fig.update_xaxes(title='Player', categoryorder='mean ascending')
        fig.update_yaxes(title='Draft Position')
        c2.plotly_chart(fig, use_container_width=True)

        d3=df_d1.groupby(['player','team']).agg(
            avg_pick=('pick','mean'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values('times_picked',ascending=False).reset_index(drop=False)
        

        d3['team_picks'] = d3['team'] + ' - ' + d3['times_picked'].astype('str').replace('\.0', '', regex=True)
        d3=d3.groupby(['player']).agg(
            team_picks=('team_picks', lambda x: ', '.join(x))
        ).reset_index()

        df_grouped = df_avg_pick_all
        df_grouped = df_grouped.loc[df_grouped['times_picked']>=(df_grouped['times_picked'].max()*.5)].reset_index(drop=True).reset_index(drop=False)
        df_grouped = pd.merge(df_grouped,d3)
        df_grouped.columns = ['Rank','Player','del1','del2','Times Picked','del3','Average Pick','Team Picks']
        df_grouped['Rank'] +=1
        df_grouped['Average Pick'] = df_grouped['Average Pick'].round(1)

        del df_grouped['del1']
        del df_grouped['del2']
        del df_grouped['del3']

        c1,c2,c3 = st.columns([2,6,2])
        c2.dataframe(
            df_grouped,
            hide_index=True,
            use_container_width =True
            )

    with t2:
        st.markdown('### # of Times a Player is Mocked to a Given Pick / Team')
        d=df.groupby(['team','team_img','player']).agg({'pick':['min','mean','median','size']}).reset_index()
        d.columns=['team','team_img','player','min_pick','avg_pick','median_pick','cnt']
        fig=px.scatter(d,
            x='cnt',
            y='avg_pick',
            color='team',
            template='plotly_white',
            #    title='# of Times a Player is Mocked to a Given Pick / Team',
            hover_data=['player'])
        fig.update_xaxes(title='# of Occurences')
        fig.update_yaxes(title='Avg. Draft Pick')
        fig.update_traces(mode='markers',
                        marker=dict(size=8,
                                    line=dict(width=1,
                                                color='DarkSlateGrey')))
        fig = update_colors(fig)
        st.plotly_chart(fig, use_container_width=True)





    ## filter by mock draft section
    with t5:
        df['source_date'] = df['source'] + ' - ' +df['date'].astype('str')
        draft = st.selectbox('Pick a draft to view:',df.sort_values(['date','source'],ascending=[False,True])['source_date'].unique())

        col1, col2, col3 = st.columns((2,4,2))
        df_table=df.loc[df['source_date'] == draft].sort_values('pick',ascending=True).reset_index(drop=True)
        df_table['team'] = ["<img src='" + r.team_img
        + f"""' style='display:block;margin-left:auto;margin-right:auto;width:32px;border:0;'><div style='text-align:center'>"""
        # + "<br>".join(r.team.split()) + "</div>"
        for ir, r in df_table.iterrows()]
        df_table['pick'] = df_table['pick'].astype('str').replace('\.0', '', regex=True)

        col2.write(df_table[['pick','team','player']].to_html(index=False,escape=False), unsafe_allow_html=True)

    with t4:
        player = st.selectbox('Pick a player to view:',df['player'].unique())

        d=df.loc[df.player == player].copy()
        d=d.sort_values('date',ascending=True).reset_index(drop=True)
        d['mock_draft'] = d['date'].astype('str').str[5:]+ ' - ' + d['source']
        fig=px.scatter(d,
                        x='mock_draft',
                        y='pick',
                        color='team',
                        template='plotly_white',
                        category_orders={'mock_draft': d["mock_draft"]},
                        title='Mock Drafts over Time for ' + player)
        fig.update_xaxes(title='Mock Draft / Date')
        fig.update_traces(mode='markers',
                            marker=dict(size=8,
                                        line=dict(width=1,
                                                color='DarkSlateGrey')))
        fig.update_layout(height=600)
        fig = update_colors(fig)
        # fig=replace_scatter_with_logos(fig)
        st.plotly_chart(fig, use_container_width=True)


        d2=d.groupby(['team']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False)
        fig=px.bar(d2,
            orientation='h',
            y='team',
            color='team',
            x='cnt',
            template='plotly_white',
            title='What team has picked ' + player + ' the most?')

        fig.update_yaxes(title='Team', categoryorder='total descending')
        fig.update_xaxes(title='# of Times Mocked')
        fig.update_yaxes(autorange="reversed")
        fig = update_colors(fig)
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        team = st.selectbox('Pick a team to view:',df['team'].unique())

        d=df.loc[df.team == team].copy()
        d=d.reset_index(drop=True)

        d=d.sort_values('date',ascending=True).reset_index(drop=True)
        d['mock_draft'] = d['date'].astype('str').str[5:]+ ' - ' + d['source']
        fig=px.scatter(d,
                        x='mock_draft',
                        y='player',
                        color='player',
                        template='plotly_white',
                        category_orders={'mock_draft': d["mock_draft"]},
                        title='Mock Drafts over Time for ' + team)
        fig.update_xaxes(title='Mock Draft / Date')
        fig.update_yaxes(categoryorder='array', categoryarray= d.groupby('player').size().to_frame('cnt').sort_values('cnt',ascending=True).reset_index()['player'].to_numpy())
        fig.update_layout(height=600)
        fig.update_traces(mode='markers',
                            marker=dict(size=8,
                                        line=dict(width=1,
                                                color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)

        f = lambda x: x["player_details"].partition('|')[0]
        d['position']=d.apply(f, axis=1)
        

        # d2=d.groupby(['team_pick','player','position']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False)
        # fig=px.bar(d2,
        #        orientation='h',
        #        y=d2['team_pick'] + ' - ' + d2['player'],
        #        color='position',
        #        x='cnt',
        #        title='How many times a Pick / Team has been mocked to a player')
        # fig.update_yaxes(title='Pick / Team', categoryorder='total descending')
        # fig.update_xaxes(title='# of Times Mocked')
        # fig.update_yaxes(autorange="reversed")

        d2=d.groupby(['team','player','position']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False)
        fig=px.bar(d2,
            orientation='h',
            y=d2['team'] + ' - ' + d2['player'],
            color='position',
            x='cnt',
            template='plotly_white',
            title='How many times a Team has been mocked to a player')
        fig.update_yaxes(title='Team', categoryorder='total descending')
        fig.update_xaxes(title='# of Times Mocked')
        fig.update_yaxes(autorange="reversed")


        st.plotly_chart(fig, use_container_width=True)


        # d2=d.groupby(['team_pick','position']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False)
        # fig=px.bar(d2,
        #        orientation='h',
        #        y=d2['team_pick'] + ' - ' + d2['position'],
        #        x='cnt',
        #        title='How many times a Pick / Team has been mocked to a Position')

        # fig.update_yaxes(title='Pick / Team', categoryorder='total descending')
        # fig.update_xaxes(title='# of Times Mocked')
        # fig.update_yaxes(autorange="reversed")


        d2=d.groupby(['team','position']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False)
        fig=px.bar(d2,
            orientation='h',
            y=d2['team'] + ' - ' + d2['position'],
            x='cnt',
            template='plotly_white',
            title='How many times a Team has been mocked to a Position')

        fig.update_yaxes(title='Pick', categoryorder='total descending')
        fig.update_xaxes(title='# of Times Mocked')
        fig.update_yaxes(autorange="reversed")


        st.plotly_chart(fig, use_container_width=True)

    with t6:

        df_draft_order = df.groupby(['pick'])['team'].agg(pd.Series.mode).to_frame('team').reset_index()

        df['date'] = pd.to_datetime(df['date'])
        
        c1,c2,c3 = st.columns(3)
        min_date = c1.date_input("Minimum Date for Consensus", 
                                 value = df.date.min(),
                                 min_value = df.date.min(),
                                 max_value = df.date.max())
        
        adjustment_num = 35

        df_avg_pick_all = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].groupby(['player']).agg(
            avg_pick=('pick','mean'),
            pick_total=('pick','sum'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values(['avg_pick'],ascending=[True]).reset_index()
        df_avg_pick_all['mocks_not_picked'] = df_avg_pick_all['times_picked'].max() - df_avg_pick_all['times_picked']
        df_avg_pick_all['adjusted_avg_pick'] = (df_avg_pick_all['pick_total'] + (adjustment_num * df_avg_pick_all['mocks_not_picked'])) / df_avg_pick_all['times_picked'].max()



        df_bpa = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].groupby(['player']).agg(
            avg_pick=('pick','mean'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values(['avg_pick'],ascending=True).reset_index(drop=False).reset_index(drop=False)
        
        ## moving from df_bpa to df_avg_pick_all
        # df_dpa = df_avg_pick_all[['player','adjusted_avg_pick','times_picked']].reset_index(drop=False)
        df_bpa.columns=['rank','player','avg_pick','times_picked']
        df_bpa['rank']+=1
        df_bpa['times_picked_pct']=df_bpa['times_picked']/df_bpa['times_picked'].max()
        df_bpa=df_bpa.loc[df_bpa['times_picked_pct']>.3]
        

        # st.write(df_bpa)

        # ###version 1 -- based on player & pick
        # df_consensus = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].groupby(['pick','team','player']).agg(times_picked=('source_key','size')).sort_values(['pick','times_picked'],
        #                                                                                                        ascending=[True,False]).reset_index()
        # total_mocks = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].source_key.nunique()
        # # st.write(total_mocks)
        # df_consensus['pick_pct'] = df_consensus['times_picked'] / total_mocks
        # df_consensus = pd.merge(df_draft_order,df_consensus)
        # # st.write(df_consensus)
        # players_picked =[]
        # for i,r in df_draft_order.iterrows():
        #     # st.write(r['pick'])
        #     # st.write(r['team'])

        #     # st.write(df_consensus.loc[
        #     #     (df_consensus['pick'] == r['pick'])&
        #     #     (df_consensus['team'] == r['team'])&
        #     #     (~df_consensus['player'].isin(players_picked))
        #     # ])
        #     picked_player = df_consensus.loc[
        #         (df_consensus['pick'] == r['pick'])&
        #         (df_consensus['team'] == r['team'])&
        #         (~df_consensus['player'].isin(players_picked))
        #     ]['player'].values[0]

        #     pick_pct = df_consensus.loc[
        #         (df_consensus['pick'] == r['pick'])&
        #         (df_consensus['team'] == r['team'])&
        #         (~df_consensus['player'].isin(players_picked))
        #     ]['pick_pct'].values[0]

        #     ideal_pick = df_consensus.loc[
        #         (df_consensus['pick'] == r['pick'])&
        #         (df_consensus['team'] == r['team'])
        #     ]['player'].values[0]

        #     ideal_pick_pct = df_consensus.loc[
        #         (df_consensus['pick'] == r['pick'])&
        #         (df_consensus['team'] == r['team'])
        #     ]['pick_pct'].values[0]

        #     if picked_player == ideal_pick:
        #         was_ideal = ''
        #     else:
        #         was_ideal = f'(Ideal pick was {ideal_pick} -- picked {round(ideal_pick_pct*100,2)}% of the time)'

        #     st.write(f"Pick {r['pick']} - {r['team']} -- {picked_player} -- picked {round(pick_pct*100,2)}% of the time {was_ideal}")
        #     with st.expander('BPA'):
        #         st.dataframe(df_bpa.loc[~df_bpa['player'].isin(players_picked)].head(20),
        #                      hide_index=True)
        #     # st.write(picked_player)
        #     players_picked.append(picked_player)


        ###version 2 -- based on player, less than pick
        df_consensus = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].groupby(['team','team_img','player']).agg(avg_pick=('pick','mean'),
                                                                                                                  times_picked=('source_key','size')).sort_values(['times_picked','avg_pick'],
                                                                                                               ascending=[False,True]).reset_index()
        total_mocks = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].source_key.nunique()
        # st.write(total_mocks)
        df_consensus['pick_pct'] = df_consensus['times_picked'] / total_mocks

        df_player_avg = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].groupby(['player']).agg(avg_player_rank=('pick','mean')).reset_index()
        df_consensus = pd.merge(df_consensus,df_player_avg).sort_values(['times_picked','avg_player_rank','avg_pick'],ascending=[False,True,True])

        ## updating df_consense to have our new version from df_avg_pick_all
        # del df_consensus['avg_player_rank']

        # df_merge = df_avg_pick_all[['player','adjusted_avg_pick']]
        # df_merge.columns =['player','avg_player_rank']
        # df_consensus = pd.merge(df_consensus, df_avg_pick_all)



        # st.write(df_bpa)
        # st.write(df_consensus)
        # st.write(df_avg_pick_all)





        # df_consensus = pd.merge(df_draft_order,df_consensus)
        players_picked =[]
        df_draft_order = df_draft_order.loc[df_draft_order.pick<=30]
        for i,r in df_draft_order.iterrows():
            st.divider()




            big_board = df_consensus.loc[
                # (df_consensus['pick'] == r['pick'])&
                (df_consensus['team'] == r['team'])&
                (~df_consensus['player'].isin(players_picked))
            ]

            # st.write(big_board)
            try:
                picked_player = big_board['player'].values[0]
            except:
                st.write(big_board)

            pick_pct = big_board['pick_pct'].values[0]

            ideal_pick = df_consensus.loc[
                # (df_consensus['pick'] == r['pick'])&
                (df_consensus['team'] == r['team'])
            ]['player'].values[0]

            ideal_pick_pct = df_consensus.loc[
                # (df_consensus['pick'] == r['pick'])&
                (df_consensus['team'] == r['team'])
            ]['pick_pct'].values[0]

            if picked_player == ideal_pick:
                was_ideal = ''
            else:
                was_ideal = f'(Ideal pick was {ideal_pick} -- picked {round(ideal_pick_pct*100,2)}% of the time)'

            team_img = df_consensus.loc[
                (df_consensus['team'] == r['team'])
            ]['team_img'].values[0]

            c0,c1,c2=st.columns([1,3,8])

            c1.markdown(f"Pick {r['pick']}: {r['team']} -- {picked_player} ")
            c1.markdown(f"Picked {round(pick_pct*100,2)}% of the time {was_ideal}")
            c0.image(team_img,width=80)

            with c2:
                with st.expander('BPA & Big Board'):
                    c3,c4=st.columns(2)
                    c3.write('BPA')
                    c3.dataframe(df_bpa.loc[~df_bpa['player'].isin(players_picked)].head(20),
                            hide_index=True)
                    c4.write('Big Board')
                    c4.dataframe(big_board[['team','player','avg_pick','times_picked','pick_pct','avg_player_rank']],hide_index=True)
                    



            # with st.expander('BPA'):
            #     c2.dataframe(df_bpa.loc[~df_bpa['player'].isin(players_picked)].head(20),
            #                  hide_index=True)
            # st.write(picked_player)
            players_picked.append(picked_player)



if __name__ == "__main__":
    #execute
    app()