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
        '2025': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db_2025.csv',
        '2024': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db_2024.csv',
        '2023': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db_2023.csv',
        '2022': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db_2022.csv',
        '2021': 'https://raw.githubusercontent.com/aaroncolesmith/nfl_mock_draft_db/main/new_nfl_mock_draft_db.csv'
    }

    # Load the appropriate data based on the selected year
    df = pd.read_csv(url_mapping.get(draft_year, ''))

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


    st.markdown("<h4 style='text-align: center; color: black;'>Network diagram showing relationships between teams and drafted players in recent mock drafts</h4>", unsafe_allow_html=True)


#TEAM PICK VERSION OF THE NETWORK GRAPH

    # d=df.groupby(['player','team_pick','team_img']).size().to_frame('cnt').reset_index()

    # player=d.groupby(['player']).agg({'cnt':'sum'}).reset_index()
    # player.columns=['player','times_picked']
    # team=d.groupby(['team_pick']).agg({'cnt':'sum'}).reset_index()
    # team.columns=['team_pick','team_times_picked']

    # d=pd.merge(d,player)
    # d=pd.merge(d,team)

    # d=d.sort_values('cnt',ascending=False)

    # d['pick_str'] = d['team_pick']+ ' - '+d['cnt'].astype('str')+' times'
    # d['player_pick_str'] = d['player']+ ' - '+d['cnt'].astype('str')+' times'

    # nt = Network(directed=False,
    #              # notebook=True,
    #              height="480px",
    #              width="620px",
    #              heading='')

    # nt.force_atlas_2based(damping=2)

    # # icon1 = st.checkbox('Show icons (slows it down a bit)'key='icon1')
    # icon1 = ''
    icon2 = ''

    # for i, r in d.iterrows():
    #     nt.add_node(r['player'],
    #                 size=r['times_picked'],
    #                 color={'background':'#40D0EF','border':'#03AED3'},
    #                 title = '<b>'+r['player'] + ' - Picked '+str(r['times_picked'])+'  times </b> <br> ' + d.loc[d.player==r['player']].groupby('player').apply(lambda x: ', <br>'.join(x.pick_str)).to_frame('pick_str').reset_index()['pick_str'].item())
    #     if icon1:
    #         nt.add_node(r['team_pick'],
    #                     size=r['team_times_picked'],
    #                     color={'background':'#FA70C8','border':'#EC0498'},
    #                     shape='image',
    #                     image =r['team_img'],
    #                     title='<b>' +r['team_pick'] + ' - ' +str(r['team_times_picked'])+'  total picks</b> <br> ' + d.loc[d.team_pick == r['team_pick']].groupby('team_pick').apply(lambda x: ', <br>'.join(x.player_pick_str)).to_frame('cnt').reset_index()['cnt'].item())
    #     else:
    #         nt.add_node(r['team_pick'],
    #                     size=r['team_times_picked'],
    #                     color={'background':'#FA70C8','border':'#EC0498'},
    #                     # shape='image',
    #                     # image =r['team_img'],
    #                     title='<b>' +r['team_pick'] + ' - ' +str(r['team_times_picked'])+'  total picks</b> <br> ' + d.loc[d.team_pick == r['team_pick']].groupby('team_pick').apply(lambda x: ', <br>'.join(x.player_pick_str)).to_frame('cnt').reset_index()['cnt'].item())

    #     nt.add_edge(r['player'],
    #                 r['team_pick'],
    #                 value = r['cnt'],
    #                 color='#9DA0DC',
    #                 title=r['team_pick']+' picked '+r['player']+' '+str(r['cnt'])+ '  times')
    
 # TEAM ONLY VERSION OF THE NETWORK GRAPH 
    d=df.groupby(['player','team','team_img']).size().to_frame('cnt').reset_index()

    player=d.groupby(['player']).agg({'cnt':'sum'}).reset_index()
    player.columns=['player','times_picked']
    team=d.groupby(['team']).agg({'cnt':'sum'}).reset_index()
    team.columns=['team','team_times_picked']

    d=pd.merge(d,player)
    d=pd.merge(d,team)

    d=d.sort_values('cnt',ascending=False)

    d['pick_str'] = d['team']+ ' - '+d['cnt'].astype('str')+' times'
    d['player_pick_str'] = d['player']+ ' - '+d['cnt'].astype('str')+' times'

    d['times_picked_log']=np.log2(d['times_picked'])
    d.loc[d['times_picked_log']<1,'times_picked_log'] = 1
    d['times_picked_log'] = d['times_picked_log']*10

    d['team_times_picked_log']=np.log2(d['team_times_picked'])
    d.loc[d['team_times_picked_log']<1,'team_times_picked_log'] = 1
    d['team_times_picked_log'] = d['team_times_picked_log']*10

    nt = Network(directed=False,
                # notebook=True,
                height="480px",
                width="620px",
                # width="940px",
                heading='')

    # nt.force_atlas_2based(damping=1, 
    #     spring_length=100,
    #     # overlap=1
    #     )
    
    nt.barnes_hut(
        spring_length=25,
    )

    # icon2 = st.checkbox('Show icons (slows it down a bit)',key='icon2')

    # st.write(d.head(2))

    for i, r in d.iterrows():
        nt.add_node(r['player'],
                    size=r['times_picked_log'],
                    color={'background':'#40D0EF','border':'#03AED3'},
                    title = '<b>'+r['player'] + ' - Picked '+str(r['times_picked'])+'  times </b> <br> ' + d.loc[d.player==r['player']].groupby('player').apply(lambda x: ', <br>'.join(x.pick_str)).to_frame('pick_str').reset_index()['pick_str'].item())
        if icon2:
            nt.add_node(r['team'],
                        size=r['team_times_picked_log'],
                        color={'background':'#FA70C8','border':'#EC0498'},
                        shape='image',
                        image =r['team_img'],
                        title='<b>' +r['team'] + ' - ' +str(r['team_times_picked'])+'  total picks</b> <br> ' + d.loc[d.team == r['team']].groupby('team').apply(lambda x: ', <br>'.join(x.player_pick_str)).to_frame('cnt').reset_index()['cnt'].item())
        else:
            nt.add_node(r['team'],
                        size=r['team_times_picked_log'],
                        color={'background':'#FA70C8','border':'#EC0498'},
                        # shape='image',
                        # image =r['team_img'],
                        title='<b>' +r['team'] + ' - ' +str(r['team_times_picked'])+'  total picks</b> <br> ' + d.loc[d.team == r['team']].groupby('team').apply(lambda x: ', <br>'.join(x.player_pick_str)).to_frame('cnt').reset_index()['cnt'].item())

        nt.add_edge(r['player'],
                    r['team'],
                    value = r['cnt'],
                    color='#9DA0DC',
                    title=r['team']+' picked '+r['player']+' '+str(r['cnt'])+ '  times')


# SHOW THE NETWORK GRAPH  
    try:                     
        nt.show('mock_draft_network.html')

        html_file = open('./mock_draft_network.html', 'r', encoding='utf-8')
        source_code = html_file.read()
        components.html(source_code, height=510,width=640)
    except:
        pass


# SANKEY DIAGRAM OF TOP 10 PICKS

    df['player_pick'] = df['team_pick'] + ' - ' + df['player']
    d=df.loc[df.pick.isin([1,2,3,4,5,6,7,8,9,10])].groupby(['source','date']).agg({'player_pick': lambda x: ','.join(x),
                                                                'team_img':'size'}).reset_index()
    d.columns = ['source','date','player','recs']
    d = d.loc[d.recs <= 10]
    d['player']=d['player'].str.replace('Aidan ','').str.replace('Jacksonville ','').str.replace('Kayvon ','').str.replace('Ahmand Gardner','Sauce').str.replace('New York ','').str.replace('Pick ','')
    picks=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']
    d[picks]=d['player'].str.split(',',n=9,expand=True)

    fig = genSankey(d.groupby(picks).size().to_frame('cnt').reset_index(),
                    cat_cols=picks,
                    value_cols='cnt',
                    title='Sankey Diagram of Top 10 Picks')

    del df['player_pick']
    del d

    fig = go.Figure(fig)
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)



    fig=px.bar(df.groupby(['team','player']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False).head(15),
           y=df.groupby(['team','player']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False).head(15).team + ' - '+df.groupby(['team','player']).size().to_frame('cnt').reset_index().sort_values('cnt',ascending=False).head(15).player,
           x='cnt',
           # color='#FA70C8',
           orientation='h',
           template='plotly_white',
           title='Most Common Team - Player Pairings')
    
    fig.update_yaxes(title='Team / Player', categoryorder='total descending')
    fig.update_xaxes(title='Count')
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df.loc[df.player.isin(df.groupby('player').agg({'pick':'size'}).reset_index().sort_values('pick',ascending=False).head(40)['player'])], 
            x="player", 
            y="pick", 
            points="all", 
            hover_data=['team','date','source'], 
            title='Distribution of Draft Position by Player',
            template='plotly_white', 
            width=1600)
    fig.update_xaxes(title='Player', categoryorder='mean ascending')
    fig.update_yaxes(title='Draft Position')
    st.plotly_chart(fig, use_container_width=True)

    d=df.groupby(['team','team_img','player']).agg({'pick':['min','mean','median','size']}).reset_index()
    d.columns=['team','team_img','player','min_pick','avg_pick','median_pick','cnt']
    fig=px.scatter(d,
          x='cnt',
          y='avg_pick',
           color='team',
           template='plotly_white',
           title='# of Times a Player is Mocked to a Given Pick / Team',
          hover_data=['player'])
    fig.update_xaxes(title='# of Occurences')
    fig.update_yaxes(title='Avg. Draft Pick')
    fig.update_traces(mode='markers',
                      marker=dict(size=8,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    fig = update_colors(fig)
    st.plotly_chart(fig, use_container_width=True)

    d=d.sort_values('avg_pick',ascending=True)
    fig=px.scatter(d,
          x='player',
          y='avg_pick',
           size='cnt',
          color='team',
          template='plotly_white',
          height=600,
          title='Avg. Pick Placement by Player / Team')
    fig.update_xaxes(title='Player')
    fig.update_xaxes(categoryorder='mean ascending')
    fig.update_yaxes(title='Avg. Draft Position')
    fig = update_colors(fig)
    st.plotly_chart(fig, use_container_width=True)

    df['source_date'] = df['source'] + ' - ' +df['date']
    draft = st.selectbox('Pick a draft to view:',df['source_date'].unique())

    col1, col2, col3 = st.columns((2,4,2))
    df_table=df.loc[df['source_date'] == draft].sort_values('pick',ascending=True).reset_index(drop=True)
    df_table['team'] = ["<img src='" + r.team_img
    + f"""' style='display:block;margin-left:auto;margin-right:auto;width:32px;border:0;'><div style='text-align:center'>"""
    # + "<br>".join(r.team.split()) + "</div>"
    for ir, r in df_table.iterrows()]
    df_table['pick'] = df_table['pick'].astype('str').replace('\.0', '', regex=True)

    col2.write(df_table[['pick','team','player']].to_html(index=False,escape=False), unsafe_allow_html=True)

    player = st.selectbox('Pick a player to view:',df['player'].unique())

    d=df.loc[df.player == player].copy()
    d=d.sort_values('date',ascending=True).reset_index(drop=True)
    d['mock_draft'] = d['date'].str[5:]+ ' - ' + d['source']
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
    fig = update_colors(fig)
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

    team = st.selectbox('Pick a team to view:',df['team'].unique())

    d=df.loc[df.team == team].copy()
    d=d.reset_index(drop=True)

    d=d.sort_values('date',ascending=True).reset_index(drop=True)
    d['mock_draft'] = d['date'].str[5:]+ ' - ' + d['source']
    fig=px.scatter(d,
                    x='mock_draft',
                    y='player',
                    color='player',
                    template='plotly_white',
                    category_orders={'mock_draft': d["mock_draft"]},
                    title='Mock Drafts over Time for ' + team)
    fig.update_xaxes(title='Mock Draft / Date')
    fig.update_yaxes(categoryorder='array', categoryarray= d.groupby('player').size().to_frame('cnt').sort_values('cnt',ascending=True).reset_index()['player'].to_numpy())
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


if __name__ == "__main__":
    #execute
    app()