import streamlit as st
from st_files_connection import FilesConnection
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog',
    layout='wide'
    )



@st.cache_data(ttl=300)
def get_s3_data(filename):
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read(f"bet-model-data/{filename}.parquet", 
                input_format="parquet", 
                ttl=600
                )
    return df

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


def rising_falling(df):
    mid_point = df.index.size//2

    d=pd.merge(df.iloc[0:mid_point].groupby('player').agg({'pick':'mean','player_details':'size'}).reset_index(),
                df.iloc[mid_point:].groupby('player').agg({'pick':'mean','player_details':'size'}).reset_index(),
                left_on='player',
                right_on='player',
                suffixes=('_before','_recent')
    )

    d=d.loc[d.player_details_recent+d.player_details_before>=25]
    d['chg']=d['pick_recent'] - d['pick_before']
    d['pct_chg'] = (d['pick_recent'] - d['pick_before'])/d['pick_before']

    col1, col2 = st.columns(2)
    col1.success("### Players Rising :fire:")
    for i, r in d.sort_values('chg',ascending=True).head(5).iterrows():
        col1.write(r['player']+' - trending ' + str(round(abs(r['chg']),2)) + ' picks earlier')


    col2.warning("### Players Falling 🧊")
    for i, r in d.sort_values('chg',ascending=False).head(5).iterrows():
        col2.write(r['player'] + ' - trending ' + str(round(r['chg'],2)) +' picks later')

    del d

def update_colors(fig):
    teams_df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/teams_db.parquet?raw=true', engine='pyarrow')
    teams_df=teams_df.groupby(['team_full_name','team_logo','team_primary_color','team_secondary_color']).size().to_frame('cnt').reset_index()
    teams_df['team_primary_color'] = '#'+teams_df['team_primary_color']

    team_colors_dict = dict(zip(teams_df['team_full_name'], teams_df['team_primary_color']))

    for trace in fig.data:
        if trace.name in team_colors_dict:
            trace.marker.color = team_colors_dict[trace.name]
    return fig


def replace_scatter_with_logos(fig):
    """
    Replace scatter points with logos in the given Plotly figure based on the team data dataframe.
    
    Parameters:
    fig (plotly.graph_objects.Figure): The Plotly figure to update.
    team_data_df (pandas.DataFrame): DataFrame containing 'TEAM', 'COLOR', and 'LOGO' columns.
    
    Returns:
    plotly.graph_objects.Figure: The updated Plotly figure.
    """
    # Create a dictionary for quick lookup of logos by team name
    teams_df=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/teams_db.parquet?raw=true', engine='pyarrow')
    teams_df=teams_df.groupby(['team_full_name','team_logo','team_primary_color','team_secondary_color']).size().to_frame('cnt').reset_index()
    teams_df['team_primary_color'] = '#'+teams_df['team_primary_color']
    team_logos_dict = dict(zip(teams_df['team_full_name'], teams_df['team_logo']))

    headers = {
            'Host': 'www.nbamockdraftdatabase.com',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'Content-Type': 'text/html; charset=UTF-8',
            'sec-ch-ua': '"Chromium";v="88", "Google Chrome";v="88", ";Not A Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Dest': 'document',
            'Referer': 'https://www.nbamockdraftdatabase.com/mock-drafts/2024/nbc-sports-philadelphia-2024-brian-brennan?date=2024-06-11',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cookie': 'announcement_seen=1; announcement_seen=1; _nflmockdb_session=ZkJIWFdiMnllVkM0SGNMeHBxUE56VlVRa3lCR0N6ampwUzNLYnRxSy9JU256bENQb2k0cXZqallIWjNaWElPR2VQcVVXUmVOY0pGcS84MG81WFNJd2pnRHJJWVZVenRzdEg5SERjR0tEU0RELzRBMzFLL1Zsa3dYMld4Z2diWEpPK2pHZlVpbmplZkdGNFJNdlpDY0RnPT0tLWNjZDgyUGhLeGEyRG9xS3dndUdzZXc9PQ%3D%3D--f84bf880c0f79c2eccf440638588c5aaf682859e',
            'If-None-Match': 'W/"fe2fe61b64b30d8eff802d92afd8da1b"'
            }
    
    for trace in fig.data:
        # Replace scatter points with logos if the trace is a scatter plot

        # if isinstance(trace, go.Scatter) and trace.name in team_logos_dict:

            # st.write(fig.data)  
            # st.write(team_logos_dict)
            for i in range(len(trace.x)):
                # st.write(i)
                # st.write(team_logos_dict[trace.name])
                response = requests.get(team_logos_dict[trace.name],headers=headers)
                st.write(response.text)
                img = Image.open(BytesIO(response.content))

                st.image(img)
                fig.add_layout_image(
                    dict(
                        source=img,
                        x=trace.x[i],
                        y=trace.y[i],
                        xref="x",
                        yref="y",
                        sizex=0.1,  # Adjust the size as needed
                        sizey=0.1,
                        xanchor="center",
                        yanchor="middle"
                    )
                )
            # Hide original scatter points
            trace.marker.opacity = 0
    
    return fig



def add_sheets_data(df):
    df_img=df.groupby(['team'])['team_img'].agg(pd.Series.mode).to_frame('team_img').reset_index()
    team_img_dict = dict(zip(df_img['team'], df_img['team_img']))

    df_player_details=df.groupby(['player'])['player_details'].agg(pd.Series.mode).to_frame('player_details').reset_index()
    player_details_dict = dict(zip(df_player_details['player'], df_player_details['player_details']))

    url='https://docs.google.com/spreadsheets/d/1Fq_DdIsiMKe3tTkUj4eb9X5dlzAPnrRzkHvpNpJKnpk/gviz/tq?tqx=out:csv&gid=0'
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
    st.markdown('# NBA Mock Draft Database')
    df=get_s3_data('nba_mock_draft_db')
    df=df.loc[df.source != 'Tankathon']
    df=add_sheets_data(df)
    df=df.sort_values(['date','source_key','pick'],ascending=[True,True,True]).reset_index(drop=True)


    df.loc[df.date.isnull(), 'date'] = pd.to_datetime(df.source_key.str.split('date=',expand=True)[1])

    global_min_date = st.sidebar.date_input("Minimum Date for Mock Draft DB", 
                                 value = df.date.min(),
                                 min_value = df.date.min(),
                                 max_value = df.date.max())
    df = df.loc[pd.to_datetime(df['date'])>=pd.to_datetime(global_min_date)]

    df = df.sort_values(['date','source'],ascending=[True,True]).reset_index(drop=True)

    rising_falling(df)

    st.divider()

    t1,t2,t3,t4,t5,t6 = st.tabs(['Avg Player Rank','Times Player Mocked to Team','Filter by Team','Filter by Player','Filter by Mock Draft','Consensus Mock Draft'])

    with t1:
        st.markdown('### Avg Pick Over Time by Player')

        d1=df.groupby(['player','date']).agg(
            avg_pick=('pick','mean'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values(['date','player'],ascending=True).reset_index()

        d2=df.groupby(['player','team','date']).agg(
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

        top_n = st.slider("Top n players?", 2, 50, 25)
        top_players=df.groupby(['player']).agg(
            avg_pick=('pick','mean'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values(['times_picked','avg_pick'],ascending=[False,True]).reset_index().head(top_n)['player'].tolist()

        start_color='#FF6600'
        end_color='#55E0FF'
        color_list = generate_gradient(start_color, end_color, top_n)
        c1,c2=st.columns(2)
        fig = px.scatter(
            d1.loc[d1.player.isin(top_players)],
            x="date",
            y="avg_pick",
            hover_data=['team_picks'],
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
        
        fig = px.box(df.loc[df.player.isin(top_players)], 
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






    with t5:
        df['source_date'] = df['source'] + ' - ' +df['date'].astype('str')
        draft = st.selectbox('Pick a draft to view:',df['source_date'].unique())

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



        
        c1,c2,c3 = st.columns(3)
        min_date = c1.date_input("Minimum Date for Consensus", 
                                 value = df.date.min(),
                                 min_value = df.date.min(),
                                 max_value = df.date.max())
        
        df_bpa = df.loc[pd.to_datetime(df.date) >= pd.to_datetime(min_date)].groupby(['player']).agg(
            avg_pick=('pick','mean'),
            times_picked=('source_key',lambda x: x.nunique())
            ).sort_values(['avg_pick'],ascending=True).reset_index(drop=False).reset_index(drop=False)
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











    with st.expander('data'):
        st.write(df)


if __name__ == "__main__":
    #execute
    app()
