import streamlit as st
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly_express as px
import random
from io import BytesIO
from posthog import Posthog


posthog = Posthog(
  project_api_key='phc_izEfF9RePzi6AdGbi3x0NeXPjCu1ShPQtCPkS5HJH7C',
  host='https://us.i.posthog.com',
  disable_geoip=False
)



def get_closest_players(df, player_name, n=25):
    # Get the Cluster_x and Cluster_y values for the given player
    # st.write(player_name)
    # st.write(df.head(3))
    player_data = df[df['player'] == player_name][['Cluster_x', 'Cluster_y']].values
    if len(player_data) == 0:
        return f"Player '{player_name}' not found in the dataframe."

    # Calculate the Euclidean distance from the given player to all other players
    df['distance'] = np.sqrt((df['Cluster_x'] - player_data[0][0])**2 + (df['Cluster_y'] - player_data[0][1])**2)

    # Sort by distance and exclude the player itself
    closest_players = df[df['player'] != player_name].sort_values('distance').head(n)

    # Return the player names of the closest players
    return closest_players[['player', 'distance']]



def quick_clstr(df, num_cols, str_cols, color, player):
    df1=df.copy()
    df1=df1[num_cols]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df1)

    pca = PCA(n_components=2)
    pca.fit(x_scaled)

    x_pca = pca.transform(x_scaled)

    kmeans = KMeans(n_clusters=5, random_state=2).fit_predict(x_pca)

    p=pd.DataFrame(np.transpose(pca.components_[0:2, :]))
    p=pd.merge(p,pd.DataFrame(np.transpose(num_cols)),left_index=True,right_index=True)
    p.columns = ['x','y','field']

    df['Cluster'] = kmeans.astype('str')
    df['Cluster_x'] = x_pca[:,0]
    df['Cluster_y'] = x_pca[:,1]
    df['Cluster'] = pd.to_numeric(df['Cluster'])

    pc=p.copy()
    pc=pc[['x','y']]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(pc)

    pca = PCA(n_components=2)
    pca.fit(x_scaled)

    x_pca = pca.transform(x_scaled)

    num_clusters=3
    if len(num_cols) == 2:
        num_clusters=2

    kmeans = KMeans(n_clusters=num_clusters, random_state=2).fit_predict(x_pca)

    p['Cluster'] = kmeans.astype('str')
    p['Cluster_x'] = x_pca[:,0]
    p['Cluster_y'] = x_pca[:,1]
    p['Cluster'] = pd.to_numeric(p['Cluster'])

    pviz=p.groupby(['Cluster']).agg({'field' : lambda x: ', '.join(x),'x':'mean','y':'mean'}).reset_index()

    mean_x = p['x'].mean()
    mean_y = p['y'].mean()

    x_factor = (df.Cluster_x.max() / p.x.max())*.75
    y_factor = (df.Cluster_y.max() / p.y.max())*.75

    p['x'] = p['x'].round(2)
    p['y'] = p['y'].round(2)

    p['distance'] = np.sqrt((p['x'] - mean_x)**2 + (p['y'] - mean_y)**2)
    p['distance_from_zero'] = np.sqrt((p['x'] - 0)**2 + (p['y'] - 0)**2)

    dvz=pd.DataFrame()
    dvz=pd.concat([dvz,p.sort_values('x',ascending=True).head(5)])
    dvz=pd.concat([dvz,p.sort_values('x',ascending=True).tail(5)])
    dvz=pd.concat([dvz,p.sort_values('y',ascending=True).head(5)])
    dvz=pd.concat([dvz,p.sort_values('y',ascending=True).tail(5)])
    dvz=dvz.drop_duplicates()

    # key_vals=p.sort_values('distance_from_zero',ascending=False).head(20).field.tolist()
    key_vals=dvz.field.tolist()



    df[color] = df[color].astype('str')



    # Predefined list of hex colors
    hex_colors = ['#ffd900', '#ff2a00', '#35d604', '#59ffee', '#1d19ff','#ff19fb']

    # Function to generate a random hex color
    def generate_random_hex_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # Get unique values from the color column
    unique_values = df[color].unique()

    # If there are more unique values than predefined colors, generate additional random colors
    while len(hex_colors) < len(unique_values):
        hex_colors.append(generate_random_hex_color())

    # Create a dictionary to map unique values to colors (this is your discrete color map)
    discrete_color_map = {value: hex_colors[i] for i, value in enumerate(unique_values)}

    # Apply the color mapping to the dataframe
    df['assigned_color'] = df[color].map(discrete_color_map)

    # color_discrete_map

    closest_players = get_closest_players(df, player)
    closest_players_list =[]
    closest_players_list = closest_players.player.tolist()
    closest_players_list.append(player)




    fig=px.scatter(df,
                   x='Cluster_x',
                   y='Cluster_y',
                #    width=800,
                #    height=800,
                   template='simple_white',
                  #  color=color,
                   color_discrete_map = discrete_color_map,
                #    color_discrete_sequence=marker_color,
                   hover_data=str_cols+key_vals,
                   opacity=np.where(df['player']==player,0.9, 0.2),
                   category_orders={color:df.sort_values(color,ascending=True)[color].unique().tolist()}
                  )
    fig.update_layout(legend_title_text=color,
            font_family='Futura',
            height=800,
            font_color='black',

                      )


    dvz['x']=dvz['x'].round(1)
    dvz['y']=dvz['y'].round(1)
    # for i, r in p.sort_values('distance_from_zero',ascending=False).head(20).groupby(['x','y',]).agg(field=('field', lambda x: '<br>'.join(x))).reset_index().iterrows():
    for i,r in dvz.groupby(['x','y',]).agg(field=('field', lambda x: '<br>'.join(x))).reset_index().iterrows():
            x_val = r['x']*x_factor
            y_val = r['y']*y_factor

            if x_val < df.Cluster_x.min():
                x_val = df.Cluster_x.min()
            if x_val > df.Cluster_x.max():
                x_val = df.Cluster_x.max()

            if y_val < df.Cluster_y.min():
                y_val = df.Cluster_y.min()
            if y_val > df.Cluster_y.max():
                y_val = df.Cluster_y.max()

            fig.add_annotation(
                x=x_val,
                y=y_val,
                text=r['field'],
                showarrow=False,
                # bgcolor="#F5F5F5",
                opacity=.25,
                font=dict(
                    color="black",
                    size=12
                    )
                )
    fig.update_traces(mode='markers',
                      # opacity=.75,
                      marker=dict(size=16,line=dict(width=2,color='DarkSlateGrey'))
                      )
    fig.update_xaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    fig.update_yaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')




    for i,r in df.iterrows():
      # if r['distance'] < .75:
      if r['player'] in closest_players_list:
        fig.add_annotation(
        x=r['Cluster_x'],
        y=r['Cluster_y']+.015,
        text=r['player'],
        bgcolor="gray",
        opacity=.85,
        showarrow=False,
        font=dict(
                size=12,
                color="#ffffff"))
    #   elif r['distance'] < 1:
    #     fig.add_annotation(
    #     x=r['Cluster_x'],
    #     y=r['Cluster_y']+.015,
    #     text=r['player'],
    #     bgcolor="gray",
    #     opacity=.85,
    #     showarrow=False,
    #     font=dict(
    #             size=8,
    #             color="#ffffff"))
      # else:
      #   fig.add_annotation(
      #   x=r['Cluster_x'],
      #   y=r['Cluster_y']+.5,
      #   text=r['Player'],
      #   bgcolor="gray",
      #   opacity=.5,
      #   showarrow=False,
      #   font=dict(
      #           size=8,
      #           color="#ffffff"
      #           )
      #   )


    default_template = fig.data[0].hovertemplate  # Get the existing template
    updated_template = default_template.replace('=', ': ')

    fig.update_traces(hovertemplate=updated_template)

    st.plotly_chart(fig)
    fig_scatter = fig
    # st.write(fig.data[0]['hovertemplate'])

    # for val in dvz.sort_values('distance_from_zero',ascending=False).field.tolist():
    #    fig=px.bar(df.sort_values(val,ascending=False),
    #               x='player',
    #               color=color,
    #               color_discrete_map = discrete_color_map,
    #               y=val,
    #               category_orders={color:df.sort_values(color,ascending=True)[color].unique().tolist()}
    #               )
    #    fig.update_xaxes(categoryorder='total descending')
    #    fig.update_traces(marker=dict(
    #     #    color='lightblue',
    #        line=dict(color='navy', width=2)
    #        )
    #    )
    #    fig.update_layout(
    #     font=dict(
    #     family='Futura',  # Set font to Futura
    #     size=12,          # You can adjust the font size if needed
    #     color='black'
    #     ))


    #    fig.show()
    return df, fig_scatter






@st.cache_data
def load_data():
   df=pd.read_parquet('https://drive.google.com/file/d/1S2N4a3lhohq_EtuY3aMW_d9nIsE4Bruk/view?usp=sharing', engine='pyarrow')
   return df


@st.cache_data
def load_google_file(code):
    url = f"https://drive.google.com/uc?export=download&id={code}"
    file = requests.get(url)
    bytesio = BytesIO(file.content)
    return pd.read_parquet(bytesio)

def app():
    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
        )
    
    posthog.capture('test-id', 'nba_player_similarity_load_event')

    code='1_0FAJsULjo-gz2pvy365dQNqbo1ORDMU'
    d1=load_google_file(code)
    # st.write(d1.tail(10))
    code='1S2N4a3lhohq_EtuY3aMW_d9nIsE4Bruk'
    d2=load_google_file(code)
    # st.write(d2.tail(10))

    d = pd.merge(d2.loc[d2.player!='Eddie Johnson'],d1.loc[d1.game_type == 'Regular Season']).sort_values(by=['date', 'player']).copy()

    d['missed_shots'] = (d['fga'].fillna(0) - d['fg'].fillna(0))+(d['fta'].fillna(0) - d['ft'].fillna(0))
    d['all_stat'] = d['pts'] + d['trb'] + d['ast']

    num_cols=['Minutes',
        'Field Goals',
        'Field Goals Attempted',
        '3P Made',
        '3P Attempted',
        'Free Throws',
        'Free Throws Attempted',
        'Off Reb',
        'Def Reb',
        'Rebounds',
        'Assists',
        'Steals',
        'Blocks',
        'Turnovers',
        'Fouls',
        'Points',
        'Box +/-',
        'True Shot Pct',
        'Eff FG Pct',
        'Off Rebound Pct',
        'Total Rebound Pct',
        'Assist %',
        'Steal Pct',
        'Block %',
        'TOV %',
        'Usage Rate',
        'Off Rtg',
        'Def Rtg',
        'Missed Shots',
        'Pts + Reb + Ast',
        'Points Per Game',
        'Field Goal Pct',
        '3p %',
        'FT %',
        'Rebounds Per Game',
        'Assists Per Game',
        'Steals Per Game',
        'Blocks Per Game',
        'TOV Per Game',
        'Points Per 36',
        'Rebounds Per 36',
        'Assists Per 36',
        'Steals Per 36',
        'Blocks Per 36',
        '+/-',
        'TOV Per 36']

    players = d.groupby('player').agg(all_stat=('all_stat','mean')).sort_values('all_stat',ascending=False).reset_index()['player'].tolist()
    player = st.selectbox('Select a player',players)
    
    with st.form(key='select_form'):
        # players = d.groupby('player').agg(all_stat=('all_stat','mean')).sort_values('all_stat',ascending=False).reset_index()['player'].tolist()
        # player = st.selectbox('Select a player',players)
        games_played = d.loc[d.player == player].game_id.nunique()
        c1,c2,c3=st.columns(3)
        games_played_select = c1.slider('Through his first number of games played',10,games_played,games_played)
        minutes_played = int(d.loc[d.player == player].mp.sum())
        minutes_played_half = int(minutes_played/2)
        minutes_played_select = c2.slider(f'Filter players that played less than x amount of minutes -- for reference, {player} played {minutes_played} minutes',0,minutes_played,minutes_played_half)
        old_players_select = c3.checkbox('Include older players (they don\'t have advanced stats like Usage Rate)',value=True)

        # start_date = d.loc[d.player == player].date.min()
        # end_date = d.loc[d.player == player].date.max()
        # c1,c2 = st.columns(2)
        # start_date_select = c1.date_input("Start date", value = start_date, min_value = start_date, max_value = end_date)
        # end_date_select = c2.date_input("Start date", value = end_date, min_value = start_date_select, max_value = end_date)

        num_cols_select = st.multiselect('Which stats should be used?',num_cols,num_cols)
        giddy_up = st.form_submit_button('Giddy Up')

    if giddy_up:
        posthog.capture('test-id', f'nba_player_similarity_{player}')
        df_filtered = d.groupby('player').head(games_played_select)
        df_agg = df_filtered.groupby(['player']).agg(
            games_played=('game_id','count'),
            mp=('mp','sum'),
            fg=('fg','sum'),
            fga=('fga','sum'),
            fg3=('3p','sum'),
            fg3a=('3pa','sum'),
            ft=('ft','sum'),
            fta=('fta','sum'),
            orb=('orb','sum'),
            drb=('drb','sum'),
            trb=('trb','sum'),
            ast=('ast','sum'),
            stl=('stl','sum'),
            blk=('blk','sum'),
            tov=('tov','sum'),
            pf=('pf','sum'),
            pts=('pts','sum'),
            plus_minus=('+/-','sum'),
            box_plus_minus=('bpm','sum'),
            ts_pct=('ts_pct','mean'),
            efg_pct=('efg_pct','mean'),
            three_par = ('3par','mean'),
            orb_pct = ('orb_pct','mean'),
            drb_pct = ('drb_pct','mean'),
            trb_pct = ('trb_pct','mean'),
            ast_pct = ('ast_pct','mean'),
            stl_pct = ('stl_pct','mean'),
            blk_pct = ('blk_pct','mean'),
            tov_pct = ('tov_pct','mean'),
            usg_pct = ('usg_pct','mean'),
            ortg = ('ortg','mean'),
            drtg = ('drtg','mean'),
            missed_shots=('missed_shots','sum'),
            all_stat=('all_stat','sum'),
        ).reset_index()

        df_agg['ppg'] = df_agg['pts']/df_agg['games_played']
        df_agg['fg_pct'] = df_agg['fg']/df_agg['fga']
        df_agg['3p_pct'] = df_agg['fg3']/df_agg['fg3a']
        df_agg['ft_pct'] = df_agg['ft']/df_agg['fta']
        df_agg['rpg'] = df_agg['trb']/df_agg['games_played']
        df_agg['apg'] = df_agg['ast']/df_agg['games_played']
        df_agg['spg'] = df_agg['stl']/df_agg['games_played']
        df_agg['bpg'] = df_agg['blk']/df_agg['games_played']
        df_agg['tovpg'] = df_agg['tov']/df_agg['games_played']

        df_agg['ppm'] = 36*(df_agg['pts']/df_agg['mp'])
        df_agg['rpm'] = 36*(df_agg['trb']/df_agg['mp'])
        df_agg['apm'] = 36*(df_agg['ast']/df_agg['mp'])
        df_agg['spm'] = 36*(df_agg['stl']/df_agg['mp'])
        df_agg['bpm'] = 36*(df_agg['blk']/df_agg['mp'])
        df_agg['tovpm'] = 36*(df_agg['tov']/df_agg['mp'])

        df_agg.rename(columns=dict(sorted({
                "3par": "3pa Rate",
                "ts_pct": "True Shot Pct",
                "pts": "Points",
                "ast": "Assists",
                "efg_pct": "Eff FG Pct",
                "drtg": "Def Rtg",
                "ortg": "Off Rtg",
                "ft": "Free Throws",
                "fta": "Free Throws Attempted",
                "stl_pct": "Steal Pct",
                "team": "Team",
                "mp": "Minutes",
                "fg": "Field Goals",
                "fga": "Field Goals Attempted",
                "fg_pct": "Field Goal Pct",
                "ftp_pct": "Free Throw Pct",
                "trb_pct": "Total Rebound Pct",
                "trb": "Rebounds",
                "stl": "Steals",
                "box_plus_minus": "Box +/-",
                "drb": "Def Reb",
                "orb": "Off Reb",
                "pf": "Fouls",
                "tov": "Turnovers",
                "ftr": "FT Rate",
                "games_played": "Games",
                "fg3": "3P Made",
                "fg3a": "3P Attempted",
                "blk": "Blocks",
                "orb_pct": "Off Rebound Pct",
                "drb_pct": "Def Rebound Pct",
                "ast_pct": "Assist %",
                "blk_pct": "Block %",
                "tov_pct": "TOV %",
                "usg_pct": "Usage Rate",
                "all_stat": "Pts + Reb + Ast",
                "ppg": "Points Per Game",
                "rpg": "Rebounds Per Game",
                "apg": "Assists Per Game",
                "3p_pct": "3p %",
                "ft_pct": "FT %",
                "spg": "Steals Per Game",
                "bpg": "Blocks Per Game",
                "tovpg": "TOV Per Game",
                "ppm": "Points Per 36",
                "rpm": "Rebounds Per 36",
                "apm": "Assists Per 36",
                "spm": "Steals Per 36",
                "tovpm": "TOV Per 36",
                "bpm": "Blocks Per 36",
                "missed_shots": "Missed Shots",
                "plus_minus": '+/-'
            }.items())), inplace=True)
        non_num_cols = ['player']
        if old_players_select:
            df_clstr = df_agg.loc[(df_agg['Minutes'] >= minutes_played_select)].fillna(0)
        else:
            df_clstr = df_agg.loc[(df_agg['Minutes'] >= minutes_played_select)&(df_agg['Usage Rate'].notnull())].fillna(0)
        with st.expander('Raw Data'):
            st.write(df_clstr)

        if player in df_clstr['player'].unique().tolist():
            df_results, fig_scatter = quick_clstr(df_clstr, num_cols_select, non_num_cols, 'Cluster',player)
        else:
            st.write('Adjust your filters because you have filtered out your player')


    # c1,c2=st.columns([1,3])
    # date = c1.date_input(
    #     "Select a date / month for games",
    #     value=pd.to_datetime(d1.date.max()),
    #     min_value=pd.to_datetime('1966-02-19'),
    #     max_value=pd.to_datetime(d1.date.max())
    #     )

    # # date='2023-08-14'
    # date=pd.to_datetime(date)

    # # st.write(d1.loc[d1.date == date])
    # d1['game_str'] = d1['date'].dt.strftime('%Y-%m-%d') + ' - ' + d1['visitor_team'] + ' ' + d1['visitor_score'].astype('str')+'-'+d1['home_score'].astype('str')+' ' +d1['home_team']
    # games=d1.loc[d1.date == date]['game_str'].tolist()
    # game_select = c2.selectbox('Select a game: ', games)

    # game_id = d1.loc[d1.game_str == game_select].game_id.min()
    # # st.write(game_id)

    # df = d2.loc[d2.game_id==game_id]
    # try:
    #     df['+/-'] = pd.to_numeric(df['+/-'].astype('str').str.replace('+',''))
    # except:
    #     pass
    # # st.write(df)

    # # with st.form(key='game_select_form'):
    # #     game_select = st.selectbox('Select a match: ', games)



    # # r = requests.get(f'https://www.basketball-reference.com/leagues/NBA_{season}_games-{month.lower()}.html')
    # # if r.status_code==200:
    # #     soup = BeautifulSoup(r.content, 'html.parser')
    # #     table = soup.find('table', attrs={'id': 'schedule'})
    # #     if table:
    # #         df = pd.read_html(str(table))[0]
    # #         # game_id = []
    # #         # game_url = []
    # #         for row in table.findAll('tr'):
    # #             try:
    # #                 if 'csk' in str(row):
    # #                     game_id.append(str(row).split('csk="')[1].split('"')[0])
    # #                     game_url.append(str(row).split('data-stat="box_score_text"><a href="')[1].split('"')[0])
    # #                     # visitor_team.append(str(row).split('data-stat="visitor_team_name"><a href="/teams/')[1].split('/')[0])
    # #                     # home_team.append(str(row).split('data-stat="home_team_name"><a href="/teams/')[1].split('/')[0])
    # #                     # visitor_score.append(str(row).split('data-stat="visitor_pts">')[1].split('<')[0])
    # #                     # home_score.append(str(row).split('data-stat="home_pts">')[1].split('<')[0])
    # #             except:
    # #                 pass
    # #         df['game_id'] = game_id
    # #         game_url_df = pd.DataFrame({'game_url':game_url})
    # #         df=pd.concat([df,game_url_df],axis=1)
    # #         df.columns=['date','time','visitor','visitor_pts','home','home_pts','del_1','del_2','attendance','arena','notes','game_id','game_url']
    # #         df = df.loc[df.home_pts>0]
    # #         df['visitor_pts']=df['visitor_pts'].astype('str').str.replace('\.0','',regex=True)
    # #         df['home_pts']=df['home_pts'].astype('str').str.replace('\.0','',regex=True)
    # #         for col in df.columns:
    # #             if 'del' in col:
    # #                 del df[col]
    # # df['game_string'] = df['visitor'] + ' ' + df['visitor_pts'].astype('str') + ' - ' + df['home'] + ' ' + df['home_pts'].astype('str')

    # # games=df['game_string'].tolist()

    # # with st.form(key='game_select_form'):
    # #     game_select = st.selectbox('Select a match: ', games)

    # #     submit_button = st.form_submit_button(label='Submit')

    # # game_id=df.query('game_string == @game_select').game_id.min()
    # # visiting_team=df.query('game_string == @game_select').visitor.min()
    # # home_team=df.query('game_string == @game_select').home.min()

    # # if submit_button:
    # #     d=get_box_score(game_id,visiting_team,home_team)


    # #     st.write(d.head(3))

    # df.rename(columns={
    #         "3par":"3pa Rate",
    #         "ts_pct":"True Shot Pct",
    #         "pts":"Points",
    #         "ast":"Assists",
    #         "efg_pct":"Eff FG Pct",
    #         "drtg":"Def Rtg",
    #         "ortg":"Off Rtg",
    #         "ast":"Assists",
    #         "efg_pct":"Eff FG Pct",
    #         "drtg":"Def Rtg",
    #         "ortg":"Off Rtg",
    #         "3par":"3pa Rate",
    #         "ft":"Free Throws",
    #         "fta":"Free Throws Attempted",
    #         "stl_pct":"Steal Pct",
    #         "player":"Player",
    #         "team":"Team",
    #         "mp":"Minutes",
    #         "fg":"Field Goals",
    #         "fga":"Field Goals Attempted",
    #         "fg_pct":"Field Goal Pct",
    #         "ftp_pct":"Free Throw Pct",
    #         "trb_pct":"Total Rebound Pct",
    #         "trb":"Rebounds",
    #         "stl":"Steals",
    #         "bpm":"Box +/-",
    #         "drb":"Def Reb",
    #         "orb":"Off Reb",
    #         "pf":"Fouls",
    #         "tov":"Turnovers",
    #         "ftr":"FT Rate",
    #         # "drtg":"Def Rtg",
    #         "ortg":"Off Rtg",
    #         "3par":"3pa Rate",
    #         "ft":"Free Throws",
    #         "fta":"Free Throws Attempted",
    #         "stl_pct":"Steal Pct",
    #         "player":"Player",
    #         "team":"Team",
    #         "mp":"Minutes",
    #         "fg":"Field Goals",
    #         "fga":"Field Goals Attempted",
    #         "fg_pct":"Field Goal Pct",
    #         "ortg":"Off Rtg",
    #     },
    #         inplace=True)


    #     # for x in d.columns:
    #     #     if 'del_' in x:
    #     #         del d[x]

    # num_cols=[]
    # non_num_cols=[]
    # for col in df.columns:
    #     if col in ['date','last_game']:
    #         try:
    #             df[col]=pd.to_datetime(df[col])
    #         except:
    #             pass
    #     else:
    #         try:
    #             df[col] = pd.to_numeric(df[col])
    #             num_cols.append(col)
    #         except:
    #             # print(f'{col} failed')
    #             non_num_cols.append(col)
        

        

    #     # non_num_cols=['player']
    # color='team'
    # with st.form(key='clstr_form'):
    #     num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
    #     non_num_cols_select=st.multiselect('Select non numeric columns for hover data',non_num_cols,['Player'])
    #     list_one=['Cluster']
    #     list_two=df.columns.tolist()
    #     color_options=list_one+list_two

    #     color_select=st.selectbox('What attribute should color points on the graph?',color_options)
    #     mp_filter=st.slider('Filter players by minimum minutes played?', min_value=0.0, max_value=df.Minutes.max(), value=0.0, step=1.0)
    #     df=df.query("Minutes > @mp_filter")
    #     submit_button = st.form_submit_button(label='Submit')


        

    #     # # for color in num_cols:
    #     # #   p=quick_clstr(d.fillna(0), num_cols, non_num_cols, color)

    # if submit_button:
    #     quick_clstr(df.fillna(0), num_cols_select, non_num_cols_select, color_select)







   



if __name__ == "__main__":
    #execute
    app()