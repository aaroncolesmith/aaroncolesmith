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
  


    default_template = fig.data[0].hovertemplate  # Get the existing template
    updated_template = default_template.replace('=', ': ')

    fig.update_traces(hovertemplate=updated_template)

    st.plotly_chart(fig)
    fig_scatter = fig


    closest_player_select = st.multiselect('Closest Players',
                                           df.sort_values('distance',ascending=True).head(100)['player'].tolist(),
                                           df.sort_values('distance',ascending=True).head(10)['player'].tolist()
                                           )


    df_closest = df.loc[df['player'].isin(closest_player_select)].sort_values('distance',ascending=True).copy()

    scaler = StandardScaler()
    polar_scaled = scaler.fit_transform(df_closest[key_vals])
    scaled_cols = [f'{col}_scaled' for col in key_vals]
    df_closest[scaled_cols] = polar_scaled

    df_melted = pd.melt(df_closest,
                    id_vars=['player'],
                    value_vars=scaled_cols,
                    var_name='Statistic',
                    value_name='Value')
    
    df_melted_unscaled = pd.melt(df_closest,
                    id_vars=['player'],
                    value_vars=key_vals,
                    var_name='Statistic',
                    value_name='Value')
    
    df_melted_unscaled.columns = ['player','Statistic','Value_Unscaled']
    df_melted_unscaled['Statistic'] = df_melted_unscaled['Statistic'].str.replace('_scaled','')
    df_melted['Statistic'] = df_melted['Statistic'].str.replace('_scaled','')
    df_melted = pd.merge(df_melted,df_melted_unscaled,left_on=['player','Statistic'],right_on=['player','Statistic'],how='left')

    ## create a radial diagram of the closest 10 players to the selected player
    fig = px.line_polar(df_melted,
                     r='Value',
                     theta='Statistic',
                     color='player',
                     line_close=True,
                     template='simple_white',
                     hover_data=['Value_Unscaled'],
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(fill='toself')
    fig.update_layout(
        title=f"Closest Players to {player}",
        font_family='Futura',
        height=800,
        font_color='black',
        showlegend=True
    )

    fig.update_traces(mode='markers',
                      # opacity=.75,
                      marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey'))
                      )
    st.plotly_chart(fig)


    return df, fig_scatter






# @st.cache_data
# def load_data():
#    df=pd.read_parquet('https://drive.google.com/file/d/1S2N4a3lhohq_EtuY3aMW_d9nIsE4Bruk/view?usp=sharing', engine='pyarrow')
#    return df


# @st.cache_data
# def load_google_file(code):
#     url = f"https://drive.google.com/uc?export=download&id={code}"
#     file = requests.get(url)
#     bytesio = BytesIO(file.content)
#     return pd.read_parquet(bytesio)



@st.cache_data(ttl=3600)
def load_nba_games():
    df = pd.read_parquet('https://github.com/aaroncolesmith/bbref/raw/refs/heads/main/data/nba_games.parquet', engine='pyarrow')
    return df

@st.cache_data(ttl=3600)
def load_nba_box_scores():
    df = pd.read_parquet('https://github.com/aaroncolesmith/bbref/raw/refs/heads/main/data/nba_box_scores.parquet', engine='pyarrow')
    return df



def app():
    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
        )
    
    posthog.capture('test-id', 'nba_player_similarity_load_event')

    d1 = load_nba_games()
    d2 = load_nba_box_scores()

    d = pd.merge(d2.loc[d2.player!='Eddie Johnson'],d1).sort_values(by=['date', 'player']).copy()

    d['missed_shots'] = (d['fga'].fillna(0) - d['fg'].fillna(0))+(d['fta'].fillna(0) - d['ft'].fillna(0))
    d['all_stat'] = d['pts'] + d['trb'] + d['ast']
    d['playoff_game'] = np.where(d['game_type'] == 'Playoffs', 1, 0)
    d['win'] = np.select(
        [
            (d['team'] == d['home_team']) & (d['home_score'] > d['visitor_score']),
            (d['team'] == d['home_team']) & (d['home_score'] < d['visitor_score']),
            (d['team'] != d['home_team']) & (d['home_score'] < d['visitor_score']),
            (d['team'] != d['home_team']) & (d['home_score'] > d['visitor_score'])
        ],
        [
            1,  
            0,  
            1,  
            0   
        ],
    )
    d['loss'] = np.where(d['win'] == 0, 1, 0)
    d['playoff_win'] = np.where(d['playoff_game'] == 1, d['win'], 0)
    d['playoff_loss'] = np.where(d['playoff_game'] == 1, 1-d['win'], 0)

    # st.write(d.loc[d.player == 'Shai Gilgeous-Alexander'])

    # st.write(d.loc[d.player == 'Shai Gilgeous-Alexander'].groupby(['player','game_id']).agg(
    #     Minutes=('mp','sum'),
    #     Count=('game_url','count')
    # ).reset_index().sort_values('Count',ascending=False).head(10)
    # )

    # st.write(d.loc[d.player == 'Shai Gilgeous-Alexander'].groupby(['player','season']).agg(
    #     Minutes=('mp','sum'),
    #     Count=('game_url','count'),
    #     Playoff_Count=('playoff_game','sum'),
    # ).reset_index().sort_values('Count',ascending=False)
    # )



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
        'Missed Shots Per Game',
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
        'FTA Per 36',
        '3P Made Per 36',
        'Missed Shots Per 36',
        '+/- Per 36',
        '+/-',
        'TOV Per 36',
        'Games Played',
        'Wins',
        'Losses',
        'Playoff Wins',
        'Playoff Losses',
        'Win %',
        'Playoff Win %',
        'Playoff Games',
        
        ]

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

    # if giddy_up:
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
        wins=('win','sum'),
        losses=('loss','sum'),
        playoff_games=('playoff_game','sum'),
        playoff_wins=('playoff_win','sum'),
        playoff_losses=('playoff_loss','sum'),
    ).reset_index()


    ## filter out players that have not played at least 75% of the games
    df_agg = df_agg.loc[df_agg['games_played'] >= (games_played_select * .75)].copy()

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
    df_agg['ftapm'] = 36*(df_agg['fta']/df_agg['mp'])
    df_agg['fg3pm'] = 36*(df_agg['fg3']/df_agg['mp'])
    df_agg['missed_shots_pm'] = 36*(df_agg['missed_shots']/df_agg['mp'])
    df_agg['plus_minus_pm'] = 36*(df_agg['plus_minus']/df_agg['mp'])

    df_agg['missed_shots_per_game'] = df_agg['missed_shots']/df_agg['games_played']

    df_agg['win_pct'] = df_agg['wins']/df_agg['games_played']
    df_agg['playoff_win_pct'] = df_agg['playoff_wins']/df_agg['playoff_games']

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
            "ftapm": "FTA Per 36",
            "fg3pm": "3P Made Per 36",
            "missed_shots_pm": "Missed Shots Per 36",
            "plus_minus_pm": "+/- Per 36",
            "missed_shots": "Missed Shots",
            "missed_shots_per_game": "Missed Shots Per Game",
            "plus_minus": '+/-',
            "games_played": "Games Played",
            "wins": "Wins",
            "losses": "Losses",
            "playoff_wins": "Playoff Wins",
            "playoff_losses": "Playoff Losses",
            "playoff_win_pct": "Playoff Win %",
            "playoff_games": "Playoff Games",
            "win_pct": "Win %",
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


   



if __name__ == "__main__":
    #execute
    app()