import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random


def get_closest_players(df, player_name, n=25):
    # Get the Cluster_x and Cluster_y values for the given player

    player_data = df[df['Player'] == player_name][['Cluster_x', 'Cluster_y']].values
    if len(player_data) == 0:
        return f"Player '{player_name}' not found in the dataframe."

    # Calculate the Euclidean distance from the given player to all other players
    df['distance'] = np.sqrt((df['Cluster_x'] - player_data[0][0])**2 + (df['Cluster_y'] - player_data[0][1])**2)

    # Sort by distance and exclude the player itself
    closest_players = df[df['Player'] != player_name].sort_values('distance').head(n)

    # Return the player names of the closest players
    return closest_players[['Player', 'distance']]


@st.fragment
def quick_clstr(df, num_cols, str_cols, color, player=None):
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

    if player is None:
        opacity_values = 0.75
        # If player is None, we want to include 'color' and 'color_discrete_map'
        scatter_kwargs = {
            'color': color,
            'color_discrete_map': discrete_color_map,
            'category_orders': {color: df.sort_values(color, ascending=True)[color].unique().tolist()},
            'opacity': opacity_values
        }
    else:
        opacity_values = np.where(df['Player'] == player, 0.9, 0.2)
        # If player is defined, we do NOT want to include 'color' or 'color_discrete_map'
        scatter_kwargs = {'opacity': opacity_values} # Empty dictionary, no color-related arguments


    fig=px.scatter(df,
                   x='Cluster_x',
                   y='Cluster_y',
                   width=800,
                   height=800,
                   # Pass the dynamically created keyword arguments
                   **scatter_kwargs,
                   # We will handle opacity manually below to ensure per-point
                   # if 'color' is used, or apply globally if 'color' is not used.
                   hover_data=str_cols + key_vals,
                   template='simple_white' # Added template for consistency with previous snippets
                  )
    

    fig.update_layout(legend_title_text=color,
            font_family='Futura',
            height=800,
            font_color='black',
                      )


    dvz['x']=dvz['x'].round(1)
    dvz['y']=dvz['y'].round(1)


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
                      opacity=.75,
                      marker=dict(size=16,line=dict(width=2,color='DarkSlateGrey'))
                      )
    fig.update_xaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    fig.update_yaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')


    ## Closest players section
    if player is not None:
        closest_players = get_closest_players(df, player)
        closest_players_list =[]
        closest_players_list = closest_players['Player'].tolist()
        closest_players_list.append(player)
        for i,r in df.iterrows():
            if r['Player'] in closest_players_list:
                fig.add_annotation(
                    x=r['Cluster_x'],
                    y=r['Cluster_y']+.015,
                    text=r['Player'],
                    bgcolor="gray",
                    opacity=.85,
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="#ffffff")
                        )
                

    for i in range(0,len(fig.data)):
        default_template = fig.data[i].hovertemplate
        updated_template = default_template.replace('=', ': ')
        fig.data[i].hovertemplate = updated_template

    st.plotly_chart(fig,use_container_width=True)


    ## closest player radial diagram
    if player is not None:
        closest_player_select = st.multiselect('Closest Players',
                                           df.sort_values('distance',ascending=True).head(100)['Player'].tolist(),
                                           df.sort_values('distance',ascending=True).head(10)['Player'].tolist()
                                           )


        df_closest = df.loc[df['Player'].isin(closest_player_select)].sort_values('distance',ascending=True).copy()

        scaler = StandardScaler()
        polar_scaled = scaler.fit_transform(df_closest[key_vals])
        scaled_cols = [f'{col}_scaled' for col in key_vals]
        df_closest[scaled_cols] = polar_scaled

        df_melted = pd.melt(df_closest,
                        id_vars=['Player'],
                        value_vars=scaled_cols,
                        var_name='Statistic',
                        value_name='Value')
        
        df_melted_unscaled = pd.melt(df_closest,
                        id_vars=['Player'],
                        value_vars=key_vals,
                        var_name='Statistic',
                        value_name='Value')
        
        df_melted_unscaled.columns = ['Player','Statistic','Value_Unscaled']
        df_melted_unscaled['Statistic'] = df_melted_unscaled['Statistic'].str.replace('_scaled','')
        df_melted['Statistic'] = df_melted['Statistic'].str.replace('_scaled','')
        df_melted = pd.merge(df_melted,df_melted_unscaled,left_on=['Player','Statistic'],right_on=['Player','Statistic'],how='left')

        fill_checkbox = st.checkbox('Include a fill on the visualization (note, some points will be covered)',value=True)

        ## create a radial diagram of the closest 10 players to the selected player
        fig = px.line_polar(df_melted,
                        r='Value',
                        theta='Statistic',
                        color='Player',
                        line_close=True,
                        template='simple_white',
                        hover_data=['Value_Unscaled'],
                        color_discrete_sequence=px.colors.qualitative.Plotly)
        if fill_checkbox:
            fig.update_traces(fill='toself')
        fig.update_layout(
            title=f"Closest Players to {player}",
            font_family='Futura',
            height=800,
            font_color='black',
            showlegend=True
        )

        fig.update_traces(
            mode='lines+markers',
            marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
            line=dict(width=4, dash='dash')
            )
        st.plotly_chart(fig)


    ## Top Attributes Section
    top_attributes = dvz.sort_values('distance_from_zero',ascending=False).field.tolist()
    for i, tab in enumerate(st.tabs(top_attributes)):
        with tab:
            val = top_attributes[i]
            fig=px.bar(df.sort_values(val,ascending=False),
                  x='Player',
                  color=color,
                  color_discrete_map = discrete_color_map,
                  y=val,
                  category_orders={color:df.sort_values(color,ascending=True)[color].unique().tolist()}
                  )
            fig.update_xaxes(categoryorder='total descending')
            fig.update_traces(marker=dict(
                line=dict(color='navy', width=2) 
                )
                )
            fig.update_layout(
               height=800,
               title=val,
               font=dict(
                   family='Futura',
                   size=12, 
                   color='black' 
                   )
            )
            for i in range(0,len(fig.data)):
               default_template = fig.data[i].hovertemplate
               updated_template = default_template.replace('=', ': ')
               fig.data[i].hovertemplate = updated_template
            st.plotly_chart(fig,use_container_width=True)








@st.cache_data(ttl=3600)
def load_nba_games():
    df = pd.read_parquet('https://github.com/aaroncolesmith/bbref/raw/refs/heads/main/data/nba_games.parquet', engine='pyarrow')

    return df

@st.cache_data(ttl=3600)
def load_nba_box_scores():
    df = pd.read_parquet('https://github.com/aaroncolesmith/bbref/raw/refs/heads/main/data/nba_box_scores.parquet', engine='pyarrow')
    df = df.loc[df['mp'] > 0]
    
    df['gmsc'] = pd.to_numeric(df['gmsc'])
    df['bpm'] = pd.to_numeric(df['bpm'])
    df['ortg'] = pd.to_numeric(df['ortg'])
    df['drtg'] = pd.to_numeric(df['drtg'])
    df['missed_shots'] = (df['fga'].fillna(0) - df['fg'].fillna(0))+(df['fta'].fillna(0) - df['ft'].fillna(0))
    df['all_stat'] = df['pts'] + df['trb'] + df['ast']

    df["darko_lite"] = (
    0.25 * df["bpm"] +
    0.20 * (df["ortg"] - df["drtg"]) +
    0.15 * df["usg_pct"] +
    0.15 * df["ts_pct"].fillna(0) +  # Fill missing TS% with 0 to avoid NaNs
    0.10 * df["ast_pct"] +
    0.10 * df["trb_pct"] +
    0.05 * df["stl_pct"] +
    0.05 * df["blk_pct"] -
    0.05 * df["tov_pct"].fillna(0)
    )

    return df


def aggregate_box_scores(df):
    df = df.groupby(['player']).agg(
                        team=('team',lambda x: ', '.join(set(x))),
                        mp=('mp','sum'),
                        gp=('game_id','nunique'),
                        fg=('fg','sum'),
                        fga=('fga','sum'),
                        threep=('3p','sum'),
                        threepa=('3pa','sum'),
                        ft=('ft','sum'),
                        fta=('fta','sum'),
                        pts=('pts','sum'),
                        ast=('ast','sum'),
                        trb=('trb','sum'),
                        stl=('stl','sum'),
                        blk=('blk','sum'),
                        orb=('orb','sum'),
                        drb=('drb','sum'),
                        bpm=('bpm','sum'),
                        tov=('tov','sum'),
                        gmsc=('gmsc','mean'),
                        orgt=('ortg','mean'),
                        drtg=('drtg','mean'),
                        usg_pct = ('usg_pct','mean'),
                        ts_pct=('ts_pct','mean'),
                        efg_pct=('efg_pct','mean'),
                        plus_minus=('+/-','sum'),
                        missed_shots=('missed_shots','sum'),
                        all_stat=('all_stat','sum'),
                        wins=('win','sum'),
                        losses=('loss','sum'),
                        playoff_games=('playoff_game','sum'),
                        playoff_wins=('playoff_win','sum'),
                        playoff_losses=('playoff_loss','sum'),
                        darko_lite=('darko_lite','mean'),
                            ).reset_index()
    
    df['fg_pct'] = df['fg'] / df['fga']
    df['3p_pct'] = df['threep'] / df['threepa']
    df['ft_pct'] = df['ft'] / df['fta']
    df['ppg']   = df['pts'] / df['gp']
    df['apg']   = df['ast'] / df['gp']
    df['rpg']   = df['trb'] / df['gp']
    df['spg']   = df['stl'] / df['gp']
    df['bpg']   = df['blk'] / df['gp']
    df['ppm']   = df['pts'] / df['mp']
    df['apm']   = df['ast'] / df['mp']
    df['rpm']   = df['trb'] / df['mp']
    df['spm']   = df['stl'] / df['mp']
    df['tovpm'] = df['tov'] / df['mp']
    df['plus_minus_per_min'] = df['plus_minus'] / df['mp']

    df['missed_shots_per_game'] = df['missed_shots']/df['gp']

    df['win_pct'] = df['wins']/df['gp']
    df['playoff_win_pct'] = df['playoff_wins']/df['playoff_games']


    return df



def rename_columns(df):
    df.rename(columns={
            "3par":"3pa Rate",
            "ts_pct":"True Shot Pct",
            "pts":"Points",
            "ast":"Assists",
            "efg_pct":"Eff FG Pct",
            "drtg":"Def Rtg",
            "ortg":"Off Rtg",
            "ast":"Assists",
            "efg_pct":"Eff FG Pct",
            "drtg":"Def Rtg",
            "ortg":"Off Rtg",
            "3par":"3pa Rate",
            "ft":"Free Throws",
            "fta":"Free Throws Attempted",
            "stl_pct":"Steal Pct",
            "player":"Player",
            "team":"Team",
            "mp":"Minutes",
            "fg":"Field Goals",
            "fga":"Field Goals Attempted",
            "fg_pct":"Field Goal Pct",
            "ftp_pct":"Free Throw Pct",
            "trb_pct":"Total Rebound Pct",
            "trb":"Rebounds",
            "stl":"Steals",
            "bpm":"Box +/-",
            "drb":"Def Reb",
            "orb":"Off Reb",
            "pf":"Fouls",
            "tov":"Turnovers",
            "ftr":"FT Rate",
            # "drtg":"Def Rtg",
            "ortg":"Off Rtg",
            "3par":"3pa Rate",
            "ft":"Free Throws",
            "fta":"Free Throws Attempted",
            "stl_pct":"Steal Pct",
            "player":"Player",
            "team":"Team",
            "mp":"Minutes",
            "fg":"Field Goals",
            "fga":"Field Goals Attempted",
            "fg_pct":"Field Goal Pct",
            "ortg":"Off Rtg",
            "gmsc":"Game Score",
            "3p_pct":'3p Pct',
            'ppm':'Points Per Minute',
            'apm':'Assists Per Minute',
            'rpm':'Rebounds Per Minute',
            'spm':'Steals Per Minute',
            'ppg':'Points Per Game',
            'apg':'Assists Per Game',
            'rpg':'Rebounds Per Game',
            'spg':'Steals Per Game',
            'bpg':'Blocks Per Game',
            'threep':'3 Pointers Made',
            'threepa':'3 Pointers Attempted',
            'gp':'Games Played',
            'blk':'Blocks',
            'orgt':'Offensive Rating',
            'drtg':'Defensive Rating',
            'ft_pct':'Free Throw Pct',
            "usg_pct": "Usage Rate",
            'darko_lite':'Darko Lite'
        },
            inplace=True)
    
    return df



def single_game_viz(d1, d2):
    st.subheader('Single Game Visualization')
    c1,c2=st.columns([1,3])
    date = c1.date_input(
        "Select a date / month for games",
        value=pd.to_datetime(d1.date.max()),
        min_value=pd.to_datetime('1966-02-19'),
        max_value=pd.to_datetime(d1.date.max())
        )

    # date='2023-08-14'
    date=pd.to_datetime(date)

    # st.write(d1.loc[d1.date == date])
    d1['game_str'] = d1['date'].dt.strftime('%Y-%m-%d') + ' - ' + d1['visitor_team'] + ' ' + d1['visitor_score'].astype('str')+'-'+d1['home_score'].astype('str')+' ' +d1['home_team']
    games=d1.loc[d1.date == date]['game_str'].tolist()
    game_select = c2.selectbox('Select a game: ', games)

    game_id = d1.loc[d1.game_str == game_select].game_id.min()
    # st.write(game_id)

    df = d2.loc[d2.game_id==game_id]
    try:
        df['+/-'] = pd.to_numeric(df['+/-'].astype('str').str.replace('+',''))
    except:
        pass

    df = rename_columns(df)

    num_cols=[]
    non_num_cols=[]
    for col in df.columns:
        if col in ['date','last_game']:
            try:
                df[col]=pd.to_datetime(df[col])
            except:
                pass
        else:
            try:
                df[col] = pd.to_numeric(df[col])
                num_cols.append(col)
            except:
                # print(f'{col} failed')
                non_num_cols.append(col)
    color='team'
    with st.form(key='clstr_form'):
        num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
        non_num_cols_select=st.multiselect('Select non numeric columns for hover data',non_num_cols,['Player'])
        list_one=['Cluster']
        list_two=df.columns.tolist()
        color_options=list_one+list_two

        color_select=st.selectbox('What attribute should color points on the graph?',color_options)
        mp_filter=st.slider('Filter players by minimum minutes played?', min_value=0.0, max_value=df.Minutes.max(), value=0.0, step=1.0)
        df=df.query("Minutes > @mp_filter")
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        quick_clstr(df.fillna(0), num_cols_select, non_num_cols_select, color_select)





def date_range_viz(d1, d2):
    st.subheader('Date Range Visualization')

    c1,c2,c3=st.columns([1,1,2])
    start_date = c1.date_input(
            "Select a start date",
            value=pd.to_datetime(d1.date.max())-pd.DateOffset(months=1),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(d1.date.max())
            )
    end_date = c2.date_input(
            "Select an end date",
            value=pd.to_datetime(d1.date.max()),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(d1.date.max())
            )
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)
    df = d2.loc[(d2.date >= start_date) & (d2.date <= end_date)]

    df = pd.merge(df.loc[df.player!='Eddie Johnson'],d1).sort_values(by=['date', 'player']).copy()
    

    df['playoff_game'] = np.where(df['game_type'] == 'Playoffs', 1, 0)
    df['win'] = np.select(
        [
            (df['team'] == df['home_team']) & (df['home_score'] > df['visitor_score']),
            (df['team'] == df['home_team']) & (df['home_score'] < df['visitor_score']),
            (df['team'] != df['home_team']) & (df['home_score'] < df['visitor_score']),
            (df['team'] != df['home_team']) & (df['home_score'] > df['visitor_score'])
        ],
        [
            1,  
            0,  
            1,  
            0   
        ],
    )
    df['loss'] = np.where(df['win'] == 0, 1, 0)
    df['playoff_win'] = np.where(df['playoff_game'] == 1, df['win'], 0)
    df['playoff_loss'] = np.where(df['playoff_game'] == 1, 1-df['win'], 0)



    df = aggregate_box_scores(df)
    df = rename_columns(df)


    num_cols=[]
    non_num_cols=[]
    for col in df.columns:
        if col in ['date','last_game']:
            try:
                df[col]=pd.to_datetime(df[col])
            except:
                pass
        else:
            try:
                df[col] = pd.to_numeric(df[col])
                num_cols.append(col)
            except:
                # print(f'{col} failed')
                non_num_cols.append(col)
        

        

        # non_num_cols=['player']
    color='team'
    with st.form(key='clstr_form'):
        num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
        non_num_cols_select=st.multiselect('Select non numeric columns for hover data',non_num_cols,['Player'])
        list_one=['Cluster']
        list_two=df.columns.tolist()
        color_options=list_one+list_two

        color_select=st.selectbox('What attribute should color points on the graph?',color_options)
        mp_filter=st.slider('Filter players by minimum minutes played?', min_value=0.0, max_value=df.Minutes.max(), value=0.0, step=1.0)
        df=df.query("Minutes > @mp_filter")
        submit_button = st.form_submit_button(label='Submit')
    
    
    if submit_button:
        quick_clstr(df.fillna(0), num_cols_select, non_num_cols_select, color_select)



def analyze_null_percentages(df: pd.DataFrame, threshold: float = 0.50) -> dict:

    null_percentages = df.isnull().sum() / len(df) * 100
    null_percentages = null_percentages.reset_index()
    null_percentages.columns = ['column', 'null_percentage']

    columns_below_threshold = null_percentages.loc[null_percentages['null_percentage'] < (threshold * 100), 'column'].tolist()

    return columns_below_threshold


    # return {
    #     'null_percentages': null_percentages,
    #     'columns_above_threshold': columns_above_threshold
    # }


def player_comparison_viz(d1, d2):
    st.subheader('Player Comparison')

    d = pd.merge(d2.loc[d2.player!='Eddie Johnson'],d1).sort_values(by=['date', 'player']).copy()
    
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

    players = d.groupby('player').agg(all_stat=('all_stat','mean')).sort_values('all_stat',ascending=False).reset_index()['player'].tolist()
    player = st.selectbox('Select a player',players)

    color='team'
    with st.form(key='clstr_form'):
        

        st.write(f'Player selected: {player}')
        games_played = d.loc[d.player == player].game_id.nunique()

        c1,c2,c3=st.columns(3)
        games_played_select = c1.slider('Through his first number of games played',10,games_played,games_played)

        minutes_played = int(d.loc[d.player == player].head(games_played_select).mp.sum())
        minutes_played_half = int(minutes_played/2)
        minutes_played_select = c2.slider(f'Filter players that played less than x amount of minutes -- for reference, {player} played {minutes_played} minutes',0,minutes_played,minutes_played_half)
        old_players_select = c3.checkbox('Include older players (they don\'t have advanced stats like Usage Rate)',value=True)
        df_filtered = d.groupby('player').head(games_played_select)
        df_agg = aggregate_box_scores(df_filtered)
        ## filter out players that have not played at least 75% of the games
        df_agg = df_agg.loc[df_agg['gp'] >= (games_played_select * .75)].copy()
        df_agg = rename_columns(df_agg)
        df_agg=df_agg.query("Minutes > @minutes_played_select")

        if old_players_select:
            df_agg = df_agg.loc[(df_agg['Minutes'] >= minutes_played_select)].fillna(0)
        else:
            df_agg = df_agg.loc[(df_agg['Minutes'] >= minutes_played_select)&(df_agg['Usage Rate'].notnull())].fillna(0)

        num_cols, non_num_cols = get_num_cols(df_agg)
        

        num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
        non_num_cols_select=st.multiselect('Select non numeric columns for hover data',non_num_cols,['Player'])
        list_one=['Cluster']
        list_two=df_agg.columns.tolist()
        color_options=list_one+list_two

        # color_select=st.selectbox('What attribute should color points on the graph?',color_options)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        quick_clstr(df_agg.fillna(0), num_cols_select, non_num_cols_select, 'Cluster', player)




    # with st.form(key='select_form'):
    #     # players = d.groupby('player').agg(all_stat=('all_stat','mean')).sort_values('all_stat',ascending=False).reset_index()['player'].tolist()
    #     # player = st.selectbox('Select a player',players)
    #     games_played = d.loc[d.player == player].game_id.nunique()

    #     # num_cols_select = st.multiselect('Which stats should be used?',num_cols,num_cols)
    #     giddy_up = st.form_submit_button('Giddy Up')

    


def get_num_cols(df):
    num_cols=[]
    non_num_cols=[]
    for col in df.columns:
        if col in ['date','last_game']:
            try:
                df[col]=pd.to_datetime(df[col])
            except:
                pass
        else:
            try:
                df[col] = pd.to_numeric(df[col])
                num_cols.append(col)
            except:
                # print(f'{col} failed')
                non_num_cols.append(col)
    return num_cols, non_num_cols




def app():

    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
        )
    

    st.title("NBA Player Stats Visualization")

    d1 = load_nba_games()
    d2 = load_nba_box_scores()


    # with st.form('method_select'):
    #     st.subheader('Would you live to visualize a single game, a date range of games or a player comparison model?')
    #     method = st.radio('Select a method:', ['Single Game', 'Date Range', 'Player Comparison'])
    #     submit_button = st.form_submit_button(label='Submit')

    ## use a drop down to select the method
    method = st.selectbox('Select a method:', ['Single Game', 'Date Range', 'Player Comparison'])
    


    if method == 'Single Game':
        single_game_viz(d1, d2)
    elif method == 'Date Range':
        date_range_viz(d1, d2)
    elif method == 'Player Comparison':
        player_comparison_viz(d1, d2)










if __name__ == "__main__":
    #execute
    app()