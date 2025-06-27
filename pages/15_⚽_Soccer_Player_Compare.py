import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import functools as ft
import random
import re
from datetime import datetime, timedelta


def extract_league_from_url(url):
    """
    Extracts the league name from a given Fbref match URL.

    The league name is expected to be the last part of the URL path,
    following the four-digit year (e.g., '2025-Major-League-Soccer').

    Args:
        url (str): The URL of the match page.

    Returns:
        str or None: The extracted league name (e.g., 'Major-League-Soccer'),
                     or None if the pattern is not found.
    """
    if not isinstance(url, str):
        return None # Handle non-string inputs gracefully

    # Get the last segment of the URL path (e.g., 'New-York-City-FC-New-York-Red-Bulls-May-17-2025-Major-League-Soccer')
    path_segment = url.split('/')[-1]

    # Use a regular expression to find the pattern:
    # \d{4}   - Matches exactly four digits (for the year)
    # -       - Matches a literal hyphen
    # ([a-zA-Z0-9-]+) - This is the capturing group:
    #                   [a-zA-Z0-9-] - Matches any alphanumeric character or a hyphen
    #                   +            - Matches one or more of the preceding characters
    # $       - Asserts position at the end of the string
    # This regex specifically looks for the league name that appears directly
    # after the four-digit year and a hyphen at the very end of the URL segment.
    match = re.search(r'\d{4}-([a-zA-Z0-9-]+)$', path_segment)

    if match:
        # If a match is found, return the content of the first capturing group,
        # which is the league name.
        return match.group(1)
    else:
        # If no match is found, return None. This could happen if the URL
        # format deviates from the expected pattern.
        return None





def rename_columns(df):
    df.rename(columns={
            "performance_gls":"goals",
            "performance_ast":"assists",
            "performance_pk":"pks",
            "performance_pkatt":"pk_att",
            "performance_sh":"shots",
            "performance_sot":"shots_on_goal",
            "performance_crdy":"yellow_card",
            "performance_crdr":"red_card",
            "performance_touches":"touches",
            "performance_tkl":"tackles",
            "performance_int":"ints",
            "performance_blocks":"blocks",
            "expected_xg":"xg",
            "expected_npxg":"npxg",
            "expected_xag":"xag",
            "sca_sca":"sca",
            "sca_gca":"gca",
            "passes_cmp":"passes_cmp",
            "passes_att":"passes_att",
            "passes_cmp_pct":"passes_cmp_pct",
            "passes_prgp":"progressive_passes",
            "carries_carries":"carries",
            "carries_prgc":"progressive_carries",
            "take-ons_att":"take_ons_attempted",
            "take-ons_succ":"take_ons_successful",
            "total_cmp":"del_asd",
            "total_att":"del_dd",
            "total_cmp_pct":"del_cmp",
            "total_totdist":"passing_distance",
            "total_prgdist":"progressive_passing_distance",
            "short_cmp":"short_passes_cmp",
            "short_att":"short_passes_att",
            "short_cmp_pct":"short_passes_cmp_pct",
            "medium_cmp":"med_passes_cmp",
            "medium_att":"med_passes_att",
            "medium_cmp_pct":"med_passes_cmp_pct",
            "long_cmp":"long_passes_cmp",
            "long_att":"long_passes_att",
            "long_cmp_pct":"long_passes_cmp_pct",
            "ast":"del_ast",
            "xag":"del_xag",
            "xa":"xa",
            "kp":"key_passes",
            "1/3":"final_third_passes",
            "ppa":"passes_penalty_area",
            "crspa":"crosses_penalty_area",
            "prgp":"del_prgp",
            "att":"del_att",
            "pass_types_live":"live_ball_passes",
            "pass_types_dead":"dead_ball_passes",
            "pass_types_fk":"fk_passes",
            "pass_types_tb":"through_balls",
            "pass_types_sw":"switches",
            "pass_types_crs":"crosses",
            "pass_types_ti":"throw_ins",
            "pass_types_ck":"corners",
            "corner_kicks_in":"corners_inswing",
            "corner_kicks_out":"corners_outswing",
            "corner_kicks_str":"corners_straight",
            "outcomes_cmp":"del_pass_com",
            "outcomes_off":"offside_passes",
            "outcomes_blocks":"blocked_passes",
            "tackles_tkl":"del_tackles_tkl",
            "tackles_tklw":"tackles_won",
            "tackles_def_3rd":"tackles_def_3rd",
            "tackles_mid_3rd":"tackles_mid_3rd",
            "tackles_att_3rd":"tackles_att_3rd",
            "challenges_tkl":"dribblers_tackled",
            "challenges_att":"dribblers_challenged",
            "challenges_tkl_pct":"tackle_pct",
            "challenges_lost":"challenges_lost",
            "blocks_blocks":"del_blocks_blocks",
            "blocks_sh":"block_shot",
            "blocks_pass":"block_pass",
            "int":"del_int",
            "tkl+int":"tkl+int",
            "clr":"clearances",
            "err":"errors",
            "touches_touches":"del_touches_touches",
            "touches_def_pen":"touches_def_pen",
            "touches_def_3rd":"touches_def_3rd",
            "touches_mid_3rd":"touches_mid_3rd",
            "touches_att_3rd":"touches_att_3rd",
            "touches_att_pen":"touches_att_pen",
            "touches_live":"touches_live_ball",
            "take-ons_succ_pct":"take_ons_succ_pct",
            "take-ons_tkld":"take_ons_tackled",
            "take-ons_tkld_pct":"take_ons_tkld_pct",
            "carries_totdist":"carries_total_distance",
            "carries_prgdist":"carries_progressive_distance",
            "carries_1/3":"carries_third",
            "carries_cpa":"carries_into_penalty_area",
            "carries_mis":"carries_mis",
            "carries_dis":"carries_dis",
            "receiving_rec":"passes_received",
            "receiving_prgr":"progressive_passes_received",
            "performance_2crdy":"del_performance_2crdy",
            "performance_fls":"fouls_committed",
            "performance_fld":"fouls_drawn",
            "performance_off":"offsides",
            "performance_crs":"del_performance_crs",
            "performance_tklw":"del_performance_tklw",
            "performance_pkwon":"pk_won",
            "performance_pkcon":"pk_conceded",
            "performance_og":"own_goals",
            "performance_recov":"balls_recovered",
            'player':'Player'
        },
             inplace=True)
    
    return df







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


@st.fragment()
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

    # st.write('**5 Closest Players**')
    # for i,r in closest_players.head(5).iterrows():
    #     st.write(f"{r['player']}: {round(r['distance'],2)}")
    st.plotly_chart(fig)
    fig_scatter = fig


    closest_player_select = st.multiselect('Closest Players',
                                           df.sort_values('distance',ascending=True).head(100)['player'].tolist(),
                                           df.sort_values('distance',ascending=True).head(10)['player'].tolist()
                                           )


    df_closest = df.loc[df['player'].isin(closest_player_select)].sort_values('distance',ascending=True).copy()
    # df_closest[key_vals] = scaler.fit_transform(df_closest[key_vals])


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

    ## add a checkbox to remove the fill
    fill_checkbox = st.checkbox('Include a fill on the visualization (note, some points will be covered)',value=True)

    ## add a button to download the data
    csv = df_melted.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"{player}_closest_players.csv",
        mime="text/csv",
        key='download-csv'
    )



    ## create a radial diagram of the closest 10 players to the selected player
    fig = px.line_polar(df_melted,
                     r='Value',
                     theta='Statistic',
                     color='player',
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


    return df, fig_scatter



@st.cache_data
def get_data():
    df = pd.read_parquet('https://github.com/aaroncolesmith/data_action_network/raw/refs/heads/main/data/fb_ref_data.parquet', engine='pyarrow')
    d = pd.read_parquet('https://github.com/aaroncolesmith/soccer_data/raw/refs/heads/main/data/fb_ref_data_box_scores.parquet', engine='pyarrow')
    d = pd.merge(d,df,left_on=['match_url'],right_on=['url'],how='left')
    d['date'] = pd.to_datetime(d['date'])

    d.rename(columns={'Player':'player',
                      }, inplace=True)
    
    # Apply the function to the 'url' column to create the new 'league' column
    d['league'] = d['url'].apply(extract_league_from_url)
    d['count_of_null'] = d.isnull().sum(axis=1)

    ## if the league is 'Serie-A' and the count of nulls is greater than 20, then rename league to 'Serie-A-Ecuador'
    d.loc[(d['league'] == 'Serie-A') & (d['count_of_null'] > 20), 'league'] = 'Serie-A-Ecuador'

    ## if league is 'Serie-A' and the team is in brazil_league_teams, then rename league to 'Serie-A-Brazil'
    brazil_league_teams = d.loc[d['player']== 'Breno']['Home'].unique().tolist()
    d.loc[(d['league'] == 'Serie-A') & (d['team'].isin(brazil_league_teams)), 'league'] = 'Serie-A-Brazil'

    # d['birthdate'] = d.apply(lambda row: calculate_birthdate(str(row['date']), row['age']), axis=1)


    ## fill in an age (55-001) for players that don't have an age
    d['age'] = d['age'].fillna('55-001')

    d['age_year'] = d['age'].fillna(0).str.split('-').str[0].astype(int)
    d['age_day_of_year'] = d['age'].fillna(0).str.split('-').str[1].astype(int)
    d['age_total_days'] = d['age_year'] * 365 + d['age_day_of_year']
    d['age_year'] = d['age_year'].astype(int)
    d['birthdate'] = pd.to_datetime(d['date']) - pd.to_timedelta(d['age_total_days'], unit='D')

    ## if a player's name is 'Antony' and his birthdate is in the month of september 2001, change his name to 'Antony Alves Santos'
    d.loc[(d['player'] == 'Antony') & (d['birthdate'].dt.month == 9) & (d['birthdate'].dt.year == 2001), 'player'] = 'Antony Alves Santos'

    return d




def app():
    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
        )
    
    st.title('Soccer Player Compare')

    d = get_data()

    per_90_stat_list = ['goals','assists','ints','xg','xag','xa',
                     'passes_cmp','carries','carries_total_distance',
                     'progressive_passes','progressive_carries','take_ons_attempted','take_ons_successful',
                     'crosses','tackles_won',
                     'through_balls','key_passes'
                     ]
    appended_per_90_stat_list = [f'{stat}_per_90' for stat in per_90_stat_list]



    num_cols_base = ['games_played', 'min', 'goals', 'assists', 'pks', 'pk_att', 'shots', 'shots_on_goal', 
                     'yellow_card', 'red_card', 'touches', 'tackles', 'ints', 'blocks', 'xg', 'xa', 
                     'npxg', 'xag', 'sca', 'gca', 'passes_cmp', 'passes_att', 'passes_cmp_pct', 'progressive_passes', 
                     'passing_distance', 'progressive_passing_distance', 'short_passes_cmp', 'short_passes_att', 'short_passes_cmp_pct', 
                     'med_passes_cmp', 'med_passes_att', 'med_passes_cmp_pct', 'long_passes_cmp', 'long_passes_att', 'long_passes_cmp_pct', 
                     'key_passes', 'final_third_passes', 'passes_penalty_area', 'crosses_penalty_area', 'through_balls', 'switches', 'crosses', 
                     'throw_ins', 'offside_passes', 'blocked_passes', 'tackles_won', 'dribblers_tackled', 'dribblers_challenged', 'tackle_pct', 'carries', 
                     'progressive_carries', 'carries_progressive_distance', 'carries_total_distance', 'take_ons_attempted', 
                     'take_ons_successful', 'take_ons_succ_pct', 'take_ons_tackled', 'take_ons_tkld_pct',

                     'xg_conversion_rate_per_90'
            ]
    
    ## add per 90 stats to num_cols_base
    num_cols_base = num_cols_base + appended_per_90_stat_list


    with st.expander('data example',expanded=False):
        st.write(d.sample(5))

        st.write(d.loc[d['league']== 'Serie-A'].sample(5))
        st.write('Number of players:',d.player.nunique())
        st.write('Number of games:',d.match_url.nunique())
        st.write('Number of rows:',d.shape[0])
        st.write('Number of columns:',d.shape[1])
        st.write('Columns:',d.columns.tolist())



    league_selector = st.multiselect('Select a league',
        d.league.unique().tolist(),
        d.league.unique().tolist()
        )
    d = d.loc[d['league'].isin(league_selector)].copy()

    c1,c2,c3=st.columns(3)
    players = d.groupby('player').agg(xg=('xg','sum')).sort_values('xg',ascending=False).reset_index()['player'].tolist()
    player = c1.selectbox('Select a player',players)


    with st.form(key='select_form'):
        # players = d.groupby('player').agg(all_stat=('all_stat','mean')).sort_values('all_stat',ascending=False).reset_index()['player'].tolist()
        # player = st.selectbox('Select a player',players)
        # games_played = d.loc[d.player == player].match_url.nunique()
        c1,c2,c3=st.columns(3)
        # games_played_select = c1.slider('How many recent games?',10,games_played,games_played)

        start_date = c1.date_input(
            "Select a start date",
            value=pd.to_datetime(d.date.max())-pd.DateOffset(months=3),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(d.date.max())
            )
        end_date = c2.date_input(
            "Select an end date",
            value=pd.to_datetime(d.date.max()),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(d.date.max())
            )
        start_date=pd.to_datetime(start_date)
        end_date=pd.to_datetime(end_date)
        





        minutes_played = int(d.loc[
            (d.player == player)&
            (d['date'] >= start_date) &
            (d['date'] <= end_date)
            ]['min'].sum())
        minutes_played_half = int(minutes_played/2)
        minutes_played_select = c2.slider(f'Filter players that played less than x amount of minutes',0,minutes_played,100)
        

        # start_date = d.loc[d.player == player].date.min()
        # end_date = d.loc[d.player == player].date.max()
        # c1,c2 = st.columns(2)
        # start_date_select = c1.date_input("Start date", value = start_date, min_value = start_date, max_value = end_date)
        # end_date_select = c2.date_input("Start date", value = end_date, min_value = start_date_select, max_value = end_date)

        num_cols_select = st.multiselect('Which stats should be used?',num_cols_base,num_cols_base)
        giddy_up = st.form_submit_button('Giddy Up')


    if giddy_up:
        df_filtered = d.loc[(d['date'] >= start_date) & (d['date'] <= end_date)].copy()

        df_agg = df_filtered.groupby(['player']).agg(
            team=('team',lambda x: ', '.join(x.unique())),
            league=('league',lambda x: ', '.join(x.unique())),
            pos=('pos',lambda x: x.value_counts().index[0]),
            games_played=('match_url','nunique'),
            min=('min','sum'),
            goals=('goals','sum'),
            assists=('assists','sum'),
            pks=('pks','sum'),
            pk_att=('pk_att','sum'),
            shots=('shots','sum'),
            shots_on_goal=('shots_on_goal','sum'),
            yellow_card=('yellow_card','sum'),
            red_card=('red_card','sum'),
            touches=('touches','sum'),
            tackles=('tackles','sum'),
            ints=('ints','sum'),
            blocks=('blocks','sum'),
            xg=('xg','sum'),
            xa=('xa','sum'),
            npxg=('npxg','sum'),
            xag=('xag','sum'),
            sca=('sca','sum'),
            gca=('gca','sum'),
            passes_cmp=('passes_cmp','sum'),
            passes_att=('passes_att','sum'),
            passes_cmp_pct=('passes_cmp_pct','mean'),
            progressive_passes=('progressive_passes','sum'),
            passing_distance=('passing_distance','sum'),
            progressive_passing_distance=('progressive_passing_distance','sum'),
            short_passes_cmp=('short_passes_cmp','sum'),
            short_passes_att=('short_passes_att','sum'),
            short_passes_cmp_pct=('short_passes_cmp_pct','mean'),
            med_passes_cmp=('med_passes_cmp','sum'),
            med_passes_att=('med_passes_att','sum'),
            med_passes_cmp_pct=('med_passes_cmp_pct','mean'),
            long_passes_cmp=('long_passes_cmp','sum'),
            long_passes_att=('long_passes_att','sum'),
            long_passes_cmp_pct=('long_passes_cmp_pct','mean'),
            key_passes=('key_passes','sum'),
            final_third_passes=('final_third_passes','sum'),
            passes_penalty_area=('passes_penalty_area','sum'),
            crosses_penalty_area=('crosses_penalty_area','sum'),
            through_balls=('through_balls','sum'),
            switches=('switches','sum'),
            crosses=('crosses','sum'),
            throw_ins=('throw_ins','sum'),
            offside_passes=('offside_passes','sum'),
            blocked_passes=('blocked_passes','sum'),
            tackles_won=('tackles_won','sum'),
            dribblers_tackled=('dribblers_tackled','sum'),
            dribblers_challenged=('dribblers_challenged','sum'),
            tackle_pct=('tackle_pct','mean'),
            carries=('carries','sum'),
            progressive_carries=('progressive_carries','sum'),
            carries_progressive_distance=('carries_progressive_distance','sum'),
            carries_total_distance=('carries_total_distance','sum'),
            take_ons_attempted=('take_ons_attempted','sum'),
            take_ons_successful=('take_ons_successful','sum'),
            take_ons_succ_pct=('take_ons_succ_pct','mean'),
            take_ons_tackled=('take_ons_tackled','sum'),
            take_ons_tkld_pct=('take_ons_tkld_pct','mean'),
        ).reset_index()



        for stat_col in per_90_stat_list:
            df_agg[f'{stat_col}_per_90'] = (df_agg[stat_col] / df_agg['min']) * 90

        df_agg['xg_conversion_rate_per_90'] = (df_agg['goals_per_90'] / df_agg['xg_per_90']).replace([np.inf, -np.inf], 0).fillna(0)


        non_num_cols = ['player','team','league','pos']

        df_clstr = df_agg.loc[(df_agg['min'] >= minutes_played_select)].fillna(0)
        # df_clstr = df_agg.fillna(0)

        if player in df_clstr['player'].unique().tolist():
            df_results, fig_scatter = quick_clstr(df_clstr, num_cols_select, non_num_cols, 'Cluster',player)

        else:
            st.write('Adjust your filters because you have filtered out your player')





   



if __name__ == "__main__":
    #execute
    app()