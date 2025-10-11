import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
from utils.utils import test_util, quick_clstr_util


@st.cache_data
def get_data():
    df = pd.read_parquet('https://github.com/aaroncolesmith/data_action_network_clean/raw/refs/heads/main/data/fb_ref_data.parquet', engine='pyarrow')
    d = pd.read_parquet('https://github.com/aaroncolesmith/soccer_data/raw/refs/heads/main/data/fb_ref_data_box_scores.parquet', engine='pyarrow')
    df[['home_score', 'visitor_score']] = df['Score'].str.split('–', expand=True)

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

    d.loc[(d['player'] == 'Marquinhos') & (d['birthdate'].dt.month == 4) & (d['birthdate'].dt.year == 2003), 'player'] = 'Marquinhos - Marcus Vinícius Oliveira Alencar'

    d.loc[(d['player'] == 'Marquinhos') & (d['birthdate'].dt.month == 1) & (d['birthdate'].dt.year == 1997), 'player'] = 'Marquinhos - Marcos Vinícius Sousa Natividade'
    d.loc[(d['player'] == 'Marquinhos') & (d['birthdate'].dt.month == 2) & (d['birthdate'].dt.year == 1997), 'player'] = 'Marquinhos - Marcos Vinícius Sousa Natividade'

    d.loc[(d['player'] == 'Marquinhos') & (d['birthdate'].dt.month == 10) & (d['birthdate'].dt.year == 1999), 'player'] = 'Marquinhos - José Marcos Costa Martins'

    d.loc[(d['player'] == 'Rodri') & (d['birthdate'].dt.month == 2) & (d['birthdate'].dt.year == 1989), 'player'] = 'Rodri - Alberto Rodríguez Expósito'
    d.loc[(d['player'] == 'Rodri') & (d['birthdate'].dt.month == 3) & (d['birthdate'].dt.year == 1989), 'player'] = 'Rodri - Alberto Rodríguez Expósito'

    d.loc[(d['player'] == 'Rodri') & (d['birthdate'].dt.month == 6) & (d['birthdate'].dt.year == 1990), 'player'] = 'Rodri - Rodrigo Ríos Lozano'

    d.loc[(d['player'] == 'João Pedro') & (d['birthdate'].dt.month == 9) & (d['birthdate'].dt.year == 2001), 'player'] = 'João Pedro Junqueira de Jesus'
    d.loc[(d['player'] == 'João Pedro') & (d['birthdate'].dt.month == 10) & (d['birthdate'].dt.year == 2001), 'player'] = 'João Pedro Junqueira de Jesus'
    
    d.loc[(d['player'] == 'Vitinha') & (d['birthdate'].dt.month == 3) & (d['birthdate'].dt.year == 2000), 'player'] = 'Vítor Manuel Carvalho Oliveira'

    # st.write(d.loc[d['player'] == 'Vitinha'].sort_values('date',ascending=False).tail(30))


    d['home_team'] = d['Home']
    d['visitor_team'] = d['Away']

    d['game_id'] = d['match_url'].str.split('/').str[-1].str.replace('.html','')
    d['Minutes'] = pd.to_numeric(d['min'], errors='coerce')

    return d




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



def single_game_viz(df):
    st.subheader('Single Game Visualization')
    c1,c2=st.columns([1,3])
    date = c1.date_input(
        "Select a date / month for games",
        value=pd.to_datetime(df.date.max()),
        min_value=pd.to_datetime('1966-02-19'),
        max_value=pd.to_datetime(df.date.max())
        )

    # date='2023-08-14'
    date=pd.to_datetime(date)

    # st.write(d1.loc[d1.date == date])
    df['game_str'] = df['date'].dt.strftime('%Y-%m-%d') + ' - ' + df['visitor_team'] + ' ' + df['visitor_score'].astype('str')+'-'+df['home_score'].astype('str')+' ' +df['home_team']
    games=df.loc[df.date == date]['game_str'].unique().tolist()
    game_select = c2.selectbox('Select a game: ', games)


    game_id = df.loc[df.game_str == game_select].game_id.min()
    # st.write(game_id)
    d2 = df.copy()

    df = d2.loc[d2.game_id==game_id]
    try:
        df['+/-'] = pd.to_numeric(df['+/-'].astype('str').str.replace('+',''))
    except:
        pass

    df = rename_columns(df)

    num_cols=[]
    non_num_cols=[]
    all_cols = df.columns.tolist()
    for col in all_cols:
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


    ## if an element from the num_cols_remove list is in num_cols, then remove it
    num_cols_remove = ['#','min','age_year','Attendance','Notes','count_of_null','age_day_of_year','age_total_days','Wk','birthdate','home_xg','away_xg','home_score','visitor_score']
    num_cols = [col for col in num_cols if col not in num_cols_remove]



    with st.form(key='clstr_form'):
        c1,c2=st.columns(2)
        num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
        non_num_cols_select=st.multiselect('Select columns for hover data',all_cols,['Player'])
        list_one=['Cluster']
        list_two=df.columns.tolist()
        color_options=list_one+list_two

        color_select=st.selectbox('What attribute should color points on the graph?',color_options)
        mp_filter=st.slider('Filter players by minimum minutes played?', min_value=0.0, max_value=df.Minutes.max(), value=0.0, step=1.0)
        df=df.query("Minutes > @mp_filter")
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        quick_clstr_util(df.fillna(0), num_cols_select, non_num_cols_select, color_select)





def aggregate_box_scores(df):

    df['gp'] = 1
    # Define the list of columns to sum
    sum_cols = ['gp',
        'min', 
                'goals', 
                'assists',
                'shots',
                'key_passes',
                'offsides',
                'tackles',
                'blocks',
                'clearances',
                'xg',
                'npxg',
                'xag','xa','sca','gca','passes_cmp','passes_att','progressive_passes','carries',
                'progressive_carries','take_ons_attempted','take_ons_successful','passing_distance',
                'through_balls','blocked_passes','dribblers_tackled','dribblers_challenged',
                'errors','offsides','fouls_drawn','fouls_committed','pk_won','aerial_duels_won',
                'win','loss','tie'
                ]

    # Create the aggregation dictionary
    # The format should be {new_col_name: (original_col_name, aggregation_function)}
    agg_dict = {col: ('_'.join(col.split()), 'sum') for col in sum_cols}

    # Add the 'team' aggregation separately, using the correct tuple format
    agg_dict['team'] = ('team', lambda x: ', '.join(set(x)))

    # Perform the aggregation using the new dictionary
    df_agg = df.groupby('player').agg(**agg_dict).reset_index()
    df_agg['Minutes'] = df_agg['min']


    for col in sum_cols:
        df_agg[f'{col}_per_90'] = (df_agg[col] / df_agg['min']) * 90


    return df_agg




def date_range_viz(df):
    st.subheader('Date Range Visualization')

    c1,c2,c3=st.columns([1,1,2])
    start_date = c1.date_input(
            "Select a start date",
            value=pd.to_datetime(df.date.max())-pd.DateOffset(months=1),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(df.date.max())
            )
    end_date = c2.date_input(
            "Select an end date",
            value=pd.to_datetime(df.date.max()),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(df.date.max())
            )
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)
    league_list = df.loc[(df.date >= start_date) & (df.date <= end_date)]['league'].unique().tolist()

    league_select = c3.multiselect('Select a league', league_list, ['Premier-League','La-Liga','Bundesliga','Ligue-1','Serie-A'])


    d2 = df.copy()
    df = d2.loc[(d2.date >= start_date) & (d2.date <= end_date) &(d2.league.isin(league_select))].copy()

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
    # df['playoff_win'] = np.where(df['playoff_game'] == 1, df['win'], 0)
    # df['playoff_loss'] = np.where(df['playoff_game'] == 1, 1-df['win'], 0)



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
        

    ## if league_select is empty, then show a warning
    if len(league_select) > 0:
        color='team'
        with st.form(key='clstr_form'):
            num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
            non_num_cols_select=st.multiselect('Select non numeric columns for hover data',non_num_cols,['Player'])
            list_one=['Cluster']
            list_two=df.columns.tolist()
            color_options=list_one+list_two

            show_player_select = st.multiselect('Select players to show on the graph',df['Player'].unique().tolist())

            color_select=st.selectbox('What attribute should color points on the graph?',color_options)
            mp_filter=st.slider('Filter players by minimum minutes played?', min_value=0.0, max_value=df.Minutes.max(), value=0.0, step=1.0)
            df=df.query("Minutes > @mp_filter")
            submit_button = st.form_submit_button(label='Submit')
        
        
        if submit_button:
            quick_clstr_util(df.fillna(0), num_cols_select, non_num_cols_select, color_select, player=None, player_list=show_player_select)
    else:
        st.warning('Please select at least one league to visualize data.')



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


def player_comparison_viz(df):
    st.subheader('Player Comparison')

    d = df.copy()

    ## rename min to mp
    # d.rename(columns={'min':'mp'}, inplace=True)
    d['mp'] = pd.to_numeric(d['min'], errors='coerce')

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

    d['tie'] = np.where(d['home_score'] == d['visitor_score'], 1, 0)

    d['loss'] = np.where((d['win'] == 0) & (d['tie']==0), 1, 0)

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


    with st.form(key='clstr_form'):
        c1,c2,c3=st.columns(3)
        league_selector = st.multiselect('Select a league',
            d.league.unique().tolist(),
            d.league.unique().tolist()
            )
        d = d.loc[d['league'].isin(league_selector)].copy()

        c1,c2,c3=st.columns(3)
        players = d.groupby('player').agg(xg=('xg','sum')).sort_values('xg',ascending=False).reset_index()['player'].tolist()
        player = c1.selectbox('Select a player',players)

        player_min_date = d.loc[d['player']==player]['date'].min()
        player_max_date = d.loc[d['player']==player]['date'].max()

        color='team'
        
        start_date = c1.date_input(
            "Select a start date",
            # value=pd.to_datetime(d.date.max())-pd.DateOffset(months=3),
            value=pd.to_datetime(player_min_date),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(d.date.max())
            )
        end_date = c2.date_input(
            "Select an end date",
            # value=pd.to_datetime(d.date.max()),
            value=pd.to_datetime(player_max_date),
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
        minutes_played_select = c1.slider(f'Filter players that played less than x amount of minutes',0,minutes_played,100)
        
        num_cols_select = st.multiselect('Which stats should be used?',num_cols_base,num_cols_base)
        giddy_up = st.form_submit_button('Giddy Up')
        
        
        
        
        
        
    if giddy_up:
        df_filtered = d.loc[(d['date'] >= start_date) & (d['date'] <= end_date)].copy()
        # df_agg = aggregate_box_scores(df_filtered)
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


        df_agg = rename_columns(df_agg)

        num_cols, non_num_cols = get_num_cols(df_agg)
        

        # num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
        # non_num_cols_select=st.multiselect('Select non numeric columns for hover data',non_num_cols,['Player'])
        # list_one=['Cluster']
        # list_two=df_agg.columns.tolist()
        # color_options=list_one+list_two
        df_clstr = df_agg.loc[(df_agg['min'] >= minutes_played_select)].fillna(0)

        if player in df_clstr['player'].unique().tolist():
            non_num_cols = ['Player','Team','league','pos']
            quick_clstr_util(df_agg.fillna(0), 
                            num_cols_select, 
                            non_num_cols, 
                            'Cluster', 
                            player
                    )

        else:
            st.write('Adjust your filters because you have filtered out your player')



    


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




def app():

    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
        )
    

    st.title("NBA Player Stats Visualization")

    df = get_data()




    method = st.selectbox('Select a method:', ['Single Game', 'Date Range', 'Player Comparison'])
    


    if method == 'Single Game':
        single_game_viz(df)
    elif method == 'Date Range':
        date_range_viz(df)
    elif method == 'Player Comparison':
        player_comparison_viz(df)










if __name__ == "__main__":
    #execute
    app()