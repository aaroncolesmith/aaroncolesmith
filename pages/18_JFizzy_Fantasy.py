from datetime import datetime, date
import streamlit as st
import plotly_express as px
import pandas as pd
import plotly.io as pio
import io
import requests


pio.templates.default = "simple_white"







def app():
    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
    )
    


    st.title('J Fizzy Fantasy Playoffs')

    # Fetching game data
    url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?limit=1000&dates=2025&type=post-season'
    r= requests.get(url)
    df=pd.json_normalize(r.json()['events'])
    df =df.loc[df['season.slug']=='post-season'][['id','date','name','shortName','season.slug','status.type.description']].sort_values('date',ascending=True)

    # Function to extract stats and create a row for each athlete
    def extract_stats(player, team_name):
        rows = []
        for stat_category in player['statistics']:
            for athlete in stat_category['athletes']:
                athlete_name = athlete['athlete']['displayName']
                if stat_category['name'] == 'passing':
                    stats = athlete['stats']
                    rows.append([athlete_name, team_name] + stats[:8] + [None] * 11)  # Fill remaining columns
                elif stat_category['name'] == 'rushing':
                    stats = athlete['stats']
                    rows.append([athlete_name, team_name] + [None] * 8 + stats[:5] + [None] * 6)
                elif stat_category['name'] == 'receiving':
                    stats = athlete['stats']
                    rows.append([athlete_name, team_name] + [None] * 13 + stats[:5])
        return rows

    # Collecting game statistics
    d = pd.DataFrame()

    for game_id in df['id']:
        try:
            r = requests.get(f'https://cdn.espn.com/core/nfl/boxscore?xhr=1&gameId={game_id}')
            data = r.json()['gamepackageJSON']['boxscore']
            all_rows = []
            for player in data['players']:
                team_name = player['team']['displayName']
                all_rows.extend(extract_stats(player, team_name))

            columns = ['athlete.displayName', 'team.displayName', 'C/ATT', 'YDS', 'AVG', 'TD', 'INT', 'SACKS', 'QBR', 'RTG', 'CAR', 'YDS', 'AVG', 'TD', 'LONG', 'REC', 'YDS', 'AVG', 'TD', 'LONG', 'TGTS']
            d1 = pd.DataFrame(all_rows, columns=columns)
            d1['game_id'] = game_id
            d = pd.concat([d, d1])
        except:
            pass

    df = pd.merge(df, d, left_on='id', right_on='game_id', how='left')

    # Cleaning and renaming columns
    df.columns = ['id', 'date', 'game', 'game_short', 'season_info', 'status', 'player', 'team', 'C/ATT', 'PASS_YDS', 'PASS_YDS_AVG', 'PASS_TD', 'INT', 'SACKS', 'QBR', 'RTG', 'CAR', 'RUSH_YDS', 'RUSH_YDS_AVG', 'RUSH_TD', 'RUSH_LONG', 'REC', 'REC_YDS', 'REC_YDS_AVG', 'REC_TD', 'REC_LONG', 'TGTS', 'game_id']
    df = df.fillna(0)
    df['player'] = df['player'].str.strip()
    del df['game_id']


    # Convert the columns to numeric before aggregation
    for col in ['PASS_YDS', 'PASS_YDS_AVG', 'PASS_TD', 'INT', 'SACKS','CAR', 'RUSH_YDS', 'RUSH_YDS_AVG', 'RUSH_TD', 'RUSH_LONG', 'REC', 'REC_YDS', 'REC_YDS_AVG', 'REC_TD', 'REC_LONG', 'TGTS', 'C/ATT', 'PASS_YDS', 'PASS_YDS_AVG', 'PASS_TD', 'INT', 'SACKS', 'QBR', 'RTG']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Now perform the groupby and aggregation
    df = df.groupby(['id', 'date', 'game', 'game_short', 'season_info', 'status', 'player','team']).agg(
        PASS_YDS=('PASS_YDS','sum'),
        PASS_TD=('PASS_TD','sum'),
        INT=('INT','sum'),
        CAR = ('CAR','sum'),
        RUSH_YDS = ('RUSH_YDS','sum'),
        RUSH_YDS_AVG = ('RUSH_YDS_AVG','sum'),
        RUSH_TD = ('RUSH_TD','sum'),
        RUSH_LONG = ('RUSH_LONG','sum'),
        REC = ('REC','sum'),
        REC_YDS = ('REC_YDS','sum'),
        REC_YDS_AVG = ('REC_YDS_AVG','sum'),
        REC_TD = ('REC_TD','sum'),
        REC_LONG = ('REC_LONG','sum'),
        TGTS = ('TGTS','sum'),
    ).reset_index()


    # Add the new two-point conversion columns
    df['TWO_PT_PASS_TD'] = 0
    df['TWO_PT_RUSH_TD'] = 0
    df['TWO_PT_REC_TD'] = 0

    # Manually update two-point conversion data for specific players
    df.loc[(df['game'] == 'Denver Broncos at Buffalo Bills') & (df['player'] == 'Josh Allen'), 'TWO_PT_PASS_TD'] = 1
    df.loc[(df['game'] == 'Denver Broncos at Buffalo Bills') & (df['player'] == 'Keon Coleman'), 'TWO_PT_REC_TD'] = 1

    df.loc[(df['game'] == 'Washington Commanders at Philadelphia Eagles') & (df['player'] == 'Jayden Daniels'), 'TWO_PT_PASS_TD'] = 1
    df.loc[(df['game'] == 'Washington Commanders at Philadelphia Eagles') & (df['player'] == 'Olamide Zaccheaus'), 'TWO_PT_REC_TD'] = 1


    df.loc[(df['game'] == 'Buffalo Bills at Kansas City Chiefs') & (df['player'] == 'Patrick Mahomes'), 'TWO_PT_PASS_TD'] = 1
    df.loc[(df['game'] == 'Buffalo Bills at Kansas City Chiefs') & (df['player'] == 'Justin Watson'), 'TWO_PT_REC_TD'] = 1

    # Example for calculating fantasy points
    df['fantasy_pts'] = (df['PASS_TD'].fillna(0).astype(float) * 4.0) + \
                        (df['RUSH_TD'].fillna(0).astype(float) * 6.0) + \
                        (df['REC_TD'].fillna(0).astype(float) * 6.0) + \
                        (df['REC'].fillna(0).astype(float) * 1.0) + \
                        (df['TWO_PT_PASS_TD'].fillna(0).astype(float) * 2.0) + \
                        (df['TWO_PT_RUSH_TD'].fillna(0).astype(float) * 2.0) + \
                        (df['TWO_PT_REC_TD'].fillna(0).astype(float) * 3.0)



    # The input data
    data = '''Position,Joel,Huber,Dave,Wells,Tim,Romer,Jordan,Julian,Paul,Aaron
    QB,Josh Allen,Jalen Hurts,Sam Darnold,Lamar Jackson,Justin Herbert,Baker Mayfield,C.J. Stroud,Jared Goff,Jayden Daniels,Patrick Mahomes
    WR,Cooper Kupp,Zay Flowers,Amon-Ra St. Brown,Terry McLaurin,Justin Jefferson,Puka Nacua,A.J. Brown,DeVonta Smith,Khalil Shakir,Jordan Addison
    WR,Amari Cooper,Jalen McMillan,Rashod Bateman,Hollywood Brown,Ladd McConkey,Mike Evans,Jameson Williams,Xavier Worthy,Jayden Reed,Keon Coleman
    RB,Kyren Williams,J.K. Dobbins,Jaylen Warren,Joe Mixon,Saquon Barkley,David Montgomery,Derrick Henry,Brian Robinson Jr.,Jahmyr Gibbs,James Cook
    RB,Justice Hill,Bucky Irving,Ray Davis,Jaleel McLaughlin,Kareem Hunt,Isiah Pacheco,Rachaad White,Gus Edwards,Najee Harris,Aaron Jones
    TE,Dalton Kincaid,Noah Gray,Travis Kelce,Will Dissly,Dawson Knox,Isaiah Likely,Dallas Goedert,Sam LaPorta,T.J. Hockenson,Tucker Kraft
    FLEX,Sterling Shepard,Joshua Palmer,Mark Andrews,Courtland Sutton,DeAndre Hopkins,Dalton Schultz,Nico Collins,Zach Ertz,Marvin Mims Jr.,Josh Jacobs
    FLEX,Demarcus Robinson,Olamide Zaccheaus,Tim Patrick,George Pickens,Nelson Agholor,Jahan Dotson,Quentin Johnston,Romeo Doubs,Cade Otton,Ty Johnson
    '''

    # Read the data into a DataFrame
    df_players = pd.read_csv(io.StringIO(data),header=0)

    # Melt the dataframe so that each player is a row with the associated Team and Position
    df_melted = pd.melt(df_players, id_vars='Position', var_name='Team', value_name='Player')
    df_melted.columns = ['Position','Fantasy_Team','player']

    df_merged = pd.merge(df,df_melted,left_on=['player'],right_on=['player'],how='left')

    df_merged['Fantasy_Team'] = df_merged['Fantasy_Team'].fillna('Not Drafted')

    remaining_teams = ['Kansas City Chiefs','Philadelphia Eagles']
    df_merged.loc[df_merged.team.isin(remaining_teams), 'still_playing'] = 1
    df_merged['still_playing'] = df_merged['still_playing'].fillna(0)

    df_team_player_agg = df_merged.groupby(['player','Fantasy_Team','still_playing']).agg(fantasy_points=('fantasy_pts','sum'),
                                                              games_played=('date','nunique')).sort_values('fantasy_points',ascending=False).reset_index()
    df_team_player_agg['points_per_game'] = df_team_player_agg['fantasy_points'] / df_team_player_agg['games_played']
    df_team_player_agg['remaining_points'] = df_team_player_agg['points_per_game'] * df_team_player_agg['still_playing']
    df_team_player_agg['projected_total'] = df_team_player_agg['fantasy_points'] + df_team_player_agg['remaining_points']



    df_team_agg = df_team_player_agg.loc[df_team_player_agg['Fantasy_Team']!='Not Drafted'].groupby(['Fantasy_Team']).agg(
        total_fantasy_pts=('fantasy_points','sum'),
        players_played=('player','nunique'),
        still_playing=('still_playing','sum'),
        projected_total=('projected_total','sum')
        ).sort_values('total_fantasy_pts',ascending=False).reset_index()




    fig = px.bar(df_team_agg,
        y='Fantasy_Team',
        x='total_fantasy_pts',
        hover_data=['players_played'],
    template = 'simple_white',
        text_auto=True,
        orientation = 'h')
    # fig.update_xaxes(tickformat = ',.1%')
    fig.update_traces(marker=dict(
        color='lightblue',        # Fill color of the bars
        line=dict(color='navy', width=2)  # Outline color and thickness
    ))
    fig.update_layout(
            font=dict(
            family='Futura',  # Set font to Futura
            size=12,          # You can adjust the font size if needed
            color='black'     # Font color
        ),
        yaxis=dict(categoryorder='total ascending'),  # Order the categories by total value,
        # width=1200,
        title='Fantasy Points by Team'
    )

    st.markdown('### Leader Board')
    tab1, tab2, tab3 = st.tabs(["Fantasy Pts by Team", "Games Remaining", "Projected Total"])
    with tab1:
        st.plotly_chart(fig)

    fig = px.scatter(df_team_agg,
        y='still_playing',
        x='total_fantasy_pts',
        hover_data=['players_played'],
        template = 'simple_white',
        text='Fantasy_Team',
        # text_auto=True,
        # color='Fantasy_Team'
        # orientation = 'h'
        )
    
    # fig.update_xaxes(tickformat = ',.1%')
    fig.update_traces(marker=dict(
        size=18,
        color='lightblue',        # Fill color of the bars
        line=dict(color='navy', width=2)  # Outline color and thickness
    ),
    textposition='bottom center',)
    fig.update_layout(
            font=dict(
            family='Futura',  # Set font to Futura
            size=12,          # You can adjust the font size if needed
            color='black'     # Font color
        ),
        yaxis=dict(categoryorder='total ascending'),  # Order the categories by total value,
        height=800,
        width=800,
        title='Fantasy Points by Team'
    )
    with tab2:
        c1,c2,c3 = st.columns([1,4,1])
        c2.plotly_chart(fig)

    
    fig = px.bar(df_team_agg,
        y='Fantasy_Team',
        x='projected_total',
        hover_data=['players_played'],
    template = 'simple_white',
        text_auto=True,
        orientation = 'h')
    # fig.update_xaxes(tickformat = ',.1%')
    fig.update_traces(marker=dict(
        color='lightblue',        # Fill color of the bars
        line=dict(color='navy', width=2)  # Outline color and thickness
    ))
    fig.update_layout(
            font=dict(
            family='Futura',  # Set font to Futura
            size=12,          # You can adjust the font size if needed
            color='black'     # Font color
        ),
        yaxis=dict(categoryorder='total ascending'),  # Order the categories by total value,
        # width=1200,
        title='Projected Fantasy Points by Team'
    )
    with tab3:
        st.plotly_chart(fig)

# marker=dict(size=12, opacity=1, 
#                     line=dict(width=2, 
#                             color="DarkSlateGrey"
#                             )


    c1,c2 = st.columns(2)
    c1.markdown('### Top Players - Total Playoffs')
    c1.dataframe(df_team_player_agg[['player','Fantasy_Team','fantasy_points','games_played','remaining_points','projected_total']],hide_index=True)
    c2.markdown('### Top Players - By Game')
    c2.dataframe(df_merged[['date','game_short','player','fantasy_pts','Fantasy_Team']].sort_values('fantasy_pts',ascending=False),hide_index=True)


    df_team = pd.merge(df_melted,df.groupby(['player','team']).agg(fantasy_pts=('fantasy_pts','sum'),
                                                            games_played=('id','nunique')),how='left',left_on=['player'],right_on=['player'])
    
    st.markdown('### Score by Team')
    st.dataframe(df_team,hide_index=True)

    with st.expander('Raw Data'):
        st.write(df_merged)



if __name__ == "__main__":
    #execute
    app()