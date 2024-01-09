import streamlit as st
import os
import pandas as pd
from pytz import timezone
from datetime import datetime, timedelta
import requests
from fuzzywuzzy import fuzz
import h2o
import plotly_express as px
from st_files_connection import FilesConnection
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pickle
from elosports.elo import Elo

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog',
    layout='wide'
    )


four_feat_list=['spread_away',
 'spread_home',
 'spread_away_min',
 'spread_away_max',
 'spread_away_std',
 'spread_home_min',
 'spread_home_max',
 'spread_home_std',
 'spread_home_public',
 'spread_away_public',
 'num_bets',
 'trank_spread',
 'favorite_spread_bovada',
 'updated_spread_diff',
 'neutral_site',
 'home_fav',
 'away_fav',
 'home_dog',
 'away_dog',
 'spread_diff_less_than_8',
 'spread_diff_8_to_16',
 'spread_diff_16_to_28',
 'spread_diff_28_to_58',
 'spread_diff_58_plus',
 'bets_less_than_1900',
 'bets_1900_to_3500',
 'bets_3500_to_6900',
 'spread_6900_to_14k',
 'spread_14k_plus',
 'ttq_less_than_34',
 'ttq_34_to_44',
 'ttq_44_to_56',
 'ttq_56_to_75',
 'ttq_75_plus',
 'month_1',
 'month_2',
 'month_3',
 'month_4',
 'month_11',
 'month_12',
 'home_team_rating',
 'away_team_rating',
 'home_elo',
 'away_elo']



def req_to_df(r):
  try:
    games_df=pd.json_normalize(r.json()['games'],
                      )[['id','status','start_time','away_team_id','home_team_id','winning_team_id','league_name','season','attendance',
                      'last_play.home_win_pct','last_play.over_win_pct',
                      'boxscore.total_away_points','boxscore.total_home_points','boxscore.total_away_firsthalf_points','boxscore.total_home_firsthalf_points',
                      'boxscore.total_away_secondhalf_points','boxscore.total_home_secondhalf_points','broadcast.network']]
  except:
    try:
      games_df=pd.json_normalize(r.json()['games'],
                  )[['id','status','start_time','away_team_id','home_team_id','winning_team_id','league_name','season','attendance',
                  'boxscore.total_away_points','boxscore.total_home_points','boxscore.total_away_firsthalf_points','boxscore.total_home_firsthalf_points',
                  'boxscore.total_away_secondhalf_points','boxscore.total_home_secondhalf_points']]
    except:
      games_df=pd.json_normalize(r.json()['games'],
                  )[['id','status','start_time','away_team_id','home_team_id','winning_team_id','league_name','season','attendance',
                  ]]


  odds_df=pd.DataFrame()
  for i in range(pd.json_normalize(r.json()['games']).index.size):
    try:
      odds_df=pd.concat([odds_df,
      pd.json_normalize(r.json()['games'][i],
                    'odds',
                    ['id'],
                    meta_prefix='game_',
                    record_prefix='',
                    errors='ignore'
                    )[[ 'game_id',
                      'ml_away', 'ml_home', 'spread_away', 'spread_home', 'spread_away_line','spread_home_line', 'over', 'under', 'draw', 'total',
                          'away_total','away_over', 'away_under', 'home_total', 'home_over', 'home_under',
        'ml_home_public', 'ml_away_public', 'spread_home_public',
        'spread_away_public', 'total_under_public', 'total_over_public',
        'ml_home_money', 'ml_away_money', 'spread_home_money',
        'spread_away_money', 'total_over_money', 'total_under_money',
        'num_bets', 'book_id','type','inserted'
                          ]]
    ]
                      ).reset_index(drop=True)
    except:
      pass

  teams_df=pd.DataFrame()
  for i in range(pd.json_normalize(r.json()['games']).index.size):
    teams_df=pd.concat([teams_df,
                        pd.json_normalize(r.json()['games'][i],
                    'teams',
                    ['id'],
                    meta_prefix='game_',
                    record_prefix='team_'
                    )
                        ]
                      ).reset_index(drop=True)

  df=pd.merge(
  pd.merge(
  pd.merge(games_df,
          odds_df.query('book_id == 15'),
          left_on='id',
          right_on='game_id'),
          teams_df[['team_id','team_full_name']].rename(columns={'team_id':'home_team_id', 'team_full_name':'home_team'})
  ),
  teams_df[['team_id','team_full_name']].rename(columns={'team_id':'away_team_id', 'team_full_name':'away_team'})

  )

  df['date_scraped'] = datetime.now()


  return df,teams_df

def matchup_similarity(matchup1, matchup2):
    words1 = set(matchup1.lower().split())
    words2 = set(matchup2.lower().split())

    # Calculate Jaccard similarity between the two sets of words
    similarity_score = len(words1.intersection(words2)) / len(words1.union(words2))

    return similarity_score

def find_best_match(row, choices):
    scores = [matchup_similarity(row['matchup'], choice) for choice in choices]
    best_match_index = scores.index(max(scores))

    return choices[best_match_index], scores[best_match_index]

def trank_compare(df, date):

    d=df.loc[df.date==pd.to_datetime(date).date()].reset_index(drop=True).copy()

    if d.index.size>0:

      url_date=pd.to_datetime(date).strftime('%Y%m%d')
      url=f'https://barttorvik.com/schedule.php?date={url_date}&conlimit='
      r=requests.get(url)
      try:
        trank=pd.read_html(r.content)[0]
        trank.columns = [x.lower().replace('(','').replace(')','').replace(' ','_').replace('-','_') for x in trank.columns]
        for x in trank.columns:
          if 'unnamed' in x:
            del trank[x]
        trank['date'] = pd.to_datetime(date)

        replacements = ['\d+', 'BIG12|ESPN+','Peacock', 'ESPNU', 'ESPN', 'FS', '+', 'ACCN', 'BIG|', 'CBSSN', 'PAC','truTV','CBS','BE-T','Ivy-T','NCAA-T','FOX','WCC-T','Amer-T U','CAA-T','MWC-T','NEC-T','TBS','CUSA-T','BW-T','SECN','SEC-T','P-T','MAC-T','B-T','BTN','MAAC-T','ACC-T']
        for replacement in replacements:
            trank['matchup'] = trank['matchup'].str.replace(replacement, '', regex=True)
        trank['matchup'] = trank['matchup'].str.replace('Illinois Chicago', 'UIC', regex=True)
        trank['matchup'] = trank['matchup'].str.replace('Gardner Webb', 'Gardner-Webb', regex=True)
        # trank['result'] = trank['result'].str.replace('🎯', '', regex=True)







        # trank['matchup'] = trank['matchup'].str.replace('\d+', '',regex=True).str.replace('Peacock','',regex=True).str.replace('ESPNU','',regex=True).str.replace('ESPN','',regex=True).str.replace('FS','',regex=True).str.replace('+','',regex=True).str.replace('ACCN','',regex=True).str.replace('BIG|','',regex=True).str.replace('CBSSN','',regex=True).str.replace('PAC','',regex=True)
        trank['trank_spread']='-'+trank.t_rank_line.str.split('-',expand=True)[1].str.split(',',expand=True)[0]

        # Assuming df_cbb_test and trank are your two dataframes

        # Create a new column in df_cbb_test with the best matching values and scores from trank
        d[['best_match', 'similarity_score']] = d.apply(lambda row: pd.Series(find_best_match(row, trank['matchup'].tolist())), axis=1)

        # Merge the dataframes based on the 'best_match' column
        merged_df = pd.merge(d, trank, left_on='best_match', right_on='matchup', how='left', suffixes=('','_trank'))

        # Drop the 'best_match' column if you no longer need it
        merged_df = merged_df.drop(['best_match'], axis=1)

        merged_df['favorite_spread_bovada']=merged_df[['spread_away','spread_home']].min(axis=1)
        merged_df['favorite_spread_bovada'] = pd.to_numeric(merged_df['favorite_spread_bovada'])
        merged_df['trank_spread'] = pd.to_numeric(merged_df['trank_spread'])
        merged_df['spread_diff'] = abs(merged_df['favorite_spread_bovada'] - merged_df['trank_spread'])

        merged_df['spread_diff']=pd.to_numeric(merged_df['spread_diff'])
        merged_df['ttq']=pd.to_numeric(merged_df['ttq'])
        merged_df['similarity_score']=pd.to_numeric(merged_df['similarity_score'])

        # If bovada line > trank like, bet favorite; otherwise bet the dog
        merged_df.loc[merged_df.favorite_spread_bovada > merged_df.trank_spread, 'bet_advice'] = 'bet_favorite'
        merged_df.loc[merged_df.favorite_spread_bovada < merged_df.trank_spread, 'bet_advice'] = 'bet_dog'

        merged_df.loc[(merged_df.spread_away<0)&(merged_df.bet_result=='away_wins'), 'fav_result'] = 'fav_wins'
        merged_df.loc[(merged_df.spread_home<0)&(merged_df.bet_result=='home_wins'), 'fav_result'] = 'fav_wins'

        merged_df.loc[(merged_df.spread_away>0)&(merged_df.bet_result=='away_wins'), 'fav_result'] = 'dog_wins'
        merged_df.loc[(merged_df.spread_home>0)&(merged_df.bet_result=='home_wins'), 'fav_result'] = 'dog_wins'

        merged_df.loc[(merged_df.bet_advice=='bet_favorite')&(merged_df.fav_result=='fav_wins'), 'bet_advice_result'] = 'win'
        merged_df.loc[(merged_df.bet_advice=='bet_dog')&(merged_df.fav_result=='dog_wins'), 'bet_advice_result'] = 'win'

        merged_df.loc[(merged_df.bet_advice=='bet_favorite')&(merged_df.fav_result=='dog_wins'), 'bet_advice_result'] = 'loss'
        merged_df.loc[(merged_df.bet_advice=='bet_dog')&(merged_df.fav_result=='fav_wins'), 'bet_advice_result'] = 'loss'


        return merged_df
      except:
        pass

@st.cache_data(ttl=300)
def add_trank_data(df_cbb):
    df=df_cbb.groupby(['id','date','start_time_et','matchup','home_team_id','home_team','away_team_id','away_team']).agg(
        status=('status','last'),
        spread_away=('spread_away','last'),
        spread_home=('spread_home','last'),
        spread_away_min=('spread_away','min'),
        spread_away_max=('spread_away','max'),
        spread_away_std=('spread_away','std'),
        spread_home_min=('spread_home','min'),
        spread_home_max=('spread_home','max'),
        spread_home_std=('spread_home','std'),
        score_away=('boxscore_total_away_points','last'),
        score_home=('boxscore_total_home_points','last'),
        attendance=('attendance','max'),
        spread_home_public=('spread_home_public','max'),
        spread_away_public=('spread_away_public','max'),
        num_bets=('num_bets','max'),
        rec_count=('inserted','size')
        ).reset_index()
    df['spread_away_result'] = df['score_away'] - df['score_home']
    df['spread_home_result'] = df['score_home'] - df['score_away']
    df.loc[(df.spread_home_result+df.spread_home)>0,'bet_result'] = 'home_wins'
    df.loc[(df.spread_home_result+df.spread_home)<0,'bet_result'] = 'away_wins'
    df.loc[(df.spread_home_result+df.spread_home)==0,'bet_result'] = 'push'

    start_date = df_cbb.date.min()
    # start_date=datetime.now().date() - timedelta(2)
    end_date = df_cbb.date.max()

    current_date = start_date

    all_df=pd.DataFrame()

    while current_date <= end_date:
        res_df=trank_compare(df, current_date)
        all_df=pd.concat([all_df,res_df])
        current_date += timedelta(days=1)

    all_df = all_df.reset_index(drop=True)

    all_df['trank_favorite'] = all_df['t_rank_line'].str.split('-',expand=True)[0]
    all_df.loc[all_df.spread_away < 0,'bovada_favorite'] = all_df['away_team']
    all_df.loc[all_df.spread_home < 0,'bovada_favorite'] = all_df['home_team']
    all_df['fav_similarity'] = all_df.apply(lambda row: fuzz.ratio(str(row['trank_favorite']), str(row['bovada_favorite'])) if not (pd.isna(row['trank_favorite']) or pd.isna(row['bovada_favorite'])) else 0, axis=1)

    all_df.loc[all_df.fav_similarity >=37,'updated_spread_diff'] = abs(all_df['favorite_spread_bovada'] - all_df['trank_spread'])
    all_df.loc[all_df.fav_similarity <=37,'updated_spread_diff'] = abs(-all_df['favorite_spread_bovada'] - all_df['trank_spread'])

    df = all_df.sort_values('start_time_et',ascending=False)

    df.loc[df.spread_home<0,'home_fav'] = 1
    df['home_fav']=df['home_fav'].fillna(0)

    df.loc[df.matchup_trank.str.contains('vs'),'neutral_site'] = 1
    df['neutral_site']=df['neutral_site'].fillna(0)

    df['date'] = pd.to_datetime(df['date'])  # Convert the 'date' column to datetime format
    df = pd.concat([df, pd.get_dummies(df['date'].dt.month, prefix='month')], axis=1)


    df.loc[df.fav_result == 'fav_wins','fav_wins'] = 1
    df.loc[df.fav_result == 'dog_wins','fav_wins'] = 0

    return df

def normalize_bet_team_names(d, field):
    substitutions = {'State': 'St.',
                      'Kansas City Roos': 'UMKC',
                      'Long Island University Sharks': 'LIU Brooklyn',
                    'Omaha Mavericks': 'Nebraska Omaha',
                    'North Carolina-Wilmington Seahawks': 'UNC Wilmington',
                    'Virginia Military Institute Keydets': 'VMI',
                    'Southern Methodist Mustangs': 'SMU',
                    'Virginia Commonwealth Rams':'VCU',
                    'Florida International Golden Panthers':'FIU',
                    'N.J.I.T. Highlanders':'NJIT',
                    'Ole Miss':'Mississippi',
                    'St. Peter\'s Peacocks':'Saint Peter\'s',
                    'Texas-Arlington Mavericks':'UT Arlington',
                    'Miami (FL) Hurricanes':'Miami FL',
                    'Pennsylvania Quakers':'Penn',
                    'Louisiana-Monroe Warhawks':'Louisiana Monroe',
                    'Texas A&M-CC Islanders':'Texas A&M Corpus Chris',
                    'IPFW Mastodons':'Fort Wayne',
                    'Missouri KC':'UMKC',
                    'Chaminade Silverswords':'Chaminade',
                    'Florida International':'FIU',
                    'UConn Huskies':'Connecticut',
                    'UMass Minutemen':'Massachusetts',
                    'Massachusetts Lowell River Hawks':'UMass Lowell',
                    'SIU-Edwardsville':'SIU Edwardsville',
                    'Miami (FL) Hurricanes':'Miami FL'
                    }

    for key, value in substitutions.items():
        d[field] = d[field].str.replace(key, value, regex=True)

    replacements = ['Musketeers','Longhorns','Wildcats','Panthers','Tigers','Warriors','Skyhawks','Sharks','Flames','Bulldogs','Cougars','Runnin\'','Rebels',
                    'Spartans','Razorbacks','Bears','Raiders','Cardinal','Buckeyes','Hawkeyes','Bobcats','Rockets',
                    'Gauchos']
    for replacement in replacements:
        d[field] = d[field].str.replace(replacement, '', regex=True)
    return d

@st.cache_data(ttl=300)
def get_df_cbb():
    df_cbb=pd.DataFrame()

    start_date = datetime.today().date() - timedelta(days=1)
    end_date = datetime.today().date() + timedelta(days=2)

    current_date = start_date
    fail_list=[]

    while current_date <= end_date:
        date_str=current_date.strftime('%Y%m%d')
        current_date += timedelta(days=1)
        url=f'https://api.actionnetwork.com/web/v1/scoreboard/ncaab?period=game&bookIds=15,30,76,75,123,69,68,972,71,247,79&division=D1&date={date_str}&tournament=0'

        # # generate a random sleep time between 1 and 6 seconds
        # sleep_time = random.randint(1, 6)

        # # sleep for the randomly generated time
        # time.sleep(sleep_time)

        r=requests.get(url,headers=headers)

        # try:
        if len(r.json()['games']) > 0:
            df_tmp,teams_df_tmp=req_to_df(r)
            df_cbb=pd.concat([df_cbb,df_tmp]).reset_index(drop=True)
            # teams_df=pd.concat([teams_df,teams_df_tmp]).reset_index(drop=True)
        else:
            print('no games for date')
        # except:
        #     print(date_str + ' failed')
        #     fail_list.append(date_str)



    df_cbb = df_cbb.merge(df_cbb.groupby(['id']).agg(updated_start_time=('start_time','last')).reset_index(), on='id', how='left')
    del df_cbb['start_time']

    df_cbb=df_cbb.rename(columns={'updated_start_time':'start_time'})


    df_cbb.columns = [x.lower() for x in df_cbb.columns]
    df_cbb.columns = [x.replace('.','_') for x in df_cbb.columns]
    df_cbb=df_cbb.sort_values(['start_time','date_scraped'],ascending=[True,True]).reset_index(drop=True)

    df_cbb = normalize_bet_team_names(df_cbb, 'home_team')
    df_cbb = normalize_bet_team_names(df_cbb, 'away_team')

    df_cbb['matchup'] = df_cbb['away_team'] + ' at ' + df_cbb['home_team']
    df_cbb['start_time_pt'] = pd.to_datetime(df_cbb['start_time']).dt.tz_convert('US/Pacific')
    df_cbb['start_time_et'] = pd.to_datetime(df_cbb['start_time']).dt.tz_convert('US/Eastern')
    df_cbb['date'] = df_cbb['start_time_pt'].dt.date

    return df_cbb

@st.cache_data(ttl=300)
def run_the_model(df):

    feat_list=[
    'trank_spread',
    'favorite_spread_bovada',
    'updated_spread_diff',
    'spread_away',
    'spread_home',
    'spread_away_min',
    'spread_away_max',
    'spread_away_std',
    'spread_home_min',
    'spread_home_max',
    'spread_home_std',
    #  'attendance',
    'spread_home_public',
    'spread_away_public',
    'num_bets',
    'ttq',
    'home_fav', 'neutral_site',
        # 'month_1', 'month_2', 'month_3', 'month_4', 'month_11', 'month_12',
    #  'home_team_rating',
    #  'away_team_rating'
    ]


    df = df.sort_values('start_time_et',ascending=False)

    # df.loc[df.spread_home<0,'home_fav'] = 1
    # df['home_fav']=df['home_fav'].fillna(0)

    # df.loc[df.matchup_trank.str.contains('vs'),'neutral_site'] = 1
    # df['neutral_site']=df['neutral_site'].fillna(0)

    # df['date'] = pd.to_datetime(df['date'])  # Convert the 'date' column to datetime format
    # df = pd.concat([df, pd.get_dummies(df['date'].dt.month, prefix='month')], axis=1)

    h2o.init()
    # model_path='/content/drive/MyDrive/Analytics/StackedEnsemble_AllModels_1_AutoML_1_20231129_221304'
    model_path='models/StackedEnsemble_AllModels_1_AutoML_1_20231129_221304'

    saved_model = h2o.load_model(model_path)

    fill_na_feats=['spread_away', 'spread_home', 'spread_away_min', 'spread_away_max','spread_away_std', 'spread_home_min', 'spread_home_max','spread_home_std','attendance','spread_home_public', 'spread_away_public', 'num_bets']


    for x in fill_na_feats:
        df[x] = df[x].fillna(0)


    h2o_d = h2o.H2OFrame(df[feat_list])

    # Use the predict method
    predictions = saved_model.predict(h2o_d)

    # Convert H2OFrame back to pandas DataFrame if needed
    predictions_df = h2o.as_list(predictions)

    # Add the predictions to your original DataFrame
    df['fav_wins_pred'] = predictions_df['predict'].values



    df.loc[df.fav_wins_pred>=.5,'fav_wins_bin'] = 1
    df.loc[df.fav_wins_pred<.5,'fav_wins_bin'] = 0
    df['confidence_level'] = abs(df['fav_wins_pred']-.5)

    df.loc[df.fav_result == 'fav_wins','fav_wins'] = 1
    df.loc[df.fav_result == 'dog_wins','fav_wins'] = 0

    df['fav_wins'] = pd.to_numeric(df['fav_wins'])
    df['fav_wins_bin'] = pd.to_numeric(df['fav_wins_bin'])

    # del df['ensemble_model_win']
    df.loc[df.fav_wins_bin == df.fav_wins, 'ensemble_model_win'] = 1
    # df.loc[((df.fav_wins_bin == 0 & df.fav_wins == 1) or (df.fav_wins_bin == 1 & df.fav_wins == 0)), 'ensemble_model_win'] = 0
    df.loc[((df['fav_wins_bin'] == 0) & (df['fav_wins'] == 1)) | ((df['fav_wins_bin'] == 1) & (df['fav_wins'] == 0)), 'ensemble_model_win'] = 0

    # del df['ensemble_mode_advice']
    df.loc[((df.spread_home<0)&(df.fav_wins_bin==1)),
    'ensemble_model_advice'] = df['home_team'] + ' (' + df['spread_home'].astype(str) + ')'

    df.loc[((df.spread_home>0)&(df.fav_wins_bin==0)),
    'ensemble_model_advice'] = df['home_team'] + ' (+' + df['spread_home'].astype(str) + ')'

    df.loc[((df.spread_away<0)&(df.fav_wins_bin==1)),
    'ensemble_model_advice'] = df['away_team'] + ' (' + df['spread_away'].astype(str) + ')'

    df.loc[((df.spread_away>0)&(df.fav_wins_bin==0)),
    'ensemble_model_advice'] = df['away_team'] + ' (+' + df['spread_away'].astype(str) + ')'

    # df['home_team'] + ' (' + df['spread_home'].astype(str) + ')'


    df.groupby(['fav_wins','fav_wins_bin']).size().to_frame('cnt').sort_values('cnt',ascending=False).reset_index()

    df['fav_wins_str'] = df['fav_wins'].astype('str').fillna('tbd')

    df['model_run'] = pd.Timestamp.now(tz='US/Eastern')
    df['model'] = model_path.replace('/content/drive/MyDrive/Analytics/','')

    st.write('model has ran')

    return df

@st.cache_data(ttl=300)
def get_s3_data(filename):
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read(f"bet-model-data/{filename}.parquet", input_format="parquet", ttl=600)
    return df

def upload_s3_data(df, filename):

    for x in ['date','start_time_et','model_run']:
        try:
            df[x]=pd.to_datetime(df[x])
        except:
           df[x]=df[x].apply(lambda x: pd.to_datetime(x).tz_convert('US/Eastern'))

    for x in ['id','score_home','score_away','spread_home','spread_away','spread_home_public','spread_away_public','num_bets','ttq','updated_spread_diff','fav_wins_pred','fav_wins_bin','confidence_level','fav_wins','ensemble_model_win']:
        df[x] = df[x].replace('nan',np.nan).apply(lambda x: pd.to_numeric(x))

    # for x in ['id','date','time_to_tip','model_run','start_time_et','matchup','spread_home','spread_away','spread_home_public','spread_away_public','num_bets','ttq','updated_spread_diff','fav_wins_pred','fav_wins_bin','confidence_level','fav_wins','ensemble_model_advice','ensemble_model_win','model']:
    #    df[x]=df[x].astype(str)

    # df = df.astype(str)
    # table = pa.Table.from_pandas(df[['id','date','status','time_to_tip','model_run','start_time_et','matchup','score_home','score_away','spread_home','spread_away','spread_home_public','spread_away_public','num_bets','ttq','updated_spread_diff','fav_wins_pred','fav_wins_bin','confidence_level','fav_wins','ensemble_model_advice','ensemble_model_win','model']])
    table = pa.Table.from_pandas(df)


    pq.write_table(table, f'./{filename}.parquet',compression='BROTLI',use_deprecated_int96_timestamps=True)

    session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.resource('s3')
    # Filename - File to upload
    # Bucket - Bucket to upload to (the top level directory under AWS S3)
    # Key - S3 object name (can contain subdirectories). If not specified then file_name is used
    s3.meta.client.upload_file(Filename=f'./{filename}.parquet', Bucket='bet-model-data', Key=f'{filename}.parquet')
    st.write('data uploaded')


def visualize_model_over_time(df, num):

    for x in ['date','start_time_et','model_run']:
        try:
           df[x]=pd.to_datetime(df[x])
        except:
           df[x]=df[x].apply(lambda x: pd.to_datetime(x).tz_convert('US/Eastern'))

    df=df.sort_values(['model_run','start_time_et'],ascending=[True,True]).reset_index(drop=True)

    # c1,c2,c3,c4 = st.columns(4)

    # tip_start = c1.number_input('How far back (mins since tip)',
    #                             min_value=df.loc[df.model_run==df.model_run.max()].time_to_tip.min(),
    #                             max_value=df.loc[df.model_run==df.model_run.max()].time_to_tip.max(),
    #                             # value=-90.0
    #                             )
    
    # tip_end = c2.number_input('How far forward (mins til tip)',
    #                             min_value=df.loc[df.model_run==df.model_run.max()].time_to_tip.min(),
    #                             max_value=df.loc[df.model_run==df.model_run.max()].time_to_tip.max(),
    #                             value=df.loc[df.model_run==df.model_run.max()].time_to_tip.max()
    #                             )
    
    # conf_threshold = c3.number_input('Confidence interval threshold',
    #                             # min_value=df.loc[df.model_run==df.model_run.max()].confidence_level.min(),
    #                             min_value=0.0,
    #                             max_value=df.loc[df.model_run==df.model_run.max()].confidence_level.max(),
    #                             value=.05,
    #                             step=.01
    #                             )
    
    # max_records = c4.number_input('Maximum matchups to return',
    #                             min_value=2,
    #                             max_value=25,
    #                             value=10,
    #                             step=1
    #                             )

    # tip_start=-90
    # tip_end=180
    # conf_threshold=.05
    # max_records=10

    matchups=df.sort_values(['time_to_tip','confidence_level'],ascending=[True,False]).matchup.unique()


    matchups = df.groupby(['matchup']).agg(confidence_level=('confidence_level','last')).sort_values('confidence_level',ascending=False).reset_index().head(num).matchup.tolist()

    # matchups=df.loc[
    #     (df.time_to_tip >= tip_start)
    #     &(df.model_run==df.model_run.max())
    #     &(df.time_to_tip <= tip_end)
    #     &(df.confidence_level >= conf_threshold)
    #     ].sort_values(['time_to_tip','confidence_level'],ascending=[True,False]).head(max_records).matchup.unique()


    # if len(matchups)<=1:
    #        matchups=df.loc[
    #             (df.time_to_tip > tip_start)
    #             &(df.model_run==df.model_run.max())
    #             # &(df.time_to_tip < tip_end)
    #             &(df.confidence_level > conf_threshold)
    #             ].sort_values(['time_to_tip','confidence_level'],ascending=[True,False]).head(max_records).matchup.unique()


    df.loc[(df.status=='inprogress')&(df.ensemble_model_win==1), 'bet_status'] = 'winning'
    df.loc[(df.status=='inprogress')&(df.ensemble_model_win==0.0), 'bet_status'] = 'losing'
    df.loc[(df.status=='complete')&(df.ensemble_model_win==1), 'bet_status'] = 'won'
    df.loc[(df.status=='complete')&(df.ensemble_model_win==0.0), 'bet_status'] = 'lost'
    df.loc[(df.status=='scheduled'), 'bet_status'] = 'tbd'

    df['score_home']=df['score_home'].fillna(0)
    df['score_away']=df['score_away'].fillna(0)

    fig=px.scatter(
        # save_df,
        df.loc[df.matchup.isin(matchups)],
            x='model_run',
            y='confidence_level',
                render_mode='svg',
                # height=800,
                color='matchup',
                template='simple_white',
                # width=1200,
        # color_discrete_sequence=['#FF6600','skyblue', 'red','yellow','blue'],
                category_orders={'matchup':df.loc[df.matchup.isin(matchups)].sort_values('matchup',ascending=True)['matchup'].tolist()},
            hover_data=['matchup','ensemble_model_advice','time_to_tip','status','score_home','score_away','bet_status','id']
                )
    # fig.update_layout(hovermode="x")
    fig.update_traces(
        mode='lines+markers',
        line_shape='spline',
        line=dict(width=4),
        marker=dict(size=12,opacity=.9,
            line=dict(width=1,color='DarkSlateGrey')
                )
    )

    fig.update_yaxes(title='Confidence Level',
                    tickformat = ',.0%')

    fig.update_xaxes(title='',
                    )
    
    fig.update_layout(
       legend_title_text='',
       legend=dict(orientation = "h",   # show entries horizontally
                     xanchor = "center",  # use center of legend as anchor
                     x = 0.5
    ))


    fig.update_traces(hovertemplate= "%{customdata[7]}: %{customdata[0]} <br>Tips in %{customdata[2]:.1f} mins<br>%{customdata[1]} -- Confidence: %{y:.1%}<br>%{customdata[3]} - %{customdata[4]} - %{customdata[5]} -- %{customdata[6]}" + "<extra></extra>")
    st.plotly_chart(fig,use_container_width=True)


def combine_data(df,save_df):
    df=pd.concat([df,save_df]).sort_values(['id','model_run'],ascending=[True,True]).reset_index(drop=True)
    df['matchup'] = df['matchup'].str.replace('  ',' ')
    df['time_to_tip']=pd.to_numeric(df['time_to_tip'])
    df['confidence_level']=pd.to_numeric(df['confidence_level'])
    df['time_to_tip'] = df['time_to_tip'].round(0)
    df['confidence_level'] = df['confidence_level'].round(3)

    for x in ['date','start_time_et','model_run']:
        try:
            df[x]=pd.to_datetime(df[x])
        except:
           df[x]=df[x].apply(lambda x: pd.to_datetime(x).tz_convert('US/Eastern'))


    return df





##### v2 functions


@st.cache_data(ttl=300)
def get_cbb_trank_data(date):

  headers = {
      'Authority': 'api.actionnetwork',
      'Accept': 'application/json',
      'Origin': 'https://www.actionnetwork.com',
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
  }

  df_cbb=pd.DataFrame()
  date_str=date.strftime('%Y%m%d')
  url=f'https://api.actionnetwork.com/web/v1/scoreboard/ncaab?period=game&bookIds=15,30,76,75,123,69,68,972,71,247,79&division=D1&date={date_str}&tournament=0'
  r=requests.get(url,headers=headers)
  if len(r.json()['games']) > 0:
    df_tmp,teams_df_tmp=req_to_df(r)
    df_cbb=pd.concat([df_cbb,df_tmp]).reset_index(drop=True)
  else:
     print('no games for date')
     st.write('no games')
  with st.expander('debug'):
     st.write(df_cbb)
  df_cbb = df_cbb.merge(df_cbb.groupby(['id']).agg(updated_start_time=('start_time','last')).reset_index(), on='id', how='left')
  del df_cbb['start_time']

  df_cbb=df_cbb.rename(columns={'updated_start_time':'start_time'})

  df_cbb.columns = [x.lower() for x in df_cbb.columns]
  df_cbb.columns = [x.replace('.','_') for x in df_cbb.columns]
  df_cbb=df_cbb.sort_values(['start_time','date_scraped'],ascending=[True,True]).reset_index(drop=True)

  df_cbb = normalize_bet_team_names(df_cbb, 'home_team')
  df_cbb = normalize_bet_team_names(df_cbb, 'away_team')


  df_cbb['matchup'] = df_cbb['away_team'] + ' at ' + df_cbb['home_team']
  try:
    df_cbb['start_time_pt'] = pd.to_datetime(df_cbb['start_time']).dt.tz_convert('US/Pacific')
  except:
    #  stamp.tz_convert('US/Eastern')
    df_cbb['start_time_pt'] = pd.to_datetime(df_cbb['start_time']).dt.tz_localize('US/Pacific')
  try:
     df_cbb['start_time_et'] = pd.to_datetime(df_cbb['start_time']).dt.tz_convert('US/Eastern')
  except:
     df_cbb['start_time_et'] = pd.to_datetime(df_cbb['start_time']).dt.tz_localize('US/Eastern')
     

  df_cbb['date'] = df_cbb['start_time_pt'].dt.date


  if 'boxscore_total_away_points' not in df_cbb.columns:
     df_cbb['boxscore_total_away_points'] = np.nan
     df_cbb['boxscore_total_home_points'] = np.nan




  ### def add_trank_data
  df=df_cbb.groupby(['id','date','start_time_et','matchup','home_team_id','home_team','away_team_id','away_team']).agg(
      status=('status','last'),
      spread_away=('spread_away','last'),
      spread_home=('spread_home','last'),
      spread_away_min=('spread_away','min'),
      spread_away_max=('spread_away','max'),
      spread_away_std=('spread_away','std'),
      spread_home_min=('spread_home','min'),
      spread_home_max=('spread_home','max'),
      spread_home_std=('spread_home','std'),
      score_away=('boxscore_total_away_points','last'),
      score_home=('boxscore_total_home_points','last'),
      attendance=('attendance','max'),
      spread_home_public=('spread_home_public','max'),
      spread_away_public=('spread_away_public','max'),
      num_bets=('num_bets','max'),
      rec_count=('inserted','size')
      ).reset_index()
  df['spread_away_result'] = df['score_away'] - df['score_home']
  df['spread_home_result'] = df['score_home'] - df['score_away']

  df.loc[(df.spread_home_result+df.spread_home)>0,'bet_result'] = 'home_wins'
  df.loc[(df.spread_home_result+df.spread_home)<0,'bet_result'] = 'away_wins'
  df.loc[(df.spread_home_result+df.spread_home)==0,'bet_result'] = 'push'

  start_date = df_cbb.date.min()
  # start_date=datetime.now().date() - timedelta(2)
  end_date = df_cbb.date.max()

  current_date = start_date

  all_df=pd.DataFrame()

  while current_date <= end_date:
    print(current_date)
    res_df=trank_compare(df, current_date)
    all_df=pd.concat([all_df,res_df])
    print(all_df.index.size)
    print(all_df.date.max())
    current_date += timedelta(days=1)

  all_df = all_df.reset_index(drop=True)

  all_df['trank_favorite'] = all_df['t_rank_line'].str.split('-',expand=True)[0]
  all_df.loc[all_df.spread_away < 0,'bovada_favorite'] = all_df['away_team']
  all_df.loc[all_df.spread_home < 0,'bovada_favorite'] = all_df['home_team']
  all_df['fav_similarity'] = all_df.apply(lambda row: fuzz.ratio(str(row['trank_favorite']), str(row['bovada_favorite'])) if not (pd.isna(row['trank_favorite']) or pd.isna(row['bovada_favorite'])) else 0, axis=1)

  all_df.loc[all_df.fav_similarity >=37,'updated_spread_diff'] = abs(all_df['favorite_spread_bovada'] - all_df['trank_spread'])
  all_df.loc[all_df.fav_similarity <=37,'updated_spread_diff'] = abs(-all_df['favorite_spread_bovada'] - all_df['trank_spread'])

  df = all_df.sort_values('start_time_et',ascending=False)
  df['time_to_tip'] = (df['start_time_et']-pd.Timestamp.now(tz='US/Eastern')).dt.total_seconds() / 60

  return df


@st.cache_data(ttl=300)
def add_features(df):
  df.loc[df.spread_home<0,'home_fav'] = 1
  df['home_fav']=df['home_fav'].fillna(0)

  df.loc[df.spread_away<0,'away_fav'] = 1
  df['away_fav']=df['away_fav'].fillna(0)

  df.loc[df.spread_home>0,'home_dog'] = 1
  df['home_dog']=df['home_dog'].fillna(0)

  df.loc[df.spread_away>0,'away_dog'] = 1
  df['away_dog']=df['away_dog'].fillna(0)

  df.loc[df.matchup_trank.str.contains('vs'),'neutral_site'] = 1
  df['neutral_site']=df['neutral_site'].fillna(0)

  df['date'] = pd.to_datetime(df['date'])  # Convert the 'date' column to datetime format
  df = pd.concat([df, pd.get_dummies(df['date'].dt.month, prefix='month')], axis=1)


  for x in ['month_1','month_2','month_3','month_4','month_11','month_12']:
    try:
      df[x] = df[x].fillna(0)
    except:
      df[x] = 0

  df_ratings = get_ratings()

  df=pd.merge(df,df_ratings[['id','home_team_rating', 'away_team_rating', 'home_elo', 'away_elo']], left_on='id',right_on='id')

  bin_class_list=[]

  for x in ['num_bets','updated_spread_diff','ttq']:
    df[x]=pd.to_numeric(df[x].fillna(0))

  for l in [.1,.5,.75,.9]:
    df.loc[df.num_bets>df.num_bets.quantile(l), f'num_bets_class_{l}']=1
    df[f'num_bets_class_{l}']=df[f'num_bets_class_{l}'].fillna(0)
    bin_class_list.append(f'num_bets_class_{l}')

  for l in [.1,.5,.75,.9]:
    df.loc[df.updated_spread_diff>df.updated_spread_diff.quantile(l), f'updated_spread_diff{l}']=1
    df[f'updated_spread_diff{l}']=df[f'updated_spread_diff{l}'].fillna(0)
    bin_class_list.append(f'updated_spread_diff{l}')

  for l in [.1,.5,.75,.9]:
    df.loc[df.ttq>df.ttq.quantile(l), f'ttq{l}']=1
    df[f'ttq{l}']=df[f'ttq{l}'].fillna(0)
    bin_class_list.append(f'ttq{l}')



  with open('data/quantiles_dict.pkl', 'rb') as f:
    quantiles_dict = pickle.load(f)



  bin_class_list=[]
  for a in quantiles_dict.keys():
    print(a)
    lower=0
    for b in quantiles_dict[a].keys():
      upper_limit=df[a].quantile(b)
      print(f'upper- {upper_limit} lower - {lower}')
      df.loc[(df[a]<=upper_limit)&(df[a]>lower), f'{a}_{b}']=1
      df[f'{a}_{b}']=df[f'{a}_{b}'].fillna(0)
      bin_class_list.append(f'{a}_{b}')
      lower = df[a].quantile(b)




  spread_bins = [0, 0.8, 1.6, 2.8, 5.8, 10000]
  spread_labels = ['spread_diff_less_than_8', 'spread_diff_8_to_16', 'spread_diff_16_to_28', 'spread_diff_28_to_58', 'spread_diff_58_plus']
  df = create_category_columns(df, 'updated_spread_diff', spread_bins, spread_labels)

  bets_bins = [0, 1900, 3500, 6900, 14000, 10000000]
  bets_labels = ['bets_less_than_1900', 'bets_1900_to_3500', 'bets_3500_to_6900', 'spread_6900_to_14k', 'spread_14k_plus']
  df = create_category_columns(df, 'num_bets', bets_bins, bets_labels)

  bins = [0, 34, 44, 56, 75, 10000000]
  labels = ['ttq_less_than_34', 'ttq_34_to_44', 'ttq_44_to_56', 'ttq_56_to_75', 'ttq_75_plus']
  df = create_category_columns(df, 'ttq', bins, labels)




  return df


def create_category_columns(df, column_name, bins, labels):
    df[f'{column_name}_category'] = pd.cut(df[column_name], bins=bins, labels=labels, include_lowest=True)
    df = pd.concat([df, pd.get_dummies(df[f'{column_name}_category'])], axis=1)
    return df


@st.cache_data(ttl=300)
def run_model(df, model_path, model_run):
  
    fill_na_feats=['spread_away', 'spread_home', 'spread_away_min', 'spread_away_max','spread_away_std', 'spread_home_min',
                    'spread_home_max','spread_home_std','attendance','spread_home_public', 'spread_away_public', 'num_bets'
                    ]

    for x in fill_na_feats:
            df[x] = df[x].fillna(0)
    h2o.init()
    h2o_d = h2o.H2OFrame(df)
    saved_model = h2o.load_model(model_path)

    # Use the predict method
    predictions = saved_model.predict(h2o_d)

    # Convert H2OFrame back to pandas DataFrame if needed
    predictions_df = h2o.as_list(predictions)

    # Add the predictions to your original DataFrame
    df['fav_wins_pred'] = predictions_df['predict'].values

    df.loc[df.fav_wins_pred>=.5,'fav_wins_bin'] = 1
    df.loc[df.fav_wins_pred<.5,'fav_wins_bin'] = 0
    df['confidence_level'] = abs(df['fav_wins_pred']-.5)

    df.loc[df.fav_result == 'fav_wins','fav_wins'] = 1
    df.loc[df.fav_result == 'dog_wins','fav_wins'] = 0

    df['fav_wins'] = pd.to_numeric(df['fav_wins'])
    df['fav_wins_bin'] = pd.to_numeric(df['fav_wins_bin'])

    # del df['ensemble_model_win']
    df.loc[df.fav_wins_bin == df.fav_wins, 'ensemble_model_win'] = 1
    # df.loc[((df.fav_wins_bin == 0 & df.fav_wins == 1) or (df.fav_wins_bin == 1 & df.fav_wins == 0)), 'ensemble_model_win'] = 0
    df.loc[((df['fav_wins_bin'] == 0) & (df['fav_wins'] == 1)) | ((df['fav_wins_bin'] == 1) & (df['fav_wins'] == 0)), 'ensemble_model_win'] = 0

    # del df['ensemble_mode_advice']
    df.loc[((df.spread_home<0)&(df.fav_wins_bin==1)),
    'ensemble_model_advice'] = df['home_team'] + ' (' + df['spread_home'].astype(str) + ')'

    df.loc[((df.spread_home>0)&(df.fav_wins_bin==0)),
    'ensemble_model_advice'] = df['home_team'] + ' (+' + df['spread_home'].astype(str) + ')'

    df.loc[((df.spread_away<0)&(df.fav_wins_bin==1)),
    'ensemble_model_advice'] = df['away_team'] + ' (' + df['spread_away'].astype(str) + ')'

    df.loc[((df.spread_away>0)&(df.fav_wins_bin==0)),
    'ensemble_model_advice'] = df['away_team'] + ' (+' + df['spread_away'].astype(str) + ')'

    df['fav_wins_str'] = df['fav_wins'].astype('str').fillna('tbd')
    df['model_run'] = model_run
    df['model'] = model_path.replace('/content/drive/MyDrive/Analytics/','')


    return df

@st.cache_data(ttl=60*30)
def get_ratings():
  df=pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/bet_model/main/trank_compare_data.csv')
  df=df.sort_values('start_time_et',ascending=True)

  teams = pd.concat([df.rename(columns={'home_team':'team'})['team'],
            df.rename(columns={'away_team':'team'})['team']]).unique()
  team_ratings = {team: 100 for team in teams}

  home_team_rating=[]
  away_team_rating=[]

  for index, row in df.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    home_score = row['score_home']
    away_score = row['score_away']
    home_team_rating.append(team_ratings[row['home_team']])
    away_team_rating.append(team_ratings[row['away_team']])

    if home_score > away_score:
        team_ratings[home_team] += 0.02 * team_ratings[away_team]
        team_ratings[away_team] -= 0.02 * team_ratings[away_team]
    else:
        team_ratings[home_team] -= 0.02 * team_ratings[home_team]
        team_ratings[away_team] += 0.02 * team_ratings[home_team]


  df['home_team_rating'] = home_team_rating
  df['away_team_rating'] = away_team_rating



  elo_league 	= Elo(k = 20)

  for team in teams:
    elo_league.addPlayer(team)

  df['date']=pd.to_datetime(df['date'])

  for index, game in df.iterrows():
    if index > 0:
      if (game['date']-df.loc[index-1]['date']).days > 90:
        print((game['date']-df.loc[index-1]['date']).days)
        for key in elo_league.ratingDict.keys():
          elo_league.ratingDict[key] = elo_league.ratingDict[key] - ((elo_league.ratingDict[key] - 1500) * (1/3.))
    home_elo = elo_league.ratingDict[game['home_team']]
    away_elo = elo_league.ratingDict[game['away_team']]
    df.at[index, 'home_elo'] = home_elo
    df.at[index, 'away_elo'] = away_elo
    if game['score_home'] > game['score_away']:
      elo_league.gameOver(game['home_team'], game['away_team'], True)
    else:
      elo_league.gameOver(game['away_team'], game['home_team'], 0)


  return df.groupby(['id','home_team_id','home_team','away_team_id','away_team']).agg(
      home_team_rating=('home_team_rating','median'),
      away_team_rating=('away_team_rating','median'),
      home_elo=('home_elo','median'),
      away_elo=('away_elo','median')
      ).reset_index()


def app():
    st.markdown('# Welcome to the Secret Page')
    st.markdown('### Enter Password to Continue')

    with st.form(key='pass_input'):
        input_pass=st.text_input('What\'s the password?')
        password=os.environ["TEST_PASS"]
        submit_button=st.form_submit_button(label='Go!')

    if input_pass==password:
        
        st.write('correct')

        today=pd.Timestamp.now(tz='US/Eastern').date()


        with st.form(key='date_selector_form'):
            c1,c2=st.columns(2)
            date = c1.date_input(
            "Select a date / month for games",
                value=today,
                min_value=today - timedelta(days=10),
                max_value=today + timedelta(days=2)
            )

            confidence_threshold=c2.slider(
               'Set a confidence threshold (0 selects all games)',
               min_value=0.00,
               max_value=0.50
            )

            date_submit_button=st.form_submit_button(label='Go!')



        if date < today:
    
            df=get_cbb_trank_data(date)
            df=add_features(df)
            model_run=pd.Timestamp.now(tz='US/Eastern')
            model_path='models/StackedEnsemble_BestOfFamily_4_AutoML_1_20231222_01058'
            df1 = run_model(df, model_path,model_run)
        else:
        
            df=get_cbb_trank_data(date)
            df=add_features(df)
            model_run=pd.Timestamp.now(tz='US/Eastern')
            model_path='models/StackedEnsemble_BestOfFamily_4_AutoML_1_20231222_01058'
            df1 = run_model(df, model_path,model_run)


        filename='model_runs_streamlit_v2'
        hist_df = get_s3_data(filename)

        df1=pd.concat([hist_df,df1]).reset_index(drop=True)
        df1['score'] = df1['home_team'] +' (' + df1['score_home'].astype('str') + ') - ' + df1['away_team'] +' (' + df1['score_away'].astype('str')+ ')'
        df1.loc[df1.status!='complete','ensemble_model_win']=np.nan

        last_model_run=pd.to_datetime(hist_df.model_run).max()
        current_time=pd.Timestamp.now(tz='US/Eastern')

        last_model_run = last_model_run.tz_convert('US/Eastern')
        current_time = current_time.tz_convert('US/Eastern')

        time_since_last_model=(current_time-last_model_run).total_seconds() / 60

        if time_since_last_model > 15:
            upload_s3_data(df1, filename)

        st.markdown(f'### Games for {date}')


        # confidence_threshold = 0 

        df2=df1.loc[(df1.date==date.strftime('%Y-%m-%d'))&
                    (pd.to_numeric(df1.confidence_level)>=pd.to_numeric(confidence_threshold))]

        wins=df2.loc[(df2.ensemble_model_win==1)
                     &(df2.model_run==df2.model_run.max())
                     &(df2.status=='complete')].index.size
        losses=df2.loc[(df2.ensemble_model_win==0)
                       &(df2.model_run==df2.model_run.max())
                       &(df2.status=='complete')].index.size
        try:
           win_pct=wins/(wins+losses)
        except:
           win_pct=0
      
        win_pct = round(100*win_pct, 2)
        st.markdown(f'##### Winning pct: {win_pct}%')

        st.write(df2.loc[(df2.model_run==df2.model_run.max())
                ][['id','status','start_time_et',
                   'time_to_tip','matchup',
                   'ensemble_model_advice','confidence_level','ensemble_model_win','score']].sort_values(['time_to_tip','confidence_level']
                                            ,ascending=[True,False]).reset_index(drop=True).style.background_gradient(subset=['time_to_tip','confidence_level','ensemble_model_win']))


        matchups=df2.id.nunique()

        visualize_model_over_time(
            df2.sort_values(['id','model_run'],ascending=[True,True]),
            matchups
            )

        id_list = df2.id.unique()

        id_select = st.selectbox(
            'Select a game id',
            id_list
        )

        st.write(
            df2.loc[df2.id==id_select][['status','model_run','time_to_tip','matchup','ensemble_model_advice','confidence_level','ttq']+four_feat_list].sort_values('model_run',ascending=True).style.background_gradient(axis=0)
        )

        with st.expander('All data for date:'):
           st.write(df2.loc[df2.date==date.strftime('%Y-%m-%d')].sort_values(['id','model_run'],ascending=[True,True]))






        
        
        
        # df_cbb=get_df_cbb()
        # df=add_trank_data(df_cbb)

        
        # df=run_the_model(df)


        # df['time_to_tip'] = (df['start_time_et']-pd.Timestamp.now(tz='US/Eastern')).dt.total_seconds() / 60

        # save_df=get_s3_data()

        # last_model_run=pd.to_datetime(save_df.model_run).max()
        # current_time=pd.Timestamp.now(tz='US/Eastern')

        # last_model_run = last_model_run.tz_convert('US/Eastern')
        # current_time = current_time.tz_convert('US/Eastern')



        # time_since_last_model=(current_time-last_model_run).total_seconds() / 60
       



        # df=combine_data(df,save_df)

        # if time_since_last_model > 15:
        #     upload_s3_data(df)

        # # df=pd.concat([df,save_df]).sort_values(['id','model_run'],ascending=[True,True]).reset_index(drop=True)
        # # df['matchup'] = df['matchup'].str.replace('  ',' ')
        # # upload_s3_data(df)
        # # df['time_to_tip'] = df['time_to_tip'].round(0)
        # # df['confidence_level'] = df['confidence_level'].round(3)

        # st.markdown('### Visualization of Recent / Upcoming Bets Meeting .05 Threshold')
        # visualize_model_over_time(df)

        # st.markdown('### Upcoming Games w/ Model Advice')
        # st.write(df.loc[
        #     (df.time_to_tip > -0)
        #     &(df.model_run==df.model_run.max())
        #     &(df.time_to_tip < 10*60)
        #     # &(df.confidence_level > .05)
        #     ][['start_time_et','time_to_tip','matchup','ensemble_model_advice','confidence_level']
        #                                                                             ].sort_values(['time_to_tip','confidence_level'],ascending=[True,False]).rename(columns={'time_to_tip':'Mins to Tip',
        #                                                                                                                                                                     'matchup':'Matchup',
        #                                                                                                                                                                     'start_time_et':'Start Time (ET)',
        #                                                                                                                                                                     'ensemble_model_advice':'Model Advice',
        #                                                                                                                                                                     'confidence_level':'Confidence'}).head(50))

        # st.markdown('### Raw Data')
        # st.write(current_time)
        # st.write(last_model_run)
        # st.write(time_since_last_model)
        # st.write(df)


    else:
        st.write('input password above!')




if __name__ == "__main__":
    #execute
    app()

