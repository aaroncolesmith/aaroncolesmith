import streamlit as st
import os
from datetime import datetime, timedelta
import pandas as pd
import requests
from fuzzywuzzy import fuzz
# import h2o
# import plotly_express as px
# from st_files_connection import FilesConnection
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from bs4 import BeautifulSoup
import io
# import pickle


# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import RobustScaler
# from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
# from sklearn.kernel_ridge import KernelRidge

# from sklearn.ensemble import BaggingRegressor

# from sklearn.ensemble import GradientBoostingRegressor
# import xgboost as xgb
# # import lightgbm as lgb
# from sklearn.ensemble import StackingRegressor

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import mean_squared_error
# from sklearn.utils import Bunch


st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog',
    layout='wide'
    )


@st.cache_data(ttl=300)
def get_s3_data(filename):

  session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
  )
  s3 = session.resource('s3')

  obj=s3.meta.client.get_object(Bucket='bet-model-data', Key=f'{filename}.parquet')
  df_hist=pd.read_parquet(io.BytesIO(obj['Body'].read()))

  return df_hist


@st.cache_data(ttl=300)
def get_bet_data(date):

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
    df_cbb['start_time_pt'] = pd.to_datetime(df_cbb['start_time']).dt.tz_localize('US/Pacific')
  try:
     df_cbb['start_time_et'] = pd.to_datetime(df_cbb['start_time']).dt.tz_convert('US/Eastern')
  except:
     df_cbb['start_time_et'] = pd.to_datetime(df_cbb['start_time']).dt.tz_localize('US/Eastern')


  df_cbb['date'] = df_cbb['start_time_pt'].dt.date


  if 'boxscore_total_away_points' not in df_cbb.columns:
     df_cbb['boxscore_total_away_points'] = np.nan
     df_cbb['boxscore_total_home_points'] = np.nan

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

  return df


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
                    'Spartans','Razorbacks','Bears','Raiders','Cardinals','Cardinal','Buckeyes','Hawkeyes','Bobcats','Rockets',
                    'Gauchos']
    for replacement in replacements:
        d[field] = d[field].str.replace(replacement, '', regex=True)
    return d


def upload_s3_data(df, filename):

    for x in ['date','start_time_et','model_run']:
        try:
            df[x]=pd.to_datetime(df[x])
        except:
           df[x]=df[x].apply(lambda x: pd.to_datetime(x).tz_convert('US/Eastern'))

    for x in ['id','score_home','score_away','spread_home','spread_away','spread_home_public','spread_away_public','num_bets','ttq','updated_spread_diff','fav_wins_pred','fav_wins_bin','confidence_level','fav_wins','ensemble_model_win']:
        try:
           df[x] = df[x].replace('nan',np.nan).apply(lambda x: pd.to_numeric(x))
        except Exception as e:
           print(e)

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
    s3.meta.client.upload_file(Filename=f'./{filename}.parquet', 
                               Bucket='bet-model-data', 
                               Key=f'{filename}.parquet'
                               )
    st.write('data uploaded')

@st.cache_data(ttl=300)
def get_trank_schedule_data(date):
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
      replacements = ['\d+', 'BIG12|ESPN+','Peacock', 'ESPNU', 'ESPN', 'FS', '+', 'ACCN', 'BIG|', 'CBSSN', 'PAC','truTV','CBS','BE-T','Ivy-T','NCAA-T','FOX',
                      'WCC-T','Amer-T U','CAA-T','MWC-T','NEC-T','TBS','CUSA-T','BW-T','SECN','SEC-T','P-T','MAC-T','B-T','BTN','MAAC-T','ACC-T','ABC','TNT',
                      'USA Net','ASun-T','Pat-T','Horz-T','BSth-T','OVC-T','MVC-T',' X$', ' LHN',' U$', ' \|',
                      'CW NETWORK','SC-T','Sum-T','MEAC-T','Slnd-T','BSky-T','A-T','WAC-T','Amer-T','SWAC-T','AE-T',' S$',]
      for replacement in replacements:
        trank['matchup'] = trank['matchup'].str.replace(replacement, '', regex=True).str.strip()
      trank['matchup'] = trank['matchup'].str.replace('Illinois Chicago', 'UIC', regex=True)
      trank['matchup'] = trank['matchup'].str.replace('Gardner Webb', 'Gardner-Webb', regex=True)
      trank['trank_spread']='-'+trank.t_rank_line.str.split('-',expand=True)[1].str.split(',',expand=True)[0]

      trank=trank.loc[~trank.matchup.str.contains('MOV Mean')].reset_index(drop=True)

      try:
        trank.loc[trank.matchup.str.contains(' at '), 'away_team']= trank.matchup.str.split(' at ',expand=True)[0].str.strip()
        trank.loc[trank.matchup.str.contains(' at '), 'home_team']= trank.matchup.str.split(' at ',expand=True)[1].str.strip()
      except:
         pass

      try:
        trank.loc[trank.matchup.str.contains(' vs '), 'away_team']= trank.matchup.str.split(' vs ',expand=True)[0].str.strip()
      except:
        pass
      try:
        trank.loc[trank.matchup.str.contains(' vs '), 'home_team']= trank.matchup.str.split(' vs ',expand=True)[1].str.strip()
      except:
        pass


      replacements = ['CW NETWORK','SC-T','Sum-T','MEAC-T','Slnd-T','BSky-T','A-T','WAC-T','Amer-T','SWAC-T','AE-T',' S$',]
      for replacement in replacements:
        trank['home_team'] = trank['home_team'].str.replace(replacement, '', regex=True).str.strip()
  except Exception as e:
    st.write(e)


  return trank

@st.cache_data(ttl=300)
def get_bart_rank(date):
  max_retries = 7
  retries = 0

  while retries < max_retries:
    try:

      url_date=pd.to_datetime(date).strftime('%Y%m%d')
      url = f'https://barttorvik.com/trank-time-machine.php?date={url_date}'
      html_text = requests.get(url).text
      soup = BeautifulSoup(html_text, 'html.parser')
      data = []
      for tr in soup.find('table').find_all('tr'):
          row = [td.text for td in tr.find_all('td')]
          data.append(row)

      df=pd.DataFrame(data)
      df=df.iloc[:,:16]

      df.columns=['rank',
                  'team',
                  'conf',
                  'record',
                  'adjoe',
                  'adjoe_rank',
                  'adjde',
                  'adjde_rank',
                  'barthag',
                  'proj_record',
                  'proj_record_conf',
                  'wab',
                  'wab_rank',
                  'current_rank_final_rank_change',
                  'current_rank',
                  'order',
                  ]
      df=df.loc[df['rank'].notnull()].reset_index(drop=True)
      df=df.drop_duplicates(subset=['rank','team'], keep='first').reset_index(drop=True)
      df['date'] = date

      return df
    except Exception as e:
            print(f"Error: {e}")
            retries += 1
            date -= timedelta(1)

  if retries == max_retries:
    raise Exception("Maximum retries reached. Unable to obtain BART rank.")

@st.cache_data(ttl=300)
def merge_trank_schedule_bet(d1,d2):
  d1['date_join'] = pd.to_datetime(d1['date_join'])
  d2['date'] = pd.to_datetime(d2['date'])

  result_df_merge=pd.merge(
      d1,
      d2[['date','team','rank','adjoe','adjde','barthag','wab']],
      how='left',
      left_on=['date_join','away_team'],
      right_on=['date','team'],
      suffixes=('','_rank_data')
  )


  result_df_merge = result_df_merge.drop(['date_rank_data','team'], axis=1)
  result_df_merge = result_df_merge.rename(columns={'rank':'away_team_rank',
                                                    'adjoe':'away_team_adjoe',
                                                    'adjde':'away_team_adjde',
                                                    'barthag':'away_team_barthag',
                                                    'wab':'away_team_wab'})

  result_df_merge=pd.merge(
      result_df_merge,
      d2[['date','team','rank','adjoe','adjde','barthag','wab']],
      how='left',
      left_on=['date_join','home_team'],
      right_on=['date','team'],
      suffixes=('','_rank_data')
  )

  result_df_merge = result_df_merge.drop(['date_rank_data','team'], axis=1)
  result_df_merge = result_df_merge.rename(columns={'rank':'home_team_rank',
                                                    'adjoe':'home_team_adjoe',
                                                    'adjde':'home_team_adjde',
                                                    'barthag':'home_team_barthag',
                                                    'wab':'home_team_wab'})

  result_df_merge = result_df_merge.drop(['date_join'], axis=1)

  max_rank=max(pd.to_numeric(result_df_merge['home_team_rank']).max(),pd.to_numeric(result_df_merge['away_team_rank']).max()) + 10
  result_df_merge['away_team_rank'] = result_df_merge['away_team_rank'].fillna(max_rank)

  result_df_merge=result_df_merge.loc[result_df_merge.date <= pd.to_datetime(pd.Timestamp.now(tz='US/Pacific').date())]
  return result_df_merge


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


@st.cache_data(ttl=300)
def merge_bet_trank(df,df_trank):
  unique_dates = df['date'].unique()
  df_all=pd.DataFrame()
  for date in unique_dates:
    print(date)
    try:
      df_date = df[df['date'] == date]
      df_trank_date = df_trank[df_trank['date'] == pd.to_datetime(date)]
      df_date[['best_match', 'similarity_score']] = df_date.apply(
          lambda row: pd.Series(find_best_match(row, df_trank_date['matchup'].tolist())),
          axis=1
      )

      merged_df_date = pd.merge(df_date, df_trank_date, left_on='best_match', right_on='matchup', how='left', suffixes=('', '_trank'))

      df_all=pd.concat([df_all,merged_df_date])

    except Exception as e:
      print('failed')
      print(date)
      print(e)

  df_all=df_all.reset_index(drop=True)
  df_all['trank_favorite'] = df_all['t_rank_line'].str.split('-',expand=True)[0]
  df_all['favorite_spread_bovada']=df_all[['spread_away','spread_home']].min(axis=1)
  df_all['favorite_spread_bovada'] = pd.to_numeric(df_all['favorite_spread_bovada'])

  df_all.loc[df_all.spread_away < 0,'bovada_favorite'] = df_all['away_team']
  df_all.loc[df_all.spread_home < 0,'bovada_favorite'] = df_all['home_team']
  df_all['fav_similarity'] = df_all.apply(lambda row: fuzz.ratio(str(row['trank_favorite']), str(row['bovada_favorite'])) if not (pd.isna(row['trank_favorite']) or pd.isna(row['bovada_favorite'])) else 0, axis=1)

  df_all.loc[df_all.fav_similarity >=37,'updated_spread_diff'] = abs(pd.to_numeric(df_all['favorite_spread_bovada']) - pd.to_numeric(df_all['trank_spread']))
  df_all.loc[df_all.fav_similarity <=37,'updated_spread_diff'] = abs(-pd.to_numeric(df_all['favorite_spread_bovada']) - pd.to_numeric(df_all['trank_spread']))


  return df_all

def create_category_columns(df, column_name, bins, labels):
    df[f'{column_name}_category'] = pd.cut(df[column_name], bins=bins, labels=labels, include_lowest=True)
    df = pd.concat([df, pd.get_dummies(df[f'{column_name}_category'])], axis=1)
    return df

@st.cache_data(ttl=300)
def add_bins(df):

  for col in ['updated_spread_diff','num_bets','ttq','home_team_rank','away_team_rank']:
    df[col]=pd.to_numeric(df[col])

  spread_bins = [0, 0.8, 1.6, 2.8, 5.8, 10000]
  spread_labels = ['spread_diff_less_than_8', 'spread_diff_8_to_16', 'spread_diff_16_to_28', 'spread_diff_28_to_58', 'spread_diff_58_plus']
  df = create_category_columns(df, 'updated_spread_diff', spread_bins, spread_labels)

  bets_bins = [0, 1900, 3500, 6900, 14000, 10000000]
  bets_labels = ['bets_less_than_1900', 'bets_1900_to_3500', 'bets_3500_to_6900', 'bets_6900_to_14k', 'bets_14k_plus']
  df = create_category_columns(df, 'num_bets', bets_bins, bets_labels)

  bins = [0, 34, 44, 56, 75, 10000000]
  ttq_labels = ['ttq_less_than_34', 'ttq_34_to_44', 'ttq_44_to_56', 'ttq_56_to_75', 'ttq_75_plus']
  df = create_category_columns(df, 'ttq', bins, ttq_labels)

  bins = [10, 25, 100, 200, 10000000]
  home_rank_labels = ['home_rank_less_than_10', 'home_rank_10_to_25', 'home_rank_100_to_200', 'home_rank_200_plus']
  df = create_category_columns(df, 'home_team_rank', bins, home_rank_labels)

  bins = [10, 25, 100, 200, 10000000]
  away_rank_labels = ['away_rank_less_than_10', 'away_rank_10_to_25', 'away_rank_100_to_200', 'away_rank_200_plus']
  df = create_category_columns(df, 'away_team_rank', bins, away_rank_labels)

  ## added these to support the three model
  df.loc[df.spread_home_public > 50, 'home_betting_fav'] = 1

  df.loc[df.spread_away_public > 50, 'away_betting_fav'] = 1

  df.loc[df.spread_home_public >= 75, 'big_home_betting_fav'] = 1

  df.loc[df.spread_away_public > 75, 'big_away_betting_fav'] = 1

  df.loc[(df.spread_home_public > 50)&(df.spread_home <0), 'public_on_fav'] = 1
  df.loc[(df.spread_away_public > 50)&(df.spread_away <0), 'public_on_fav'] = 1

  df.loc[(df.spread_home_public > 50)&(df.spread_home >0), 'public_on_dog'] = 1
  df.loc[(df.spread_away_public > 50)&(df.spread_away >0), 'public_on_dog'] = 1


  for col in ['home_betting_fav','away_betting_fav','big_home_betting_fav','big_away_betting_fav','public_on_fav','public_on_dog']:
    df[col] = df[col].fillna(0)

  d5=pd.read_parquet('https://github.com/aaroncolesmith/bet_model/blob/main/d5_against_spread.parquet?raw=true', engine='pyarrow')
  df=pd.merge(df,
          d5[['id','team_id','rolling_cover_10','rolling_cover_25','rolling_cover_50','rolling_cover_100']],
          left_on=['id','home_team_id'],
          right_on=['id','team_id']
          ).rename(columns={'rolling_cover_10':'home_rolling_cover_10',
                            'rolling_cover_25':'home_rolling_cover_25',
                            'rolling_cover_50':'home_rolling_cover_50',
                            'rolling_cover_100':'home_rolling_cover_100'})
  del df['team_id']

  df=pd.merge(df,
          d5[['id','team_id','rolling_cover_10','rolling_cover_25','rolling_cover_50','rolling_cover_100']],
          left_on=['id','away_team_id'],
          right_on=['id','team_id']
          ).rename(columns={'rolling_cover_10':'away_rolling_cover_10',
                            'rolling_cover_25':'away_rolling_cover_25',
                            'rolling_cover_50':'away_rolling_cover_50',
                            'rolling_cover_100':'away_rolling_cover_100'})
  del df['team_id']

  return df


def logged_in_app():
  st.write('hello and welcome')
  today=pd.Timestamp.now(tz='US/Eastern').date()
  model_run=pd.Timestamp.now(tz='US/Eastern')

  with st.form(key='date_selector_form'):
              c1,c2,c3=st.columns(3)
              date = c1.date_input(
              "Select a date / month for games",
                  value=today,
                  min_value=today - timedelta(days=10),
                  max_value=today + timedelta(days=0)
              )

              confidence_threshold=c2.slider(
                'Set a confidence threshold (0 selects all games)',
                min_value=0.00,
                max_value=0.50
              )

              # model_select=c3.selectbox(
              #   'Select a model',
              #   list(model_dict.keys())
              # )

              date_submit_button=st.form_submit_button(label='Go!')



  if date_submit_button:
    ## get historical data
    filename = 'cbb_model_data'
    try:
      df = get_s3_data(filename)
      df['model_run'] = model_run

      st.write(df.head(5))
    except:
      st.write('no data -- create new file')

    df = get_bet_data(date)
    st.write(df.head(5))
    d1=get_trank_schedule_data(date)
    st.write(d1.head(5))
    rank_date=date-timedelta(1)
    d2=get_bart_rank(rank_date)
    d1['date_join'] = rank_date
    d3=merge_trank_schedule_bet(d1,d2)
    df=merge_bet_trank(df,d3)
    st.write(df.head(5))
    df=add_bins(df)
    st.write(df.head(5))


    upload_s3_data(df, filename)

    ## if the data was updated in the past 5 minutes, don't update it, otherwise get new data



def app():
  st.title('Welcome to the CBB Model')
  st.markdown('### Enter Password to Continue')
  if 'input_pass' not in st.session_state:
    st.session_state['input_pass'] = ''
  with st.form(key='pass_input'):
    input_pass=st.text_input('What\'s the password?', st.session_state['input_pass'])
    password=os.environ["TEST_PASS"]
    submit_button=st.form_submit_button(label='Go!')

  if input_pass==password:
    st.session_state.input_pass = input_pass
    st.write('logged in')
    logged_in_app()


if __name__ == "__main__":
    #execute
    app()


