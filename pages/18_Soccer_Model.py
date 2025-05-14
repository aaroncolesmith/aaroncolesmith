import pandas as pd
import streamlit as st
import plotly.express as px
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
    )


@st.cache_resource
def load_data():
    df = pd.read_parquet('https://github.com/aaroncolesmith/data_action_network/raw/refs/heads/main/data/soccer_model.parquet', engine='pyarrow')
    return df


def get_prob(a):
    odds = 0
    if a < 0:
        odds = (-a)/(-a + 100)
    else:
        odds = 100/(100+a)

    return odds

def fav_payout(ml):
    try:
        return 100 / abs(ml)
    except ZeroDivisionError:
        return None

def dog_payout(ml):
  return ml/100


def load_model(model):
    if model == 'first_model':
        loaded_model = pickle.load(open('models/xgboost_model.pkl', 'rb'))
        loaded_feat_list = pickle.load(open('models/feat_list.pkl', 'rb'))
        loaded_preprocessor = pickle.load(open('models/score_preprocessor.pkl', 'rb'))
    elif model == 'logistic':
        loaded_model = pickle.load(open('models/logistic_model.pkl', 'rb'))
        loaded_feat_list = pickle.load(open('models/feat_list_logistic.pkl', 'rb'))
        loaded_preprocessor = pickle.load(open('models/score_preprocessor_logistic.pkl', 'rb'))
    return loaded_model, loaded_feat_list, loaded_preprocessor




def app():
    st.title('Soccer Model')
    df = load_data()
    c1,c2=st.columns([1,3])
    model = c1.selectbox(
        'Select a model to use',
        options=['first_model'],
        index=0
    )
    loaded_model, loaded_feat_list, loaded_preprocessor = load_model(model)



    
    df_clean = df[loaded_feat_list]
    X_processed = loaded_preprocessor.transform(df_clean)
    y_pred = loaded_model.predict(X_processed)


    df[['home_score_pred','away_score_pred']] = y_pred
    df['predicted_score'] = df['home_score_pred'].round(2).astype(str) + ' - ' + df['away_score_pred'].round(2).astype(str)
    df['actual_score'] = df['home_score'].astype(str) + ' - ' + df['away_score'].astype(str)
    df['home_score_pred_rounded'] = df['home_score_pred'].round(0)
    df['away_score_pred_rounded'] = df['away_score_pred'].round(0)
    df['home_score_pred_abs_diff'] = abs(df['home_score_pred']-df['home_score_pred_rounded'])
    df['away_score_pred_abs_diff'] = abs(df['away_score_pred']-df['away_score_pred_rounded'])
    df['home_score_pred_correct'] = np.where(df['home_score_pred_rounded']==df['home_score'],1,0)
    df['away_score_pred_correct'] = np.where(df['away_score_pred_rounded']==df['away_score'],1,0)
    df['both_score_pred_correct'] = np.where((df['home_score_pred_rounded']==df['home_score']) & (df['away_score_pred_rounded']==df['away_score']),1,0)
    df['predicted_total'] = df['home_score_pred']+df['away_score_pred']
    df['predicted_total_diff'] = df['predicted_total'] - df['odds_adjusted_total']
    upper_band = (df['predicted_total'] - df['total']).quantile(.75)
    lower_band = (df['predicted_total'] - df['total']).quantile(.25)
    if lower_band > 0:
        lower_band = -.35
    df['model_bet_over'] = np.where(
        (df['predicted_total_diff']) > upper_band,
        1,
        np.nan
    )
    df['model_bet_under'] = np.where(
        (df['predicted_total_diff']) < lower_band,
        1,
        np.nan
    )
    df['model_bet_over_payout'] = np.where(
        (df['predicted_total_diff']) > upper_band,
        df['over_payout'],
        np.nan
    )
    df['model_bet_under_payout'] = np.where(
        (df['predicted_total_diff']) < lower_band,
        df['under_payout'],
        np.nan
    )
    df['spread_home_pred'] = df['away_score_pred'] - df['home_score_pred']
    df['odds_adjusted_spread_home'] = df['spread_home']*((1-df['spread_home_line'].apply(get_prob))+.5)
    df['predicted_spread_home_diff'] = df['odds_adjusted_spread_home'] - df['spread_home_pred']
    df['match_title'] = df['away_team'] + ' @ ' + df['home_team']
    upper_band = df['predicted_spread_home_diff'].quantile(.75)
    lower_band = df['predicted_spread_home_diff'].quantile(.25)

    df['model_bet_home_cover'] = np.where(
        df['predicted_spread_home_diff'] > upper_band,
        1,
        np.nan
    )
    df['model_bet_away_cover'] = np.where(
        df['predicted_spread_home_diff'] < lower_band,
        1,
        np.nan
    )
    df['model_bet_home_payout'] = np.where(
        df['predicted_spread_home_diff'] > upper_band,
        df['spread_home_payout'],
        np.nan
    )
    df['model_bet_away_payout'] = np.where(
        df['predicted_spread_home_diff'] < lower_band,
        df['spread_away_payout'],
        np.nan
    )

    today = pd.to_datetime('today').date()


    c1,c2=st.columns([1,3])
    date_selector = c1.date_input(
        "Select a date for games",
        value=today,
        min_value=pd.to_datetime('1966-02-19'),
        max_value=pd.to_datetime(df.date.max())
        )



    df['start_time_pt'] = pd.to_datetime(df['start_time_pt'])
    todays_games = df.loc[df['start_time_pt'].dt.date == date_selector].reset_index(drop=True)
    ## if odds_adjusted_total minus predicted total is less than or equal to -.5 then 'bet the under'
    ## if odds_adjusted_total minus predicted total is greater than or equal to .5 then 'bet the over'
    ## if odds_adjusted_total between -.5 and .5 then 'no bet'
    todays_games['total_bet'] = np.where(
        (todays_games['odds_adjusted_total'] - todays_games['predicted_total']) > .25,
        'bet the under',
        np.where(
            (todays_games['odds_adjusted_total'] - todays_games['predicted_total']) < -.25,
            'bet the over',
            'no bet'
        )
    )

    todays_games['total_result'] = np.select(
        [
            todays_games['status'] != 'complete',
            (todays_games['total_bet'] == 'bet the over') & (todays_games['total_score'] > todays_games['total']),
            (todays_games['total_bet'] == 'bet the under') & (todays_games['total_score'] < todays_games['total']),
            todays_games['total_bet'] == 'no bet'
        ],
        [
            'game not complete',
            'win',
            'win',
            'no bet'
        ],
        default='default_value'
    )

    todays_games['predicted_spread_home'] = todays_games['away_score_pred'] - todays_games['home_score_pred']
    todays_games['odds_adjusted_score_home'] = (todays_games['odds_adjusted_total'] / 2) - (todays_games['odds_adjusted_spread_home'] / 2)
    todays_games['odds_adjusted_score_away'] = (todays_games['odds_adjusted_total'] / 2) + (todays_games['odds_adjusted_spread_home'] / 2)
    todays_games['odds_adjusted_score'] = todays_games['odds_adjusted_score_home'].round(2).astype(str) + ' - ' + todays_games['odds_adjusted_score_away'].round(2).astype(str)

    todays_games['spread_home_bet'] = np.where(
        (todays_games['predicted_spread_home_diff'] > .25),
        'bet the home team',
        np.where(
            (todays_games['predicted_spread_home_diff'] < -.25),
            'bet the away team',
            'no bet'
        )
    )

    todays_games['spread_home_result'] = np.select(
        [
            todays_games['status'] != 'complete',
            (todays_games['predicted_spread_home_diff'] > 0) & (todays_games['home_score'] > todays_games['away_score']),
            (todays_games['predicted_spread_home_diff'] < 0) & (todays_games['home_score'] < todays_games['away_score']),
            todays_games['predicted_spread_home_diff'] == 0
        ],
        [
            'game not complete',
            'win',
            'win',
            'no bet'
        ],
        default='default_value'
    )






    ## select which columns to show in a dataframe
    selected_columns = st.multiselect(
        "Select columns to display",
        options=todays_games.columns.tolist(),
        default=['start_time_pt','status','match_title','spread_home','predicted_spread_home',
                 'odds_adjusted_spread_home','predicted_spread_home_diff','predicted_total','odds_adjusted_total',
                 'predicted_total_diff','spread_home_bet','spread_home_result','predicted_score','actual_score','odds_adjusted_score'
                          ]
    )


    fig = px.bar(todays_games.sort_values('predicted_spread_home_diff', ascending=False),
      y='predicted_spread_home_diff',
      x='match_title',
      color='spread_home_bet',
      hover_name='match_title',
      hover_data=['predicted_score','odds_adjusted_score','actual_score'],
    template = 'simple_white',
    color_discrete_sequence=['lightblue','gray','coral'],
        text_auto=True,
        orientation = 'v')
    # fig.update_xaxes(tickformat = ',.1%')
    fig.update_traces(marker=dict(
        # color=['lightblue','red','gray'],        # Fill color of the bars
        line=dict(color='navy', width=2)  # Outline color and thickness
    ),texttemplate = "%{value:.2f}"
    )
    fig.update_layout(
            font=dict(
            family='Futura',  # Set font to Futura
            size=12,          # You can adjust the font size if needed
            color='black'     # Font color
        ),
    )
    st.plotly_chart(fig, use_container_width=True)



    ## do the same thing as above but for the total score
    fig = px.bar(todays_games.sort_values('predicted_total_diff', ascending=False),
      y='predicted_total_diff',
      x='match_title',
      color='total_bet',
      hover_name='match_title',
      hover_data=['predicted_score','odds_adjusted_score','actual_score'],
    template = 'simple_white',
    color_discrete_sequence=['lightblue','gray','coral'],
        text_auto=True,
        orientation = 'v')
    # fig.update_xaxes(tickformat = ',.1%')
    fig.update_traces(marker=dict(
        # color=['lightblue','red','gray'],        # Fill color of the bars
        line=dict(color='navy', width=2)  # Outline color and thickness
    ),texttemplate = "%{value:.2f}"
    )
    fig.update_layout(
            font=dict(
            family='Futura',  # Set font to Futura
            size=12,          # You can adjust the font size if needed
            color='black'     # Font color
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


    with st.expander("Show all data", expanded=False):
        st.dataframe(todays_games,
                use_container_width=True,
                hide_index=True,
                )

if __name__ == "__main__":
    #execute
    app()