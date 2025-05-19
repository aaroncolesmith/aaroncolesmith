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

    df['match_title'] = df['away_team'] + ' @ ' + df['home_team']
    df[['home_score_pred','away_score_pred']] = y_pred
    df['predicted_score'] = df['home_score_pred'].round(2).astype(str) + ' - ' + df['away_score_pred'].round(2).astype(str)
    df['actual_score'] = df['home_score'].astype(str) + ' - ' + df['away_score'].astype(str)
    # df['home_score_pred_rounded'] = df['home_score_pred'].round(0)
    # df['away_score_pred_rounded'] = df['away_score_pred'].round(0)
    # df['home_score_pred_abs_diff'] = abs(df['home_score_pred']-df['home_score_pred_rounded'])
    # df['away_score_pred_abs_diff'] = abs(df['away_score_pred']-df['away_score_pred_rounded'])
    # df['home_score_pred_correct'] = np.where(df['home_score_pred_rounded']==df['home_score'],1,0)
    # df['away_score_pred_correct'] = np.where(df['away_score_pred_rounded']==df['away_score'],1,0)
    # df['both_score_pred_correct'] = np.where((df['home_score_pred_rounded']==df['home_score']) & (df['away_score_pred_rounded']==df['away_score']),1,0)
    df['predicted_total'] = df['home_score_pred']+df['away_score_pred']
    df['predicted_total_diff'] = df['predicted_total'] - df['odds_adjusted_total']
    df['predicted_total_pct_diff'] = df['predicted_total_diff'] / df['odds_adjusted_total']

    df.loc[df['under'] < 0, 'under_payout'] = df['under'].apply(fav_payout)*df['under_hit']
    df.loc[df['under'] > 0, 'under_payout'] = df['under'].apply(dog_payout)*df['under_hit']
    df.loc[df['under_hit'] == 0, 'under_payout'] = -1


    df.loc[df['over'] < 0, 'over_payout'] = df['over'].apply(fav_payout)*df['over_hit']
    df.loc[df['over'] > 0, 'over_payout'] = df['over'].apply(dog_payout)*df['over_hit']
    df.loc[df['over_hit'] == 0, 'over_payout'] = -1

    upper_band = (df['predicted_total_pct_diff']).quantile(.75)
    lower_band = (df['predicted_total_pct_diff']).quantile(.25)
    c1,c2,c3,c4,c5=st.columns(5)
    upper_band_select = c2.number_input(
        'Select the upper band, anything above this is a bet on the over',
        value=upper_band,
        step=.01,
        format="%.2f",
        # min_value=lower_band_select
    )

    lower_band_select = c1.number_input(
        'Select the lower band, anything below this is a bet on the under',
        value=lower_band,
        step=.01,
        format="%.2f",
        max_value=upper_band_select
    )


    df['model_bet_over'] = np.where(
        (df['predicted_total_pct_diff']) > upper_band_select,
        1,
        0
    )
    df['model_bet_under'] = np.where(
        (df['predicted_total_pct_diff']) < lower_band_select,
        1,
        0
    )
    df['model_bet_over_payout'] = np.where(
        (df['predicted_total_pct_diff'] > upper_band_select) & (df['status'] == 'complete'),
        df['over_payout'],
        np.nan
    )
    df['model_bet_under_payout'] = np.where(
        (df['predicted_total_pct_diff'] < lower_band_select) & (df['status'] == 'complete'),
        df['under_payout'],
        np.nan
    )
    df['total_bet'] = np.where(
        (df['predicted_total_pct_diff']) > upper_band_select,
        'bet the over',
        np.where(
            (df['predicted_total_pct_diff']) < lower_band_select,
            'bet the under',
            'no bet'
        )
    )

    df['total_result'] = np.select(
        [
            df['status'] != 'complete',
            (df['total_bet'] == 'bet the over') & (df['total_score'] > df['total']),
            (df['total_bet'] == 'bet the over') & (df['total_score'] < df['total']),
            (df['total_bet'] == 'bet the under') & (df['total_score'] < df['total']),
            (df['total_bet'] == 'bet the under') & (df['total_score'] > df['total']),
            (df['total_score'] == df['total']),
            df['total_bet'] == 'no bet'
        ],
        [
            'game not complete',
            'win - over hit',
            'loss - under hit',
            'win - under hit',
            'loss - over hit',
            'push',
            'no bet'
        ],
        default='default_value'
    )
    


    ## home cover section
    df['spread_home_pred'] = df['away_score_pred'] - df['home_score_pred']
    df['odds_adjusted_spread_home'] = df['spread_home']*((1-df['spread_home_line'].apply(get_prob))+.5)
    df['predicted_spread_home_diff'] = df['odds_adjusted_spread_home'] - df['spread_home_pred']
    df['predicted_spread_home'] = df['away_score_pred'] - df['home_score_pred']
    df['odds_adjusted_score_home'] = (df['odds_adjusted_total'] / 2) - (df['odds_adjusted_spread_home'] / 2)
    df['odds_adjusted_score_away'] = (df['odds_adjusted_total'] / 2) + (df['odds_adjusted_spread_home'] / 2)
    df['odds_adjusted_score'] = df['odds_adjusted_score_home'].round(2).astype(str) + ' - ' + df['odds_adjusted_score_away'].round(2).astype(str)

    # df['predicted_home_score_diff'] = df['home_score_pred'] - df['odds_adjusted_score_home']
    # df['predicted_home_score_pct_diff'] = df['predicted_home_score_diff'] / df['odds_adjusted_score_home']

    df['predicted_home_spread_diff'] = df['spread_home_pred'] - df['odds_adjusted_spread_home']
    df['predicted_home_spread_pct_diff'] = df['predicted_home_spread_diff'] / df['odds_adjusted_spread_home']



    upper_band_home = (df['predicted_home_spread_pct_diff']).quantile(.75)
    lower_band_home = (df['predicted_home_spread_pct_diff']).quantile(.25)
    upper_band_home_select = c5.number_input(
        'Select the upper band for home cover, anything above this is a bet on the home team to cover',
        value=upper_band_home,
        step=.01,
        format="%.2f",
        # min_value=lower_band_select
    )
    lower_band_home_select = c4.number_input(
        'Select the lower band for away cover, anything below this is a bet on the away team to cover', 
        value=lower_band_home,
        step=.01,
        format="%.2f",
        max_value=upper_band_home_select
    )
    df['model_bet_home_cover'] = np.where(
        (df['predicted_home_spread_pct_diff']) > upper_band_home_select,
        1,
        0
    )
    df['model_bet_away_cover'] = np.where(
        (df['predicted_home_spread_pct_diff']) < lower_band_home_select,
        1,
        0
    )
    df['model_bet_home_payout'] = np.where(
        (df['predicted_home_spread_pct_diff'] > upper_band_home_select) & (df['status'] == 'complete'),
        df['spread_home_payout'],
        np.nan
    )
    df['model_bet_away_payout'] = np.where(
        (df['predicted_home_spread_pct_diff'] < lower_band_home_select) & (df['status'] == 'complete'),
        df['spread_away_payout'],
        np.nan
    )
    df['spread_home_bet'] = np.where(
        (df['predicted_home_spread_pct_diff']) > upper_band_home_select,
        'bet the home team',
        np.where(
            (df['predicted_home_spread_pct_diff']) < lower_band_home_select,
            'bet the away team',
            'no bet'
        )
    )
    
    df['spread_home_result'] = np.select(
        [
            df['status'] != 'complete',
            (df['spread_home_bet'] == 'bet the home team') & ((df['home_score']+df['spread_home']) > df['away_score']),
            (df['spread_home_bet'] == 'bet the home team') & ((df['home_score']+df['spread_home']) < df['away_score']),
            (df['spread_home_bet'] == 'bet the away team') & ((df['home_score']+df['spread_home']) < df['away_score']),
            (df['spread_home_bet'] == 'bet the away team') & ((df['home_score']+df['spread_home']) > df['away_score']),
            ((df['home_score']+df['spread_home']) == df['away_score']),
            df['spread_home_bet'] == 'no bet'
        ],
        [
            'game not complete',
            'win - home team cover',
            'loss - away team cover',
            'win - away team cover',
            'loss - home team cover',
            'push',
            'no bet'
        ],
        default='default_value'
    )

    df['total_payout'] = df['model_bet_home_payout'].fillna(0) + df['model_bet_away_payout'].fillna(0) + df['model_bet_over_payout'].fillna(0) + df['model_bet_under_payout'].fillna(0)

    today_viz, date_range_viz = st.tabs(['Today\'s Games', 'Date Range'])
    with today_viz:

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

        under_payout = todays_games['model_bet_under_payout'].sum().round(2)
        over_payout = todays_games['model_bet_over_payout'].sum().round(2)
        home_cover_payout = todays_games['model_bet_home_payout'].sum().round(2)
        away_cover_payout = todays_games['model_bet_away_payout'].sum().round(2)
        total_payout = under_payout + over_payout + home_cover_payout + away_cover_payout
        total_payout = total_payout.round(2)
        st.write(f"Total payout for the day -- betting total: {total_payout}")
        st.write(f"Total payout for the day -- betting over: {over_payout}")
        st.write(f"Total payout for the day -- betting under: {under_payout}")
        st.write(f"Total payout for the day -- betting home cover: {home_cover_payout}")
        st.write(f"Total payout for the day -- betting away cover: {away_cover_payout}")




        ## select which columns to show in a dataframe
        selected_columns = st.multiselect(
            "Select columns to display",
            options=todays_games.columns.tolist(),
            default=['start_time_pt','status','match_title','spread_home','predicted_spread_home',
                    'odds_adjusted_spread_home','predicted_spread_home_diff',
                    'predicted_total_diff','predicted_score','actual_score','odds_adjusted_score',
                    'predicted_total_pct_diff','predicted_total','odds_adjusted_total','total_bet','total_result',
                    'model_bet_home_cover','model_bet_away_cover','model_bet_home_payout','model_bet_away_payout','spread_home_bet','spread_home_result'
                            ]
        )

        st.dataframe(todays_games[selected_columns],
                    use_container_width=True,
                    hide_index=True,
                    )
        
        
        ## do the same thing as above but for the total score
        fig = px.bar(todays_games.sort_values('predicted_total_pct_diff', ascending=False),
        y='predicted_total_pct_diff',
        x='match_title',
        color='total_bet',
        hover_name='match_title',
        hover_data=['predicted_score','odds_adjusted_score','actual_score','total_result','predicted_total','odds_adjusted_total'],
        template = 'simple_white',
        color_discrete_sequence=['lightblue','gray','coral'],
            text_auto=True,
            orientation = 'v')
        # fig.update_xaxes(tickformat = ',.1%')
        fig.update_traces(marker=dict(
            # color=['lightblue','red','gray'],        # Fill color of the bars
            line=dict(color='navy', width=2)  # Outline color and thickness
        ),
        texttemplate = "%{value:.1%}",  # Format as percentage with one decimal place
        textposition='outside'  # Adjust text position for better readability
        )
        fig.update_layout(
            height= 800,
                font=dict(
                family='Futura',  # Set font to Futura
                size=12,          # You can adjust the font size if needed
                color='black'     # Font color
            ),
        )
        fig.update_yaxes(title_text='Predicted Total Percent Difference', title_font=dict(size=14), tickformat = ',.0%')
        st.plotly_chart(fig, use_container_width=True)


        fig = px.bar(todays_games.sort_values('predicted_home_spread_pct_diff', ascending=False),
        y='predicted_home_spread_pct_diff',
        x='match_title',
        color='spread_home_bet',
        hover_name='match_title',
        hover_data=['predicted_score','odds_adjusted_score','actual_score','spread_home_pred','odds_adjusted_spread_home','spread_home_result'],
        template = 'simple_white',
        color_discrete_sequence=['lightblue','gray','coral'],
            text_auto=True,
            orientation = 'v')
        # fig.update_xaxes(tickformat = ',.1%')
        fig.update_traces(marker=dict(
            # color=['lightblue','red','gray'],        # Fill color of the bars
            line=dict(color='navy', width=2)  # Outline color and thickness
        ),
        texttemplate = "%{value:.1%}",  # Format as percentage with one decimal place
        textposition='outside'  # Adjust text position for better readability
        )
        fig.update_yaxes(title_text='Predicted Home Score Percent Difference', title_font=dict(size=14), tickformat = ',.0%')
        fig.update_layout(
            height= 800,
                font=dict(
                family='Futura',  # Set font to Futura
                size=12,          # You can adjust the font size if needed
                color='black'     # Font color
            ),
        )
        st.plotly_chart(fig, use_container_width=True)


    # defaults values for date range -- 4 weeks ago to today
    date_range_end = pd.to_datetime('today').date() - pd.DateOffset(days=1)
    date_range_start = date_range_end - pd.DateOffset(weeks=4)
    with date_range_viz:
        start_date, end_date = st.date_input(
            "Select a date range",
            value=(date_range_start, date_range_end),
            min_value=pd.to_datetime('1966-02-19'),
            max_value=pd.to_datetime(df.date.max()),
            )
        df['start_time_pt'] = pd.to_datetime(df['start_time_pt'])
        date_range_games = df.loc[(df['start_time_pt'].dt.date >= start_date) & (df['start_time_pt'].dt.date <= end_date)].reset_index(drop=True)

        ## select which columns to show in a dataframe
        # selected_columns_date_range = st.multiselect(
        #     "Select columns to display",
        #     options=date_range_games.columns.tolist(),
        #     default=['start_time_pt','status','match_title','spread_home','predicted_spread_home',
        #             'odds_adjusted_spread_home','predicted_spread_home_diff',
        #             'predicted_total_diff','predicted_score','actual_score','odds_adjusted_score',
        #             'predicted_total_pct_diff','predicted_total','odds_adjusted_total','total_bet','total_result',
        #             'model_bet_home_cover','model_bet_away_cover','model_bet_home_payout','model_bet_away_payout','spread_home_bet','spread_home_result'
        #                     ],
        #                     key='date_range_select'
        # )

        # st.dataframe(date_range_games[selected_columns_date_range],
        #             use_container_width=True,
        #             hide_index=True,
        #             )
        
        date_range_games['date'] = date_range_games['start_time_pt'].dt.date
        df_date_range_payout = date_range_games.groupby(['date']).agg(
            total_payout=('total_payout', 'sum'),
            bet_over_payout=('model_bet_over_payout', 'sum'),
            bet_under_payout=('model_bet_under_payout', 'sum'),
            bet_home_cover_payout=('model_bet_home_payout', 'sum'),
            bet_away_cover_payout=('model_bet_away_payout', 'sum'),
            total_bet=('total_bet', 'count'),).reset_index()

        df_date_range_payout['total_payout_cumsum'] = df_date_range_payout['total_payout'].cumsum()

        fig = px.scatter(
            df_date_range_payout,
            x='date',
            y=["total_payout","total_payout_cumsum","bet_over_payout","bet_under_payout","bet_home_cover_payout","bet_away_cover_payout"],
            # color='team',
            template = 'simple_white',
            render_mode='svg',
            color_discrete_map={
                            "Order Capture": "#ff2600",
                            "DC Drop": "#ff6600",
                            "Order Ship": "#ffa600",
                            "Oceania": "goldenrod",
                            "Africa": "magenta"},
        )
        fig.update_traces(
            mode='markers+lines',
            line_shape='spline',
            line=dict(width=4),
                # marker_symbol=["line-ns",'star'],
                marker=dict(size=12, opacity=1, 
                            line=dict(width=2, 
                                    color="DarkSlateGrey"
                                    )
                        ),
            )
        fig.update_layout(
                    font_family='Futura',
                    height=800,
            font_color='black',
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