import pandas as pd
import numpy as np
import plotly_express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import streamlit as st

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog',
    layout='wide'
    )

@st.cache_data
def load_data():

    df=pd.read_parquet('https://github.com/aaroncolesmith/data_sources/raw/main/nba_draft_data.parquet', engine='pyarrow')
    for x in ['PPG','WS','WS_48','VORP','MP','PK','YRS','PTS','BPM','G']:
        df[x] = pd.to_numeric(df[x],errors='coerce').fillna(0)
    return df


def app():
    st.write("""
    # Welcome to NBA Draft Analsyis!
    """)
    df = load_data()

    def create_new_column(row):
        if row['PK'] >= 100:
            return '100+'
        if row['PK'] >= 31:
            return '31-99'
        if row['PK'] >= 15:
            return '15-30'
        else:
            return str(row['PK'])

    df['PICK'] = df.apply(create_new_column, axis=1)


    metric_select = st.selectbox('Select a metric:', ['WS', 'WS_48', 'VORP', 'BPM', 'PPG', 'MP'])
    min_games = st.number_input('Minimum Games Played',
                                  value=0,
                                  step=1)
    df1 = df.query("G > @min_games").copy()
    # remove_never_player = st.checkbox('Remove players who never played?')

    # if remove_never_player:
    #     df = df.query("MP > 0")

    c1,c2=st.columns(2)
    axis_sort = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "11", "12", "13", "14",
        "15-30", "31-99", "100+"
    ]

    fig = px.box(
        df1,
        x="PICK",
        y=metric_select,
        points="all",
        hover_data=['PLAYER', 'TM', 'COLLEGE', 'YEAR'],
        title=f'Distribution of {metric_select} by Draft Pick',
        template='plotly_white'
    )
    fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': axis_sort})

    c1.plotly_chart(fig, use_container_width=True)

    st.write(df1)

    sorted_df = df1.sort_values(['YEAR',metric_select],ascending=[True,False])
    # Get the top 5 players for each year
    top_players = sorted_df.groupby(['YEAR']).head(5)
    # Create a new dataframe with 'YEAR' and the top players for each year
    result_df = pd.DataFrame({'YEAR': top_players['YEAR'], 'TOP_PLAYERS': top_players['PLAYER'] + ' - ' + top_players[metric_select].astype(str)}).groupby(['YEAR']).agg(TOP_PLAYERS=('TOP_PLAYERS', lambda x: ', <br>'.join(x)))

    st.write(result_df)

    fig=px.bar(pd.merge(df1.groupby(['YEAR']).agg(MEDIAN=(metric_select,'median')).reset_index(), result_df, left_on='YEAR',right_on='YEAR'),
               x='YEAR',
               y='MEDIAN',
               title=f'Median {metric_select} by Draft Year',
               template='plotly_white',
               hover_data=['TOP_PLAYERS']
               )

    c2.plotly_chart(fig, use_container_width=True)

    st.write(df.head(3))

    # sorted_df = df.sort_values(['YEAR',metric_select],ascending=[True,False])
    # st.write(sorted_df.head(100))


    df.sort_values(['YEAR', metric_select, 'PTS'], ascending=[True, False, False], inplace=True)

    # Reset the index to ensure a continuous index range
    df.reset_index(drop=True, inplace=True)

    # Create a new column RANK and initialize it with 1
    df['REDRAFT'] = 1

    # Iterate over the DataFrame and update the RANK based on the conditions
    current_year = df.loc[0, 'YEAR']
    rank_counter = 0
    for i in range(len(df)):
        if df.loc[i, 'YEAR'] == current_year:
            rank_counter += 1
        else:
            current_year = df.loc[i, 'YEAR']
            rank_counter = 1
        df.loc[i, 'REDRAFT'] = rank_counter

    df['REDRAFT_DIFF'] = abs(df['PK'] - df['REDRAFT'])




    st.write(df)
    st.write(
        df.groupby(['YEAR']).agg(AVG_REDRAFT_DIFF=('REDRAFT_DIFF','mean')).reset_index()
             )

    year_list = df.YEAR.unique().astype('str')
    year_list=np.insert(year_list,0,'')
    year = st.selectbox('Select a year to view the draft - ',year_list,0)

if __name__ == "__main__":
    #execute
    app()
