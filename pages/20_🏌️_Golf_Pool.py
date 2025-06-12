import streamlit as st
import pandas as pd
import plotly_express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json
from datetime import datetime, timedelta



st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog',
    layout='wide'
    )


def quick_clstr(df, num_cols, str_cols, color):
    df1=df.copy()
    df1=df1[num_cols]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df1)

    pca = PCA(n_components=2)
    pca.fit(x_scaled)

    x_pca = pca.transform(x_scaled)
    
    if df1.index.size < 10:
        n_clusters = 2
    else:
        n_clusters=5

    kmeans = KMeans(n_clusters=n_clusters, random_state=2).fit_predict(x_pca)

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
    dvz=pd.concat([dvz,p.sort_values('x',ascending=True).head(4)])
    dvz=pd.concat([dvz,p.sort_values('x',ascending=True).tail(4)])
    dvz=pd.concat([dvz,p.sort_values('y',ascending=True).head(4)])
    dvz=pd.concat([dvz,p.sort_values('y',ascending=True).tail(4)])
    dvz=dvz.drop_duplicates()

    # key_vals=p.sort_values('distance_from_zero',ascending=False).head(20).field.tolist()
    key_vals=dvz.field.tolist()

    df['Cluster'] = df['Cluster'].astype('str')

    fig=px.scatter(df,
#         df.sort_values(color,ascending=True),
                   x='Cluster_x',
                   y='Cluster_y',
                   width=800,
                   height=700,
                   template='simple_white',
                   color=color,
                   hover_data=str_cols,
                   text='Team'
                  )
    fig.update_layout(legend_title_text=color)

    # fig.update_layout({
    #     'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    #     'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    #     })


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
                opacity=.2,
                font=dict(
                    family="Courier New, monospace",
                    color="black",
                    size=12
                    )
                )
    fig.update_traces(mode='markers+text',
                      textposition='top center',
                      opacity=.75,
                      marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey'))
                      )
    fig.update_xaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    fig.update_yaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    # st.plotly_chart(fig,use_container_width=True)
    return fig

    df['Cluster'] = pd.to_numeric(df['Cluster'])
    


def tourney(df,df2,df_picks):
    try:
        df2=pd.merge(df2,df[['PLAYER','SCORE','THRU']],
                    left_on='Golfer',
                    right_on='PLAYER',
                    how='left')
    except:
        df2=pd.merge(df2,df[['PLAYER','SCORE']],
                    left_on='Golfer',
                    right_on='PLAYER')
        df2['THRU'] = 'DONE'
    # st.write(df2)
    df2['SCORE_NUM'] = pd.to_numeric(df2['SCORE'].replace('E','0').replace('WD',50).replace('CUT',50).replace('-',0))
    df2=df2.sort_values(['SCORE_NUM','Pick'],ascending=[True,True])
    df3=df2.groupby('Team').head(4)

    df3=df3.groupby(['Team']).agg(
        GOLFERS=('Golfer',lambda x: ', '.join(x.unique())),
        SCORE=('SCORE_NUM','sum')
    ).sort_values('SCORE',ascending=True).reset_index(drop=False)

    fig=px.bar(df3,
               y='Team',
               x='SCORE',
               orientation='h',
               hover_data=['GOLFERS'],
               text_auto=',.4',
               
               )
    fig.update_layout(yaxis=dict(autorange="reversed"))

    tab1,tab2,tab3,tab4=st.tabs(['Pool Leaderboard','Masters Leaderboard','Team Picks','Pick Similarity'])

    with tab1:
        c1,c2,c3=st.columns([1,5,1])
        c2.plotly_chart(fig,use_container_width=True)
        c2.dataframe(df3,hide_index=True)
    with tab2:
        c1,c2,c3=st.columns([1,5,1])
        df2_agg=df2.groupby(['Golfer']).agg(
            PICKS=('Team','size'),
            PICKED_BY=('Team',lambda x: ', '.join(x.unique())),
            SCORE=('SCORE_NUM','median')).sort_values('PICKS',ascending=False).reset_index()
        try:
            c2.dataframe(pd.merge(df,df2_agg[['Golfer','PICKED_BY']], left_on='PLAYER',right_on='Golfer', how='left')[['POS','PLAYER','SCORE','TODAY','THRU','R1','R2','R3','R4','TOT','PICKED_BY']]
                        ,hide_index=True,
                        column_config={
                        "PICKED_BY": st.column_config.TextColumn(
                        "Picked By:",
                        width='large')
                        }
                        )
        except:
            c2.dataframe(pd.merge(df,df2_agg[['Golfer','PICKED_BY']], left_on='PLAYER',right_on='Golfer', how='left')[['POS','PLAYER','SCORE','R1','R2','R3','R4','TOT','PICKED_BY']]
                        ,hide_index=True,
                        column_config={
                        "PICKED_BY": st.column_config.TextColumn(
                        "Picked By:",
                        width='large')
                        }
                        )
    with tab3:
        team_list = df2.sort_values('Team',ascending=True).Team.unique().tolist()
        cols = st.columns(4)

        url='https://datagolf.com/live-model/pga-tour'
        r=requests.get(url)
        url_str = r.text
        content = url_str.split('response = JSON.parse(')[1].split('\'.replace(/\\')[0]
        data=json.loads(content[1:])
        df_ld = pd.DataFrame(data['main'])

        # Function to convert "Last, First" to "First Last"
        def convert_name(name):
            parts = name.split(',')
            return parts[1].strip() + ' ' + parts[0].strip()

        # Apply the function to the 'name' column
        df_ld['name'] = df_ld['name'].apply(convert_name)

        df_ld = df_ld[['name','cut','top5','win']]
        df_ld.columns = ['Golfer','Cut %','Top 5 %','Win %']
        df_ld['Golfer'] = df_ld['Golfer'].str.replace('Byeong Hun An','Byeong-Hun An').str.replace('Alex Noren','Alexander Noren').str.replace('Macintyre','MacIntyre')

        df2 = pd.merge(df2,df_ld,how='left')


        data_cols = ['Pick','Golfer','SCORE','THRU','Cut %']
        # st.write(df3)
        team_list = df3['Team'].tolist()
        ncol = len(team_list)
        wcol = 4  # Number of columns in each row

        # Calculate the number of rows needed
        nrow = (ncol + wcol - 1) // wcol

        for r in range(nrow):
            # Determine the number of columns for the current row
            teams_left = ncol - r * wcol
            cols = st.columns(min(teams_left, wcol))
            
            for i, col in enumerate(cols):
                team_ix = r * wcol + i
                if team_ix < ncol:
                    team = team_list[team_ix]
                    score = df3.loc[df3.Team == team, 'SCORE'].values[0]
                    score = '+' + str(score) if score > 0 else str(score)

                    avg_cut = round(df2.loc[df2.Team==team]['Cut %'].mean()*100,1)

                    col.write(f'{team} ({score}) - Avg Cut: {avg_cut}%')

                    # Create a copy to avoid modifying the original DataFrame if needed elsewhere
                    df2_display = df2.loc[df2.Team==team][data_cols].copy()

                    # Multiply the "Cut %" column by 100
                    df2_display["Cut %"] = df2_display["Cut %"] * 100

                    # Create the dataframe, configuring the "Cut %" column
                    col.dataframe(
                        df2_display,
                        hide_index=True,
                        column_config={
                            "Cut %": st.column_config.NumberColumn(
                                "Cut %",  # Column title displayed
                                format="%.1f %%",  # Format as a percentage with 2 decimal places
                            )
                        }
                    )
                    
                    # col.dataframe(df2.loc[df2.Team==team][data_cols],hide_index=True)



        # for i,x in enumerate(cols):
        #     team = team_list[i]
        #     score = df3.loc[df3.Team == team, 'SCORE'].values[0]
        #     score = '+' + str(score) if score > 0 else str(score)

        #     avg_cut = round(df2.loc[df2.Team==team_list[i]]['Cut %'].mean()*100,1)

        #     x.write(f'{team} ({score}) - Avg Cut: {avg_cut}%')
        #     x.dataframe(df2.loc[df2.Team==team_list[i]][data_cols],hide_index=True)
        
        # cols = st.columns(4)
        # for i,x in enumerate(cols):
        #     team = team_list[i+4]
        #     score = df3.loc[df3.Team == team, 'SCORE'].values[0]
        #     score = '+' + str(score) if score > 0 else str(score)
        #     avg_cut = round(df2.loc[df2.Team==team_list[i+4]]['Cut %'].mean()*100,1)
        #     x.write(f'{team} ({score}) - Avg Cut: {avg_cut}%')
        #     x.dataframe(df2.loc[df2.Team==team_list[i+4]][data_cols],hide_index=True)

        # cols = st.columns(4)
        # for i,x in enumerate(cols):
        #     team = team_list[i+8]
        #     score = df3.loc[df3.Team == team, 'SCORE'].values[0]
        #     score = '+' + str(score) if score > 0 else str(score)
        #     avg_cut = round(df2.loc[df2.Team==team_list[i+8]]['Cut %'].mean()*100,1)
        #     x.write(f'{team} ({score}) - Avg Cut: {avg_cut}%')
        #     x.dataframe(df2.loc[df2.Team==team_list[i+8]][data_cols],hide_index=True)
        
        # cols = st.columns(1)
        # for i,x in enumerate(cols):
        #     team = team_list[i+12]
        #     score = df3.loc[df3.Team == team, 'SCORE'].values[0]
        #     score = '+' + str(score) if score > 0 else str(score)
        #     avg_cut = round(df2.loc[df2.Team==team_list[i+12]]['Cut %'].mean()*100,1)
        #     x.write(f'{team} ({score}) - Avg Cut: {avg_cut}%')
        #     x.dataframe(df2.loc[df2.Team==team_list[i+12]][data_cols],hide_index=True)

    with tab4:
        c1,c2=st.columns(2)
        c1.markdown('##### How often each golfer was picked')
        fig=px.bar(df2.groupby(['Golfer']).agg(
            PICKS=('Team','size'),
            TEAMS=('Team',lambda x: ', '.join(x.unique())),
            SCORE=('SCORE_NUM','median')).sort_values('PICKS',ascending=False).reset_index(),
               x='PICKS',
               y='Golfer',
               color='SCORE',
               orientation='h',
               hover_data=['TEAMS'],
               height=800
               )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        c1.plotly_chart(fig,use_container_width=True)

        df_encoded = pd.get_dummies(df_picks, columns=['Pick1','Pick2','Pick3','Pick4','Pick5','Pick6']
                                        , drop_first=False)
        df_picks = pd.concat([df_picks, df_encoded], axis=1) 
        df_picks = df_picks.loc[:, ~df_picks.columns.duplicated()]


        num_cols=df_picks.columns.tolist()[8:]
        str_cols=['Team','Pick1','Pick2','Pick3','Pick4','Pick5','Pick6']

        df_picks = pd.merge(df_picks,df3,)
        color = c2.selectbox(
            'How to color the graph',
            ('Team','SCORE')
        )
        c2.markdown('##### Similarity of picks for each team')
        fig=quick_clstr(df_picks, num_cols, str_cols, color)
        c2.plotly_chart(fig,use_container_width=True)

def get_prob(a):
    odds = 0
    if a < 0:
        odds = (-a)/(-a + 100)
    else:
        odds = 100/(100+a)

    return odds

def get_action_network_odds():
    headers = {
        'Authority': 'api.actionnetwork',
        'Accept': 'application/json',
        'Origin': 'https://www.actionnetwork.com',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
    }
    url='https://api.actionnetwork.com/web/v1/scoreboard/pga?period=game&bookIds=15,30,76,75,123,69,68,972,71,247,79'
    r=requests.get(url,headers=headers)
    df_golfers = pd.json_normalize(r.json()['competitions'][0]['competitors'])
    df_odds = pd.json_normalize(r.json()['competitions'][0]['tournament_odds'][0]['competitors'])
    df = pd.merge(df_golfers, df_odds, left_on='id', right_on='competitor_id').sort_values('moneyline',ascending=True).reset_index(drop=True)
    df['implied_probability'] = df['moneyline'].apply(get_prob)
    df['timestamp']=pd.Timestamp.now(tz='US/Eastern')
    return df[['timestamp','full_name','moneyline','implied_probability','player.image']]


def upload_s3_data(df, filename):

    # for x in ['date','start_time_et','model_run']:
    #     try:
    #         df[x]=pd.to_datetime(df[x])
    #     except:
    #        df[x]=df[x].apply(lambda x: pd.to_datetime(x).tz_convert('US/Eastern'))

    # for x in ['id','score_home','score_away','spread_home','spread_away','spread_home_public','spread_away_public','num_bets','ttq','updated_spread_diff','fav_wins_pred','fav_wins_bin','confidence_level','fav_wins','ensemble_model_win']:
    #     try:
    #        df[x] = df[x].replace('nan',np.nan).apply(lambda x: pd.to_numeric(x))
    #     except Exception as e:
    #        print(e)

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
    # st.write('data uploaded')

@st.cache_data(ttl=300)
def get_s3_data(filename):
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read(f"bet-model-data/{filename}.parquet", 
                input_format="parquet", 
                ttl=600
                )
    return df



def app():
    st.title('Golf Pool')
    ## This is to handle a playoff -- may need to fix this 0
    try:
        df=pd.read_html('https://www.espn.com/golf/leaderboard')[1]
    except:
        try:
            df=pd.read_html('https://www.espn.com/golf/leaderboard')[0]
        except:
            st.write('tourney not started')
    try:
        del df['Unnamed: 0']
    except:
        pass

    url='https://docs.google.com/spreadsheets/d/1DYnvfi7uzaOPVz2_gThJLyZQKI6YNID0a6jH4-yw0LY/gviz/tq?tqx=out:csv&gid=0'
    df_picks=pd.read_csv(url, on_bad_lines='skip')
    df_picks.columns=['timestamp','Team', 'Pick1', 'Pick2', 'Pick3', 'Pick4', 'Pick5', 'Pick6']
    df_picks['timestamp'] = pd.to_datetime(df_picks['timestamp'])
    # Get the current date
    current_date = datetime.now()
    # Calculate the date range
    start_date = current_date - timedelta(days=6)
    end_date = current_date + timedelta(days=6)
    df_picks = df_picks[(df_picks['timestamp'] >= start_date) & (df_picks['timestamp'] <= end_date)]


    for col in ['Pick1', 'Pick2', 'Pick3', 'Pick4', 'Pick5', 'Pick6']:
        df_picks[col] = df_picks[col].str.replace('\d+', '',regex=True).str.replace(' `/','').str.replace(' /','').str.replace(' >','').str.strip().str.replace(' .%','').str.replace('Mcilroy','McIlroy').str.replace('Macintyre','MacIntyre').str.replace(' $,','').str.replace('$','').str.replace(',','').str.strip()
    
    df['PLAYER'] = df['PLAYER'].str.replace('Ludvig Åberg','Ludvig Aberg').str.replace(' \(a\)','',regex=True).str.replace('Brandon Robinson Thompson','Brandon Robinson-Thompson').str.replace('Byeong Hun An','Byeong-Hun An').str.replace('Alex Noren','Alexander Noren').str.replace('Nicolai Højgaard','Nicolai Hojgaard').str.replace('Joaquín','Joaquin').str.replace('Cam Davis','Cameron Davis').str.replace('Emilio González R.','Emilio Gonzalez').str.replace('Cameron Davis','Cam Davis')
    

    df2 = pd.melt(df_picks,
                     id_vars=['Team'],
                     value_vars=['Pick1', 'Pick2', 'Pick3', 'Pick4', 'Pick5', 'Pick6'],
                     var_name='Pick', 
                     value_name='Golfer'
                     )


    if 'SCORE' in df.columns:
        tourney(df,df2,df_picks)
    else:
        st.write('tourney not started')
        st.write('Submitted teams')
        st.dataframe(df_picks[['Team']],hide_index=True)
        st.write('Leaderboard Tee Times')
        st.write(df)
        # st.write('df2')
        # st.write(df2)

        df_merge = pd.merge(df,df2,left_on='PLAYER',right_on='Golfer',how='right').sort_values(['Team','Pick'],ascending=[True,True])
        # st.write(df_merge)

                
        # # Get the list of teams



        # df['implied_probability'] = df['implied_probability']*100
        # st.dataframe(df[['PLAYER','TEE TIME','moneyline','implied_probability']],
        #              column_config={
        #                 "implied_probability": st.column_config.NumberColumn(
        #                 "Implied Probability",
        #                 # help="The sales volume in USD",
        #                 format="%.1f%%",
        #                 # min_value=0,
        #                 # max_value=100,
        #                 # width='small
        #                 )
        #              },
        #              hide_index=True)




    



if __name__ == "__main__":
    #execute
    app()


