import streamlit as st
import pandas as pd
import plotly_express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

import numpy as np


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
    dvz=pd.concat([dvz,p.sort_values('x',ascending=True).head(2)])
    dvz=pd.concat([dvz,p.sort_values('x',ascending=True).tail(2)])
    dvz=pd.concat([dvz,p.sort_values('y',ascending=True).head(2)])
    dvz=pd.concat([dvz,p.sort_values('y',ascending=True).tail(2)])
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
    


def tourney(df,df2):
    try:
        df2=pd.merge(df2,df[['PLAYER','SCORE','THRU']],
                    left_on='Golfer',
                    right_on='PLAYER')
    except:
        df2=pd.merge(df2,df[['PLAYER','SCORE']],
                    left_on='Golfer',
                    right_on='PLAYER')
        df2['THRU'] = 'DONE'        
    # st.write(df2)
    df2['SCORE_NUM'] = pd.to_numeric(df2['SCORE'].replace('E','0').replace('CUT',20))
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
        c2.dataframe(pd.merge(df,df2_agg[['Golfer','PICKED_BY']], left_on='PLAYER',right_on='Golfer', how='left')[['POS','PLAYER','SCORE','TODAY','THRU','R1','R2','R3','R4','TOT','PICKED_BY']]
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

        for i,x in enumerate(cols):
            team = team_list[i]
            score = df3.loc[df3.Team == team, 'SCORE'].values[0]
            score = '+' + str(score) if score > 0 else str(score)
            x.write(f'{team} ({score})')
            x.dataframe(df2.loc[df2.Team==team_list[i]][['Pick','Golfer','SCORE','THRU']],hide_index=True)
        
        cols = st.columns(4)
        for i,x in enumerate(cols):
            team = team_list[i+4]
            score = df3.loc[df3.Team == team, 'SCORE'].values[0]
            score = '+' + str(score) if score > 0 else str(score)
            x.write(f'{team} ({score})')
            x.dataframe(df2.loc[df2.Team==team_list[i+4]][['Pick','Golfer','SCORE','THRU']],hide_index=True)

        cols = st.columns(3)
        for i,x in enumerate(cols):
            team = team_list[i+8]
            score = df3.loc[df3.Team == team, 'SCORE'].values[0]
            score = '+' + str(score) if score > 0 else str(score)
            x.write(f'{team} ({score})')
            x.dataframe(df2.loc[df2.Team==team_list[i+8]][['Pick','Golfer','SCORE','THRU']],hide_index=True)

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


        num_cols=df_picks.columns.tolist()[7:]
        str_cols=['Team','Pick1','Pick2','Pick3','Pick4','Pick5','Pick6']

        df_picks = pd.merge(df_picks,df3,)
        color = c2.selectbox(
            'How to color the graph',
            ('Team','SCORE')
        )
        c2.markdown('##### Similarity of picks for each team')
        fig=quick_clstr(df_picks, num_cols, str_cols, color)
        c2.plotly_chart(fig,use_container_width=True)

def app():
    st.title('Golf Pool')
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
    for col in ['Pick1', 'Pick2', 'Pick3', 'Pick4', 'Pick5', 'Pick6']:
        df_picks[col] = df_picks[col].str.replace('\d+', '',regex=True).str.replace(' `/','').str.replace(' /','').str.replace(' >','')
    
    df2 = pd.melt(df_picks,
                     id_vars=['Team'],
                     value_vars=['Pick1', 'Pick2', 'Pick3', 'Pick4', 'Pick5', 'Pick6'],
                     var_name='Pick', 
                     value_name='Golfer'
                     )


    if 'SCORE' in df.columns:
        tourney(df,df2)
    else:
        st.write('tourney not started')
        st.write('Submitted teams')
        st.write(df_picks[['Team']])
        st.write('Leaderboard Tee Times')
        st.write(df)




    



if __name__ == "__main__":
    #execute
    app()


