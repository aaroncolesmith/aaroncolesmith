import pandas as pd
import streamlit as st
import numpy as np
import plotly_express as px
import plotly.io as pio
pio.templates.default = "simple_white"

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




def app():
    st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog',
    layout='wide'
    )

    awards_df=pd.read_parquet('https://github.com/aaroncolesmith/data_sources/raw/main/awards_df.parquet', engine='pyarrow')
    c1,c2 = st.columns(2)
    season_l=awards_df.sort_values('Season',ascending=False).Season.unique().tolist()
    season = c1.selectbox('Select season', season_l)

    award_l=awards_df.query("Season in @season").Award.unique().tolist()
    award = c2.selectbox('Select award',award_l)

    df = awards_df.query("Season in @season & Award in @award")
    df['Voter'] = df['Voter'].str.split(',').str[::-1].str.join(' ')


    all_null_columns = df.columns[df.isnull().all()]

    # Drop columns with all null values
    df = df.drop(all_null_columns, axis=1)

    place_cols = [col for col in df.columns if 'Place' in col]

    for col in place_cols:
        df[col] = df[col].str.strip()

        if season == '2021-22':
            df[col] = df[col].str.replace(')','')
    
    # st.write(df)

    df=pd.merge(df,pd.get_dummies(df,
                columns=place_cols))

    columns=df.columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']
    num_columns = df.select_dtypes(include=numerics).columns.tolist()


    d=df.copy()
    d=d[num_columns]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(d)

    pca = PCA(n_components=2)
    pca.fit(x_scaled)

    x_pca = pca.transform(x_scaled)

    kmeans = KMeans(n_clusters=7, random_state=2).fit_predict(x_pca)

    p=pd.DataFrame(np.transpose(pca.components_[0:2, :]))
    p=pd.merge(p,pd.DataFrame(np.transpose(d.columns)),left_index=True,right_index=True)
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

    kmeans = KMeans(n_clusters=8, random_state=2).fit_predict(x_pca)

    p['Cluster'] = kmeans.astype('str')
    p['Cluster_x'] = x_pca[:,0]
    p['Cluster_y'] = x_pca[:,1]
    p['Cluster'] = pd.to_numeric(p['Cluster'])

    pviz=p.groupby(['Cluster']).agg({'field' : lambda x: ', '.join(x),'x':'mean','y':'mean'}).reset_index()

    pviz['field']=pviz.field.str.replace('\(10 points\)_','').str.replace('\(7 points\)_','').str.replace('\(5 points\)_','').str.replace('\(3 points\)_','').str.replace('\(1 point\)_','')
    pviz.field = pviz.field.str.wrap(50)
    pviz.field = pviz.field.apply(lambda x: x.replace('\n', '<br>'))

    # Calculate mean of x and y
    mean_x = p['x'].mean()
    mean_y = p['y'].mean()

    # Calculate distance
    p['distance'] = np.sqrt((p['x'] - mean_x)**2 + (p['y'] - mean_y)**2)

    c1,c2=st.columns(2)
    voter_list=df.Voter.unique()
    voter_list=np.insert(voter_list,0,'')
    voter_select=c1.selectbox('Highlight a specific voter: ',voter_list)

    votes_points=c2.selectbox('View Bar Graph by Votes or Points?',['Votes','Points'])

    dviz=df.groupby(['Cluster','Cluster_x','Cluster_y']+place_cols).agg(Voter_Count=('Voter','nunique'),
                                                                                            Voters=('Voter', lambda x: '|'.join(x),)).reset_index()

    def add_line_breaks(name):
        result = ''
        counter = 0
        for char in name:
            result += char
            if char == '|' and counter > 50:
                result += '<br>'
                counter=0
            counter += 1
        return result

    # Apply the function to the 'Names' column
    dviz['Voters'] = dviz['Voters'].apply(add_line_breaks)
    dviz['Voters'] = dviz['Voters'].str.replace('|',', ').str.replace('  ',' ').str.replace('<br> ','<br>')

    fig=px.scatter(dviz,
                    x='Cluster_x',
                    y='Cluster_y',
                width=800,
                size='Voter_Count',
                color_discrete_sequence=['skyblue','red'],
                opacity=np.where(dviz['Voters'].str.contains(voter_select),0.9, 0.2),
                    hover_data=['Voters']+place_cols,
                    title='Clustering Voting Results for '+season+' - '+award)
    fig.update_traces(textposition='top center',marker=dict(
                                                            line=dict(width=1,
                                                                      color=np.where(dviz['Voters'].str.contains(voter_select),'DarkSlateGray', 'gray')
                                                                      )))
    fig.update_layout(legend_title_text='Cluster')

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if len(place_cols) == 5:
        fig.update_traces(hovertemplate='<b>Voters:</b> %{customdata[0]}<br><b>Ballot:</b> 1. %{customdata[1]} 2. %{customdata[2]} <br>3. %{customdata[3]} 4. %{customdata[4]}<br>5. %{customdata[5]}')
    if len(place_cols) == 3:
        fig.update_traces(hovertemplate='<b>Voters:</b> %{customdata[0]}<br><b>Ballot:</b> 1. %{customdata[1]} 2. %{customdata[2]} <br>3. %{customdata[3]}')


    x_factor = (dviz.Cluster_x.max() / p.x.max())*.75
    y_factor = (dviz.Cluster_y.max() / p.y.max())*.75

    p['x'] = p['x'].round(2)
    p['y'] = p['y'].round(2)
    # st.write(p.sort_values('distance',ascending=False).head(10).groupby(['x','y',]).agg(field=('field', lambda x: '<br>'.join(x))).reset_index())

    for i, r in p.sort_values('distance',ascending=False).head(10).groupby(['x','y',]).agg(field=('field', lambda x: '<br>'.join(x))).reset_index().iterrows():
        x_val = r['x']*x_factor
        y_val = r['y']*y_factor

        if x_val < dviz.Cluster_x.min():
            x_val = dviz.Cluster_x.min()
        if x_val > dviz.Cluster_x.max():
            x_val = dviz.Cluster_x.max()

        if y_val < dviz.Cluster_y.min():
            y_val = dviz.Cluster_y.min()
        if y_val > dviz.Cluster_y.max():
            y_val = dviz.Cluster_y.max()


        fig.add_annotation(
            x=x_val,
            y=y_val,
            text=r['field'],
            showarrow=False,
            opacity=.5,
            font=dict(
                        color="black",
                        size=8)
        )
    config = {'displayModeBar': False}

   
    c1,c2 = st.columns(2)
    c1.plotly_chart(fig, config=config, use_container_width=True)


    vote_df = pd.DataFrame()

    for col in place_cols:
        tmp_df = df[['Voter',col]]
        tmp_df.columns = ['Voter','Player']
        tmp_df['Vote'] = col
        vote_df = pd.concat([vote_df,tmp_df])

    
    vote_df1 = vote_df.groupby(['Player','Vote']).agg(Votes=('Voter','size')).reset_index().sort_values(['Player','Vote'],ascending=True)
    if len(place_cols) == 5:
        vote_df1.loc[vote_df1.Vote == '1st Place','Points'] = vote_df1['Votes']*10
        vote_df1.loc[vote_df1.Vote == '2nd Place','Points'] = vote_df1['Votes']*7
        vote_df1.loc[vote_df1.Vote == '3rd Place','Points'] = vote_df1['Votes']*5
        vote_df1.loc[vote_df1.Vote == '4th Place','Points'] = vote_df1['Votes']*3
        vote_df1.loc[vote_df1.Vote == '5th Place','Points'] = vote_df1['Votes']*1
    if len(place_cols) == 3:
        vote_df1.loc[vote_df1.Vote == '1st Place','Points'] = vote_df1['Votes']*5
        vote_df1.loc[vote_df1.Vote == '2nd Place','Points'] = vote_df1['Votes']*3
        vote_df1.loc[vote_df1.Vote == '3rd Place','Points'] = vote_df1['Votes']*1

    vote_df1['Count_by_Vote'] = vote_df1['Vote'].str.split(' ',expand=True)[0] + ' - ' + vote_df1['Votes'].astype('str')
    vote_df1 = vote_df1.groupby(['Player']).agg(Count_By_Vote = ('Count_by_Vote', lambda x: ', '.join(x)),
                                                Points=('Points','sum')).reset_index()
    vote_df1 = pd.merge(vote_df1, vote_df.groupby(['Player']).agg(Votes=('Voter','size')).reset_index().sort_values('Votes',ascending=False), left_on='Player',right_on='Player')

    # if votes_points == 'Votes':
    #     fig=px.bar(
    #         vote_df1,
    #             x='Votes',
    #             y='Player',
    #             title='Total Number of Votes by Player',
    #             hover_data=['Count_By_Vote'],
    #                 color_discrete_sequence=['skyblue','red'],
    #             orientation='h')
    #     fig.update_traces(marker_line_color='DarkSlateGray',marker_line_width=1.5, opacity=0.6)
    #     fig.update_traces(hovertemplate='<b>%{y}:</b> %{x} Votes<br><b># of Votes by Place:</b> %{customdata[0]}')
    #     fig.update_yaxes(title='Player', categoryorder='total ascending')
    #     fig.update_yaxes(title='Total Number of Votes')
    #     fig.update_layout({
    #         'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    #         'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    #         })
    #     c2.plotly_chart(fig, config=config, use_container_width=True)
    # if votes_points == 'Points':
    #     fig=px.bar(
    #         vote_df1,
    #             x='Points',
    #             y='Player',
    #             title='Total Number of Points by Player',
    #             hover_data=['Count_By_Vote'],
    #                 color_discrete_sequence=['skyblue','red'],
    #             orientation='h')
    #     fig.update_traces(marker_line_color='DarkSlateGray',marker_line_width=1.5, opacity=0.6)
    #     fig.update_traces(hovertemplate='<b>%{y}:</b> %{x} Points<br><b># of Votes by Place:</b> %{customdata[0]}')
    #     fig.update_yaxes(title='Player', categoryorder='total ascending')
    #     fig.update_yaxes(title='Total Number of Points')
    #     fig.update_layout({
    #         'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    #         'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    #         })
    #     c2.plotly_chart(fig, config=config, use_container_width=True)

    column_mapping = {
        'Votes': {
            'x': 'Votes',
            'title': 'Total Number of Votes by Player',
            'xaxis_title': 'Total Number of Votes',
            'hovertemplate': '<b>%{y}:</b> %{x} Votes<br><b># of Votes by Place:</b> %{customdata[0]}'
        },
        'Points': {
            'x': 'Points',
            'title': 'Total Number of Points by Player',
            'xaxis_title': 'Total Number of Points',
            'hovertemplate': '<b>%{y}:</b> %{x} Points<br><b># of Votes by Place:</b> %{customdata[0]}'
        }
    }


    c2.plotly_chart(fig, config=config, use_container_width=True)


if __name__ == "__main__":
    #execute
    app()