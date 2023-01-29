import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog'
    )


def app():
    st.image('./images/clstr.png',use_column_width=True)
    ga('CLSTR', 'Page Load', 'Page Load')
    st.write('CLSTR is meant to take a multi-dimensional dataset and output a visualization with clusters to help quickly segment the data and provide some initial intelligence. Play around with some of the datasets available here or feel free to upload your own.')

    st.markdown('### Select a dataset -- or upload your own')

    df=pd.DataFrame()

    data = st.selectbox('Data',
    ('2020 NBA Playoffs','NBA Yearly Statistics','Iris Dataset','NBA 2K21', 'Upload CSV'))

    if data == '2020 NBA Playoffs':
        df=pd.read_csv('./data/2020_playoff_games.csv')

    if data == 'NBA 2K21':
        df=pd.read_csv('./data/2k_players.csv')

    if data == 'NBA Yearly Statistics':
        df=pd.read_csv('./data/nba_year_stats.csv')
        df = df.loc[df.MP > 0]

    if data == 'Iris Dataset':
        iris = load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                             columns= iris['feature_names'] + ['target'])
        df.loc[df.target==0,'species'] ='Iris Setosa'
        df.loc[df.target==1,'species'] ='Iris Versicolour'
        df.loc[df.target==2,'species'] ='Iris Virginica'
        del df['target']

    if data == 'Upload CSV':
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)

    ga('CLSTR', data, data)

    if df.size>0:

        st.markdown('### Preview of your data:')
        st.write(df.head(5))


        st.markdown('### Select which columns you\'d like to use as CLSTR features')
        columns=df.columns
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        num_columns = df.select_dtypes(include=numerics).columns.tolist()

        f = st.multiselect('Features', num_columns,num_columns)
        str_f = ''.join(f)

        if len(f) >= 2:
            clusters = st.slider('Number of clusters',2,12,5)
            ga('CLSTR', data, str_f)
            df, p = clstr(df, f, n_clusters=clusters)


            st.markdown('### Select columns to display on the graph as hover data')
            hov = st.multiselect('Hover Data',columns)
            hover=hov


            advanced = st.checkbox('Advanced mode: show features on plot?')

            if advanced:
                scatter(df,p,hover,show_features=1)
                ga('CLSTR', data, 'graph_drawn')
            else:
                scatter(df,p,hover,show_features=0)
                ga('CLSTR', data, 'graph_drawn')





        else:
            st.write('You have to select at least two columns')


def ga(event_category, event_action, event_label):
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec='+event_category+'&ea='+event_action+'&el='+event_label+'">',unsafe_allow_html=True)



def clstr(df, f, n_clusters=5):
    d=df.copy()

    d=d[f]
    d = d.fillna(0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(d)

    pca = PCA(n_components=2)
    pca.fit(x_scaled)

    x_pca = pca.transform(x_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=2).fit_predict(x_pca)

    p=pd.DataFrame(np.transpose(pca.components_[0:2, :]))
    p=pd.merge(p,pd.DataFrame(np.transpose(d.columns)),left_index=True,right_index=True)
    p.columns = ['x','y','field']

    df['Cluster'] = kmeans.astype('str')
    df['Cluster_x'] = x_pca[:,0]
    df['Cluster_y'] = x_pca[:,1]
    df['Cluster'] = pd.to_numeric(df['Cluster'])

    return df, p

def scatter(df,p,hover,show_features):
    fig=px.scatter(df,
                   x='Cluster_x',
                   y='Cluster_y',
                   color=df['Cluster'].astype('str'),
                   hover_data=hover)
    fig.update_traces(textposition='top center',marker=dict(size=8,opacity=.75,line=dict(width=1,color='DarkSlateGrey')))
    fig.update_layout(legend_title_text='Cluster')

    x_factor = df.Cluster_x.max() / p.x.max()
    y_factor = df.Cluster_y.max() / p.y.max()

    if show_features == 1:
        for i,r in p.iterrows():
            fig.add_annotation(
                x=r['x']*x_factor,
                y=r['y']*y_factor,
                text=r['field'],
            showarrow=False,
            bgcolor="white",
            opacity=.75)
    # fig.show()
    st.plotly_chart(fig)


if __name__ == "__main__":
    #execute
    app()
