import pandas as pd
import numpy as np
import plotly_express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog'
    )

@st.cache(suppress_st_warning=True)
def load_nba_data():
    df = pd.read_csv('./data/nba_year_stats.csv')
    df=df.loc[df.Year >= 1952]
    df=df.loc[df.MP > 100]
    df=df.loc[df.All_Stat_PM.notnull()]
    df=df.reset_index(drop=True)
    df['Year'] = df['Year'].astype(np.int64)
    df=df.round({'PPG': 1, 'APG':1, 'RPG':1,'BPG':1,'SPG':1,'PPM':1,'All_Stat_PM':1})
    return df


def cluster_nba(df, d, clusters):
    x_scaled = scaler.fit_transform(d)
    pca = PCA(n_components=8)
    pca.fit(x_scaled)

    x_pca = pca.transform(x_scaled)

    kmeans = KMeans(n_clusters=clusters, random_state=2).fit_predict(x_pca)

    df['Cluster'] = kmeans.astype('str')
    df['Cluster_x'] = x_pca[:,0]
    df['Cluster_y'] = x_pca[:,1]

    df=df.sort_values('Cluster',ascending=True)
    return df


def scatter_nba_clusters(df, title, hover_data):
    fig=px.scatter(df,
                   x='Cluster_x',
                   y='Cluster_y',
                   color='Cluster',
                   title=title,
                   hover_data=hover_data)
    fig.update_traces(mode='markers',marker=dict(
        size=8,
        opacity=.8,
        line=dict(
            width=1,
            color='DarkSlateGrey')))
    #fig.show()
    st.plotly_chart(fig)


def nba_cluster_by_season(df, clusters, year_min, year_max):
    #df=load_nba_data()
    df = df.loc[(df.Year >= year_min) & (df.Year <= year_max)]
    d = df.drop(['Rk','Player','Pos','Age','Tm','Year','Key_Player'],axis=1)
    df = cluster_nba(df, d, clusters)
    title = 'Clustering NBA Player Seasons by All Stats'
    hover_data = ['Player','Year','Age','Tm','PPG','RPG','APG','BPG','SPG']
    scatter_nba_clusters(df, title, hover_data)

def nba_cluster_by_career(df, clusters):
    df=df.groupby('Player').agg({'Rk':'size','Age':'median','G':'sum','GS':'sum','MP':'sum','FG':'sum','FGA':'sum','3P':'sum','3PA':'sum','2P':'sum','2PA':'sum','eFG%':'mean','FT':'sum','FTA':'sum','ORB':'sum','DRB':'sum','TRB':'sum','AST':'sum','STL':'sum','BLK':'sum','TOV':'sum','PTS':'sum','All_Stat':'sum','PPG':'mean','RPG':'mean','APG':'mean'}).reset_index()
    df['3PPG'] = df['3P']/df['G']
    df['2PPG'] = df['2P']/df['G']
    df['FTPG'] = df['FT']/df['G']
    df['SPG'] = df['STL']/df['G']
    df['BPG'] = df['BLK']/df['G']
    df['TPG'] = df['TOV']/df['G']
    df['All_Stat_PG'] = df['All_Stat']/df['G']
    df=df.round({'PPG': 1, 'APG':1, 'RPG':1,'BPG':1,'SPG':1,'PPM':1,'All_Stat_PM':1})
    clusters=7
    d = df.drop(['Rk','Player','Age'],axis=1)
    df = cluster_nba(df, d, clusters)
    title = 'Clustering NBA Player Careers by All Stats'
    hover_data = ['Player','G','MP','PTS','PPG','TRB','RPG','AST','APG','BLK','BPG','STL','SPG']
    scatter_nba_clusters(df, title, hover_data)

def nba_cluster_by_career_per_game(df, clusters):

    df=df.groupby('Player').agg({'Rk':'size','Age':'median','G':'sum','GS':'sum','MP':'sum','FG':'sum','FGA':'sum','3P':'sum','3PA':'sum','2P':'sum','2PA':'sum','eFG%':'mean','FT':'sum','FTA':'sum','ORB':'sum','DRB':'sum','TRB':'sum','AST':'sum','STL':'sum','BLK':'sum','TOV':'sum','PTS':'sum','All_Stat':'sum','PPG':'mean','RPG':'mean','APG':'mean'}).reset_index()
    df['3PPG'] = df['3P']/df['G']
    df['2PPG'] = df['2P']/df['G']
    df['FTPG'] = df['FT']/df['G']
    df['SPG'] = df['STL']/df['G']
    df['BPG'] = df['BLK']/df['G']
    df['TPG'] = df['TOV']/df['G']
    df['All_Stat_PG'] = df['All_Stat']/df['G']
    df = df.loc[df.MP > 0]
    df=df.round({'PPG': 1, 'APG':1, 'RPG':1,'BPG':1,'SPG':1,'PPM':1,'All_Stat_PM':1})

    clusters=7
    d = df.drop(['Player','Rk','Age','G','GS','MP','FG','FGA','3P','3PA','2P','2PA','FT','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PTS','All_Stat'],axis=1)
    df = cluster_nba(df, d, clusters)
    title = 'Clustering NBA Player Careers by Per Game Stats'
    hover_data = ['Player','PPG','RPG','APG','BPG','SPG']
    scatter_nba_clusters(df, title, hover_data)


def hide_footer():
    hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)

def app():
    df=load_nba_data()
    options = ['Clustering NBA Player Seasons by All Stats',
           'Clustering NBA Player Careers by All Stats',
          'Clustering NBA Player Careers by Per Game Stats']

    st.image('./images/nba_header.png',caption='See if superstar MJ, assist-legend Stockton, and rim-protection, finger-wagging Dikembe end up in different clusters', use_column_width=True)

    st.write("""
    # NBA Clusters
    The goal of this page is to take the yearly stats for a period of time (beginning and ending year below), select the # of clusters and group players together based on their yearly stats.

    The result should group similar types of players together...likely there will be a superstar cluster that dominate all stats, a big-man cluster that pulls tons of rebounds, and a guard cluster that contributes tons of assists.

    Play around with different year combos and see the results below!""")
    st.write('---')
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=nba_clusters&ea=nba_clusters">',unsafe_allow_html=True)


    year = df.Year.unique()
    year_min = st.selectbox('Select beginning year - ',year,0)
    year_max = st.selectbox('Select ending year - ',[row for row in year if row >= year_min],len([row for row in year if row >= year_min])-1)
    # clusters = st.selectbox('Number of clusters',[2,3,4,5,6,7,8,9,10,11],5)
    clusters = st.slider('Number of clusters',2,12,5)

    if st.button('Go!'):
        nba_cluster_by_season(df, clusters, year_min, year_max)
        #st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=nba_clusters&ea=year_min_'+str(year_min)+'_year_max_'+str(year_max)+'_clusters_'+str(clusters)+">',unsafe_allow_html=True)
        st.write('---')
        st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=nba_clusters&ea=year_min_'+str(year_min)+'_year_max_'+str(year_max)+'_clusters_'+str(clusters)+'">',unsafe_allow_html=True)

        st.write("""
        ## Table of NBA Stats
        Here is a table of all of the NBA Stats that are going into the clustering algorithms:
        """)
        st.write(df[['Year','Player','Pos','Age','Tm','PPG','RPG','APG','BPG','SPG','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','All_Stat_PM']].sort_values('All_Stat_PM',ascending=False))
        st.write("""
        ## Clustering Players Based on Entire Career
        As opposed to just looking at seasons, you can also cluster based on entire NBA careers. This view shows all stats, so players that put up long, successful careers like Karl Malone and Kareem really stand out.
        """)
        nba_cluster_by_career(df, clusters)
        st.write("""
        ## Clustering Players Based on Entire Career Per Game
        As opposed to looking at entire careers, you can also look at per game totals. This way, younger players that haven't completed their career can still have an impact. Or someone like Wilt who just demolished PPG & RPG in his career really jumps off the chart.
        """)
        nba_cluster_by_career_per_game(df, clusters)


if __name__ == "__main__":
    #execute
    hide_footer()
    main()








    # selection = st.sidebar.radio("", ('About Me','Work Experience','Projects','Data - Stocks','Data - Coronavirus','Data - NBA Clusters'))
    #
    # if selection == 'About Me':
    #     about()
    # if selection == 'Work Experience':
    #     experience()
    # if selection == 'Projects':
    #     projects()
    # if selection == 'Data - Stocks':
    #     stocks()
    # if selection == 'Price Tracker':
    #     price_tracker()
    # if selection == 'Data - Coronavirus':
    #     coronavirus()
    # if selection == 'Data - NBA Clusters':
    #     nba_clusters()
