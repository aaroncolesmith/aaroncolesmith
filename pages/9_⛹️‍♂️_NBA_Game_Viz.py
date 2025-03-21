import streamlit as st
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly_express as px
import random
from io import BytesIO


def quick_clstr(df, num_cols, str_cols, color):
    df1=df.copy()
    df1=df1[num_cols]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df1)

    pca = PCA(n_components=2)
    pca.fit(x_scaled)

    x_pca = pca.transform(x_scaled)

    kmeans = KMeans(n_clusters=5, random_state=2).fit_predict(x_pca)

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
    dvz=pd.concat([dvz,p.sort_values('x',ascending=True).head(5)])
    dvz=pd.concat([dvz,p.sort_values('x',ascending=True).tail(5)])
    dvz=pd.concat([dvz,p.sort_values('y',ascending=True).head(5)])
    dvz=pd.concat([dvz,p.sort_values('y',ascending=True).tail(5)])
    dvz=dvz.drop_duplicates()

    # key_vals=p.sort_values('distance_from_zero',ascending=False).head(20).field.tolist()
    key_vals=dvz.field.tolist()



    df[color] = df[color].astype('str')





    # Predefined list of hex colors
    hex_colors = ['#ffd900', '#ff2a00', '#35d604', '#59ffee', '#1d19ff','#ff19fb']

    # Function to generate a random hex color
    def generate_random_hex_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # Get unique values from the color column
    unique_values = df[color].unique()

    # If there are more unique values than predefined colors, generate additional random colors
    while len(hex_colors) < len(unique_values):
        hex_colors.append(generate_random_hex_color())

    # Create a dictionary to map unique values to colors (this is your discrete color map)
    discrete_color_map = {value: hex_colors[i] for i, value in enumerate(unique_values)}

    # Apply the color mapping to the dataframe
    df['assigned_color'] = df[color].map(discrete_color_map)

    # color_discrete_map

    fig=px.scatter(df,
                   x='Cluster_x',
                   y='Cluster_y',
                   width=800,
                   height=800,
                   color=color,
                   color_discrete_map = discrete_color_map,
                #    color_discrete_sequence=marker_color,
                   hover_data=str_cols+key_vals,
                   category_orders={color:df.sort_values(color,ascending=True)[color].unique().tolist()}
                  )
    fig.update_layout(legend_title_text=color,
            font_family='Futura',
            height=800,
            font_color='black',
                      
                      )


    dvz['x']=dvz['x'].round(1)
    dvz['y']=dvz['y'].round(1)
    # for i, r in p.sort_values('distance_from_zero',ascending=False).head(20).groupby(['x','y',]).agg(field=('field', lambda x: '<br>'.join(x))).reset_index().iterrows():
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
                opacity=.25,
                font=dict(
                    color="black",
                    size=12
                    )
                )
    fig.update_traces(mode='markers',
                      opacity=.75,
                      marker=dict(size=16,line=dict(width=2,color='DarkSlateGrey'))
                      )
    fig.update_xaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    fig.update_yaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    st.plotly_chart(fig,use_container_width=True)
    # st.write(fig.data[0]['hovertemplate'])

    for val in dvz.sort_values('distance_from_zero',ascending=False).field.tolist():
       fig=px.bar(df.sort_values(val,ascending=False),
                  x='Player',
                  color=color,
                  color_discrete_map = discrete_color_map,
                  y=val,
                  category_orders={color:df.sort_values(color,ascending=True)[color].unique().tolist()}
                  )
       fig.update_xaxes(categoryorder='total descending')
       fig.update_traces(marker=dict(
        #    color='lightblue',    
           line=dict(color='navy', width=2) 
           )
       )
       fig.update_layout(
        font=dict(
        family='Futura',  # Set font to Futura
        size=12,          # You can adjust the font size if needed
        color='black' 
        ))
    

       st.plotly_chart(fig,use_container_width=True)


@st.cache_data
def load_data():
   df=pd.read_parquet('https://drive.google.com/file/d/1S2N4a3lhohq_EtuY3aMW_d9nIsE4Bruk/view?usp=sharing', engine='pyarrow')
   return df


@st.cache_data
def load_google_file(code):
    url = f"https://drive.google.com/uc?export=download&id={code}"
    file = requests.get(url)
    bytesio = BytesIO(file.content)
    return pd.read_parquet(bytesio)

def app():
    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
        )
    
    code='1_0FAJsULjo-gz2pvy365dQNqbo1ORDMU'
    d1=load_google_file(code)
    # st.write(d1.tail(10))
    code='1S2N4a3lhohq_EtuY3aMW_d9nIsE4Bruk'
    d2=load_google_file(code)
    # st.write(d2.tail(10))

    c1,c2=st.columns([1,3])
    date = c1.date_input(
        "Select a date / month for games",
        value=pd.to_datetime(d1.date.max()),
        min_value=pd.to_datetime('1966-02-19'),
        max_value=pd.to_datetime(d1.date.max())
        )

    # date='2023-08-14'
    date=pd.to_datetime(date)

    # st.write(d1.loc[d1.date == date])
    d1['game_str'] = d1['date'].dt.strftime('%Y-%m-%d') + ' - ' + d1['visitor_team'] + ' ' + d1['visitor_score'].astype('str')+'-'+d1['home_score'].astype('str')+' ' +d1['home_team']
    games=d1.loc[d1.date == date]['game_str'].tolist()
    game_select = c2.selectbox('Select a game: ', games)

    game_id = d1.loc[d1.game_str == game_select].game_id.min()
    # st.write(game_id)

    df = d2.loc[d2.game_id==game_id]
    try:
        df['+/-'] = pd.to_numeric(df['+/-'].astype('str').str.replace('+',''))
    except:
        pass
    # st.write(df)

    # with st.form(key='game_select_form'):
    #     game_select = st.selectbox('Select a match: ', games)



    # r = requests.get(f'https://www.basketball-reference.com/leagues/NBA_{season}_games-{month.lower()}.html')
    # if r.status_code==200:
    #     soup = BeautifulSoup(r.content, 'html.parser')
    #     table = soup.find('table', attrs={'id': 'schedule'})
    #     if table:
    #         df = pd.read_html(str(table))[0]
    #         # game_id = []
    #         # game_url = []
    #         for row in table.findAll('tr'):
    #             try:
    #                 if 'csk' in str(row):
    #                     game_id.append(str(row).split('csk="')[1].split('"')[0])
    #                     game_url.append(str(row).split('data-stat="box_score_text"><a href="')[1].split('"')[0])
    #                     # visitor_team.append(str(row).split('data-stat="visitor_team_name"><a href="/teams/')[1].split('/')[0])
    #                     # home_team.append(str(row).split('data-stat="home_team_name"><a href="/teams/')[1].split('/')[0])
    #                     # visitor_score.append(str(row).split('data-stat="visitor_pts">')[1].split('<')[0])
    #                     # home_score.append(str(row).split('data-stat="home_pts">')[1].split('<')[0])
    #             except:
    #                 pass
    #         df['game_id'] = game_id
    #         game_url_df = pd.DataFrame({'game_url':game_url})
    #         df=pd.concat([df,game_url_df],axis=1)
    #         df.columns=['date','time','visitor','visitor_pts','home','home_pts','del_1','del_2','attendance','arena','notes','game_id','game_url']
    #         df = df.loc[df.home_pts>0]
    #         df['visitor_pts']=df['visitor_pts'].astype('str').str.replace('\.0','',regex=True)
    #         df['home_pts']=df['home_pts'].astype('str').str.replace('\.0','',regex=True)
    #         for col in df.columns:
    #             if 'del' in col:
    #                 del df[col]
    # df['game_string'] = df['visitor'] + ' ' + df['visitor_pts'].astype('str') + ' - ' + df['home'] + ' ' + df['home_pts'].astype('str')

    # games=df['game_string'].tolist()

    # with st.form(key='game_select_form'):
    #     game_select = st.selectbox('Select a match: ', games)

    #     submit_button = st.form_submit_button(label='Submit')

    # game_id=df.query('game_string == @game_select').game_id.min()
    # visiting_team=df.query('game_string == @game_select').visitor.min()
    # home_team=df.query('game_string == @game_select').home.min()

    # if submit_button:
    #     d=get_box_score(game_id,visiting_team,home_team)


    #     st.write(d.head(3))

    df.rename(columns={
            "3par":"3pa Rate",
            "ts_pct":"True Shot Pct",
            "pts":"Points",
            "ast":"Assists",
            "efg_pct":"Eff FG Pct",
            "drtg":"Def Rtg",
            "ortg":"Off Rtg",
            "ast":"Assists",
            "efg_pct":"Eff FG Pct",
            "drtg":"Def Rtg",
            "ortg":"Off Rtg",
            "3par":"3pa Rate",
            "ft":"Free Throws",
            "fta":"Free Throws Attempted",
            "stl_pct":"Steal Pct",
            "player":"Player",
            "team":"Team",
            "mp":"Minutes",
            "fg":"Field Goals",
            "fga":"Field Goals Attempted",
            "fg_pct":"Field Goal Pct",
            "ftp_pct":"Free Throw Pct",
            "trb_pct":"Total Rebound Pct",
            "trb":"Rebounds",
            "stl":"Steals",
            "bpm":"Box +/-",
            "drb":"Def Reb",
            "orb":"Off Reb",
            "pf":"Fouls",
            "tov":"Turnovers",
            "ftr":"FT Rate",
            # "drtg":"Def Rtg",
            "ortg":"Off Rtg",
            "3par":"3pa Rate",
            "ft":"Free Throws",
            "fta":"Free Throws Attempted",
            "stl_pct":"Steal Pct",
            "player":"Player",
            "team":"Team",
            "mp":"Minutes",
            "fg":"Field Goals",
            "fga":"Field Goals Attempted",
            "fg_pct":"Field Goal Pct",
            "ortg":"Off Rtg",
        },
            inplace=True)


        # for x in d.columns:
        #     if 'del_' in x:
        #         del d[x]

    num_cols=[]
    non_num_cols=[]
    for col in df.columns:
        if col in ['date','last_game']:
            try:
                df[col]=pd.to_datetime(df[col])
            except:
                pass
        else:
            try:
                df[col] = pd.to_numeric(df[col])
                num_cols.append(col)
            except:
                # print(f'{col} failed')
                non_num_cols.append(col)
        

        

        # non_num_cols=['player']
    color='team'
    with st.form(key='clstr_form'):
        num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
        non_num_cols_select=st.multiselect('Select non numeric columns for hover data',non_num_cols,['Player'])
        list_one=['Cluster']
        list_two=df.columns.tolist()
        color_options=list_one+list_two

        color_select=st.selectbox('What attribute should color points on the graph?',color_options)
        mp_filter=st.slider('Filter players by minimum minutes played?', min_value=0.0, max_value=df.Minutes.max(), value=0.0, step=1.0)
        df=df.query("Minutes > @mp_filter")
        submit_button = st.form_submit_button(label='Submit')


        

        # # for color in num_cols:
        # #   p=quick_clstr(d.fillna(0), num_cols, non_num_cols, color)

    if submit_button:
        quick_clstr(df.fillna(0), num_cols_select, non_num_cols_select, color_select)







   



if __name__ == "__main__":
    #execute
    app()