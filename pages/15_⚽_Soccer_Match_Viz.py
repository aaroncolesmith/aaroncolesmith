import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import functools as ft





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

    for i in range(0,len(fig.data)):
        default_template = fig.data[i].hovertemplate
        updated_template = default_template.replace('=', ': ')
        fig.data[i].hovertemplate = updated_template

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
           height=800,
           title=val,
        font=dict(
        family='Futura',  # Set font to Futura
        size=12,          # You can adjust the font size if needed
        color='black' 
        ))
       
       for i in range(0,len(fig.data)):
        default_template = fig.data[i].hovertemplate
        updated_template = default_template.replace('=', ': ')
        fig.data[i].hovertemplate = updated_template
    

       st.plotly_chart(fig,use_container_width=True)




def quick_clstr_bak(df, num_cols, str_cols, color):
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

    kmeans = KMeans(n_clusters=3, random_state=2).fit_predict(x_pca)

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

    df['Cluster'] = df['Cluster'].astype('str')

    fig=px.scatter(df.sort_values(color,ascending=True),
                   x='Cluster_x',
                   y='Cluster_y',
                   width=800,
                   height=800,
                   color=color,
                   hover_data=str_cols+key_vals,
                   color_continuous_scale='jet'
                  )
    fig.update_layout(legend_title_text=color)

    # fig.update_layout({
    #     'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    #     'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    #     })


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
                      marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey'))
                      )
    fig.update_xaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    fig.update_yaxes(visible=True, zeroline=True, showgrid=True, showticklabels=False, title='')
    # st.plotly_chart(fig,use_container_width=True)
    sel_data = st.plotly_chart(fig,use_container_width=True,on_select='rerun')
    st.write(sel_data)



def app():
    st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
        )
    
    c1,c2=st.columns([1,3])
    date = c1.date_input(
        "Select a date for matches",
        value=pd.to_datetime('today') - pd.Timedelta(days=1),
        min_value=pd.to_datetime('2012-02-19'),
        max_value=pd.to_datetime('today')- pd.Timedelta(days=1)
        )

    # date='2023-08-14'
    url=f'https://fbref.com/en/matches/{date}'
    r=requests.get(url)


    soup = BeautifulSoup(r.content, "html.parser")
    all_urls = []
    for td_tag in soup.find_all('td', {"class":"center"}):
        if 'href' in str(td_tag):
            all_urls.append(
                "https://fbref.com" +str(td_tag).split('href="')[1].split('">')[0]
            )


    dfs = pd.read_html(url, header=0, index_col=0)
    df = pd.DataFrame(dfs[0])
    for i in range(1, len(dfs)):
        df = pd.concat([df,dfs[i]])
    df=df.query('Score.notnull()')

    df['Home'] = df['Home'].str.rsplit(' ', n=1).str[0]
    df['Away'] = df['Away'].str.split(' ', n=1).str[1]

    df.reset_index(drop=False,inplace=True)
    df['url'] = pd.Series(all_urls)
    df['match_selector'] = df['Home']+' '+df['Score']+' '+df['Away']

    matches=df['match_selector'].tolist()
    match_select = c2.selectbox('Select a match: ', matches)   

    match_url=df.query('match_selector == @match_select').url.min()

    r = requests.get(match_url)
    soup = BeautifulSoup(r.content, 'html.parser')

    team_one = soup.select("#content > div.scorebox > div:nth-child(1) > div:nth-child(1) > strong > a")
    team_one=team_one[0].text

    team_two = soup.select("#content > div.scorebox > div:nth-child(2) > div:nth-child(1) > strong > a")
    team_two=team_two[0].text


    dfs = [pd.read_html(r.content)[3],pd.read_html(r.content)[4],pd.read_html(r.content)[5],pd.read_html(r.content)[6],pd.read_html(r.content)[7],pd.read_html(r.content)[8]]

    d1 = ft.reduce(lambda left, right: pd.merge(left, right), dfs)
    d1.columns=d1.columns.map('_'.join).str.strip().str.lower().str.replace('%','_pct').str.replace(' ','_')
    for i in range(0,50):
        d1.columns=d1.columns.str.replace('unnamed:_'+str(i)+'_level_0_','')
    d1=d1.loc[d1.pos.notnull()]
    d1['team'] = team_one

    dfs = [pd.read_html(r.content)[10],pd.read_html(r.content)[11],pd.read_html(r.content)[12],pd.read_html(r.content)[13],pd.read_html(r.content)[14],pd.read_html(r.content)[15]]

    d2 = ft.reduce(lambda left, right: pd.merge(left, right), dfs)
    d2.columns=d2.columns.map('_'.join).str.strip().str.lower().str.replace('%','_pct').str.replace(' ','_')
    for i in range(0,50):
        d2.columns=d2.columns.str.replace('unnamed:_'+str(i)+'_level_0_','')
    d2=d2.loc[d2.pos.notnull()]
    d2['team'] = team_two

    d=pd.concat([d1,d2])
    # try:
    #     d.columns = [
    #         "player",
    #         "number",
    #         "nation",
    #         "pos",
    #         "age",
    #         "min",
    #         "goals",
    #         "assists",
    #         "pks",
    #         "pk_att",
    #         "shots",
    #         "shots_on_goal",
    #         "yellow_card",
    #         "red_card",
    #         "touches",
    #         "tackles",
    #         "ints",
    #         "blocks",
    #         "xg",
    #         "npxg",
    #         "xag",
    #         "sca",
    #         "gca",
    #         "passes_cmp",
    #         "passes_att",
    #         "passes_cmp_pct",
    #         "progressive_passes",
    #         "carries",
    #         "progressive_carries",
    #         "take_ons_attempted",
    #         "take_ons_successful",
    #         "del_asd",
    #         "del_dd",
    #         "del_cmp",
    #         "passing_distance",
    #         "progressive_passing_distance",
    #         "short_passes_cmp",
    #         "short_passes_att",
    #         "short_passes_cmp_pct",
    #         "med_passes_cmp",
    #         "med_passes_att",
    #         "med_passes_cmp_pct",
    #         "long_passes_cmp",
    #         "long_passes_att",
    #         "long_passes_cmp_pct",
    #         "del_ast",
    #         "del_xag",
    #         "xa",
    #         "key_passes",
    #         "final_third_passes",
    #         "passes_penalty_area",
    #         "crosses_penalty_area",
    #         "del_prgp",
    #         "del_att",
    #         "live_ball_passes",
    #         "dead_ball_passes",
    #         "fk_passes",
    #         "through_balls",
    #         "switches",
    #         "crosses",
    #         "throw_ins",
    #         "corners",
    #         "corners_inswing",
    #         "corners_outswing",
    #         "corners_straight",
    #         "del_pass_com",
    #         "offside_passes",
    #         "blocked_passes",
    #         "del_tackles_tkl",
    #         "tackles_won",
    #         "tackles_def_3rd",
    #         "tackles_mid_3rd",
    #         "tackles_att_3rd",
    #         "dribblers_tackled",
    #         "dribblers_challenged",
    #         "tackle_pct",
    #         "challenges_lost",
    #         "del_blocks_blocks",
    #         "block_shot",
    #         "block_pass",
    #         "del_int",
    #         "tkl+int",
    #         "clearances",
    #         "errors",
    #         "del_touches_touches",
    #         "touches_def_pen",
    #         "touches_def_3rd",
    #         "touches_mid_3rd",
    #         "touches_att_3rd",
    #         "touches_att_pen",
    #         "touches_live_ball",
    #         "take_ons_succ_pct",
    #         "take_ons_tackled",
    #         "take_ons_tkld_pct",
    #         "carries_total_distance",
    #         "carries_progressive_distance",
    #         "carries_1/3",
    #         "carries_into_penalty_area",
    #         "carries_mis",
    #         "carries_dis",
    #         "passes_received",
    #         "progressive_passes_received",
    #         "del_performance_2crdy",
    #         "fouls_committed",
    #         "fouls_drawn",
    #         "offsides",
    #         "del_performance_crs",
    #         "del_performance_tklw",
    #         "pk_won",
    #         "pk_conceded",
    #         "own_goals",
    #         "balls_recovered",
    #         "aerial_duels_won",
    #         "aerial_duels_lost",
    #         "aerial_duels_won_pct",
    #         "team",
    #     ]
    # except:
    #     pass


    d.rename(columns={
            "performance_gls":"goals",
            "performance_ast":"assists",
            "performance_pk":"pks",
            "performance_pkatt":"pk_att",
            "performance_sh":"shots",
            "performance_sot":"shots_on_goal",
            "performance_crdy":"yellow_card",
            "performance_crdr":"red_card",
            "performance_touches":"touches",
            "performance_tkl":"tackles",
            "performance_int":"ints",
            "performance_blocks":"blocks",
            "expected_xg":"xg",
            "expected_npxg":"npxg",
            "expected_xag":"xag",
            "sca_sca":"sca",
            "sca_gca":"gca",
            "passes_cmp":"passes_cmp",
            "passes_att":"passes_att",
            "passes_cmp_pct":"passes_cmp_pct",
            "passes_prgp":"progressive_passes",
            "carries_carries":"carries",
            "carries_prgc":"progressive_carries",
            "take-ons_att":"take_ons_attempted",
            "take-ons_succ":"take_ons_successful",
            "total_cmp":"del_asd",
            "total_att":"del_dd",
            "total_cmp_pct":"del_cmp",
            "total_totdist":"passing_distance",
            "total_prgdist":"progressive_passing_distance",
            "short_cmp":"short_passes_cmp",
            "short_att":"short_passes_att",
            "short_cmp_pct":"short_passes_cmp_pct",
            "medium_cmp":"med_passes_cmp",
            "medium_att":"med_passes_att",
            "medium_cmp_pct":"med_passes_cmp_pct",
            "long_cmp":"long_passes_cmp",
            "long_att":"long_passes_att",
            "long_cmp_pct":"long_passes_cmp_pct",
            "ast":"del_ast",
            "xag":"del_xag",
            "xa":"xa",
            "kp":"key_passes",
            "1/3":"final_third_passes",
            "ppa":"passes_penalty_area",
            "crspa":"crosses_penalty_area",
            "prgp":"del_prgp",
            "att":"del_att",
            "pass_types_live":"live_ball_passes",
            "pass_types_dead":"dead_ball_passes",
            "pass_types_fk":"fk_passes",
            "pass_types_tb":"through_balls",
            "pass_types_sw":"switches",
            "pass_types_crs":"crosses",
            "pass_types_ti":"throw_ins",
            "pass_types_ck":"corners",
            "corner_kicks_in":"corners_inswing",
            "corner_kicks_out":"corners_outswing",
            "corner_kicks_str":"corners_straight",
            "outcomes_cmp":"del_pass_com",
            "outcomes_off":"offside_passes",
            "outcomes_blocks":"blocked_passes",
            "tackles_tkl":"del_tackles_tkl",
            "tackles_tklw":"tackles_won",
            "tackles_def_3rd":"tackles_def_3rd",
            "tackles_mid_3rd":"tackles_mid_3rd",
            "tackles_att_3rd":"tackles_att_3rd",
            "challenges_tkl":"dribblers_tackled",
            "challenges_att":"dribblers_challenged",
            "challenges_tkl_pct":"tackle_pct",
            "challenges_lost":"challenges_lost",
            "blocks_blocks":"del_blocks_blocks",
            "blocks_sh":"block_shot",
            "blocks_pass":"block_pass",
            "int":"del_int",
            "tkl+int":"tkl+int",
            "clr":"clearances",
            "err":"errors",
            "touches_touches":"del_touches_touches",
            "touches_def_pen":"touches_def_pen",
            "touches_def_3rd":"touches_def_3rd",
            "touches_mid_3rd":"touches_mid_3rd",
            "touches_att_3rd":"touches_att_3rd",
            "touches_att_pen":"touches_att_pen",
            "touches_live":"touches_live_ball",
            "take-ons_succ_pct":"take_ons_succ_pct",
            "take-ons_tkld":"take_ons_tackled",
            "take-ons_tkld_pct":"take_ons_tkld_pct",
            "carries_totdist":"carries_total_distance",
            "carries_prgdist":"carries_progressive_distance",
            "carries_1/3":"carries_third",
            "carries_cpa":"carries_into_penalty_area",
            "carries_mis":"carries_mis",
            "carries_dis":"carries_dis",
            "receiving_rec":"passes_received",
            "receiving_prgr":"progressive_passes_received",
            "performance_2crdy":"del_performance_2crdy",
            "performance_fls":"fouls_committed",
            "performance_fld":"fouls_drawn",
            "performance_off":"offsides",
            "performance_crs":"del_performance_crs",
            "performance_tklw":"del_performance_tklw",
            "performance_pkwon":"pk_won",
            "performance_pkcon":"pk_conceded",
            "performance_og":"own_goals",
            "performance_recov":"balls_recovered",
            'player':'Player'
        },
             inplace=True)


    for x in d.columns:
        if 'del_' in x:
            del d[x]

    num_cols=[]
    non_num_cols=[]
    for col in d.columns:
        if col in ['first_game','last_game']:
            try:
                d[col]=pd.to_datetime(d[col])
            except:
                pass
        else:
            try:
                d[col] = pd.to_numeric(d[col])
                num_cols.append(col)
            except:
                # print(f'{col} failed')
                non_num_cols.append(col)
    
    if 'number' in num_cols:
        num_cols.remove('number')
    else:
        num_cols.remove('#')
    

    # non_num_cols=['player']
    color='team'
    with st.form(key='my_form'):
        
            
    # c1,c2,c3 = st.columns(3)
        num_cols_select = st.multiselect('Select statistics to be used for analysis', num_cols, num_cols)
        non_num_cols_select=st.multiselect('Select non numeric columns for hover data',non_num_cols,['Player'])
        list_one=['Cluster']
        list_two=d.columns.tolist()
        color_options=list_one+list_two

        color_select=st.selectbox('What attribute should color points on the graph?',color_options)

        mp_filter=st.slider('Filter players by minimum minutes played?', min_value=0, max_value=d['min'].max(), value=0, step=1)
        d=d.query("min > @mp_filter")

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        quick_clstr(d.fillna(0), num_cols_select, non_num_cols_select, color_select)







   



if __name__ == "__main__":
    #execute
    app()