import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random


def test_util():
    st.write('Test Util')



@st.fragment
def quick_clstr_util(df, num_cols, str_cols, color, player=None, player_list=None):
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

    if player is None:
        opacity_values = 0.75
        # If player is None, we want to include 'color' and 'color_discrete_map'
        scatter_kwargs = {
            'color': color,
            'color_discrete_map': discrete_color_map,
            'category_orders': {color: df.sort_values(color, ascending=True)[color].unique().tolist()},
            'opacity': opacity_values
        }
    else:
        opacity_values = np.where(df['Player'] == player, 0.9, 0.2)
        # If player is defined, we do NOT want to include 'color' or 'color_discrete_map'
        scatter_kwargs = {'opacity': opacity_values} # Empty dictionary, no color-related arguments


    fig=px.scatter(df,
                   x='Cluster_x',
                   y='Cluster_y',
                   width=800,
                   height=800,
                   # Pass the dynamically created keyword arguments
                   **scatter_kwargs,
                   # We will handle opacity manually below to ensure per-point
                   # if 'color' is used, or apply globally if 'color' is not used.
                   hover_data=str_cols + key_vals,
                   template='simple_white' # Added template for consistency with previous snippets
                  )
    

    fig.update_layout(legend_title_text=color,
            font_family='Futura',
            height=800,
            font_color='black',
                      )


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


    ## Closest players section
    if player is not None:
        closest_players = get_closest_players(df, player)
        closest_players_list =[]
        closest_players_list = closest_players['Player'].tolist()
        closest_players_list.append(player)
        for i,r in df.iterrows():
            if r['Player'] in closest_players_list:
                fig.add_annotation(
                    x=r['Cluster_x'],
                    y=r['Cluster_y']+.015,
                    text=r['Player'],
                    bgcolor="gray",
                    opacity=.85,
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="#ffffff")
                        )
                

    if player_list is not None:
        for i,r in df.iterrows():
            if r['Player'] in player_list:
                fig.add_annotation(
                    x=r['Cluster_x'],
                    y=r['Cluster_y']+.015,
                    text=r['Player'],
                    bgcolor="gray",
                    opacity=.85,
                    showarrow=False,
                    font=dict(
                        size=12,
                        color="#ffffff")
                        )




    for i in range(0,len(fig.data)):
        default_template = fig.data[i].hovertemplate
        updated_template = default_template.replace('=', ': ')
        fig.data[i].hovertemplate = updated_template

    st.plotly_chart(fig,use_container_width=True)


    ## closest player radial diagram
    if player is not None:
        closest_player_select = st.multiselect('Closest Players',
                                           df.sort_values('distance',ascending=True).head(100)['Player'].tolist(),
                                           df.sort_values('distance',ascending=True).head(10)['Player'].tolist()
                                           )


        df_closest = df.loc[df['Player'].isin(closest_player_select)].sort_values('distance',ascending=True).copy()

        scaler = StandardScaler()
        polar_scaled = scaler.fit_transform(df_closest[key_vals])
        scaled_cols = [f'{col}_scaled' for col in key_vals]
        df_closest[scaled_cols] = polar_scaled

        df_melted = pd.melt(df_closest,
                        id_vars=['Player'],
                        value_vars=scaled_cols,
                        var_name='Statistic',
                        value_name='Value')
        
        df_melted_unscaled = pd.melt(df_closest,
                        id_vars=['Player'],
                        value_vars=key_vals,
                        var_name='Statistic',
                        value_name='Value')
        
        df_melted_unscaled.columns = ['Player','Statistic','Value_Unscaled']
        df_melted_unscaled['Statistic'] = df_melted_unscaled['Statistic'].str.replace('_scaled','')
        df_melted['Statistic'] = df_melted['Statistic'].str.replace('_scaled','')
        df_melted = pd.merge(df_melted,df_melted_unscaled,left_on=['Player','Statistic'],right_on=['Player','Statistic'],how='left')

        fill_checkbox = st.checkbox('Include a fill on the visualization (note, some points will be covered)',value=True)

        ## create a radial diagram of the closest 10 players to the selected player
        fig = px.line_polar(df_melted,
                        r='Value',
                        theta='Statistic',
                        color='Player',
                        line_close=True,
                        template='simple_white',
                        hover_data=['Value_Unscaled'],
                        color_discrete_sequence=px.colors.qualitative.Plotly)
        if fill_checkbox:
            fig.update_traces(fill='toself')
        fig.update_layout(
            title=f"Closest Players to {player}",
            font_family='Futura',
            height=800,
            font_color='black',
            showlegend=True
        )

        fig.update_traces(
            mode='lines+markers',
            marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
            line=dict(width=4, dash='dash')
            )
        st.plotly_chart(fig)


    ## Top Attributes Section
    top_attributes = dvz.sort_values('distance_from_zero',ascending=False).field.tolist()
    for i, tab in enumerate(st.tabs(top_attributes)):
        with tab:
            val = top_attributes[i]
            fig=px.bar(df.sort_values(val,ascending=False),
                  x='Player',
                  color=color,
                  color_discrete_map = discrete_color_map,
                  y=val,
                  category_orders={color:df.sort_values(color,ascending=True)[color].unique().tolist()}
                  )
            fig.update_xaxes(categoryorder='total descending')
            fig.update_traces(marker=dict(
                line=dict(color='navy', width=2) 
                )
                )
            fig.update_layout(
               height=800,
               title=val,
               font=dict(
                   family='Futura',
                   size=12, 
                   color='black' 
                   )
            )
            for i in range(0,len(fig.data)):
               default_template = fig.data[i].hovertemplate
               updated_template = default_template.replace('=', ': ')
               fig.data[i].hovertemplate = updated_template
            st.plotly_chart(fig,use_container_width=True)

