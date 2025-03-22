import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
# from modules.utils import add_logo,data_dictionary_text, sankey, rca_over_time, rca_bar_over_time, col_dict
from streamlit_sortables import sort_items

st.set_page_config(layout="wide")


@st.cache_data
def load_data():
    df=pd.read_csv('data/sankey_data_soccer.csv')
    return df


def generate_gradient(start_color, end_color, num_colors):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb_color):
        return '#{0:02X}{1:02X}{2:02X}'.format(*rgb_color)

    def calculate_opposite_color(color1_rgb, color2_rgb):
        return rgb_to_hex((
            255 - int((color1_rgb[0] + color2_rgb[0]) / 2),
            255 - int((color1_rgb[1] + color2_rgb[1]) / 2),
            255 - int((color1_rgb[2] + color2_rgb[2]) / 2)
        ))

    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)

    if num_colors <= 10:
        color_gradient = [
            rgb_to_hex((
                int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (num_colors - 1)),
                int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (num_colors - 1)),
                int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (num_colors - 1))
            ))
            for i in range(num_colors)
        ]
    else:
        additional_color = calculate_opposite_color(start_rgb, end_rgb)
        additional_rgb = hex_to_rgb(additional_color)

        color_gradient = [
            rgb_to_hex((
                int(start_rgb[0] + (additional_rgb[0] - start_rgb[0]) * i / (num_colors - 1)),
                int(start_rgb[1] + (additional_rgb[1] - start_rgb[1]) * i / (num_colors - 1)),
                int(start_rgb[2] + (additional_rgb[2] - start_rgb[2]) * i / (num_colors - 1))
            ))
            for i in range(num_colors // 2)
        ]
        try:
            color_gradient._append(additional_color)
        except:
            color_gradient.append(additional_color)

        color_gradient.extend([
            rgb_to_hex((
                int(additional_rgb[0] + (end_rgb[0] - additional_rgb[0]) * i / (num_colors // 2 - 1)),
                int(additional_rgb[1] + (end_rgb[1] - additional_rgb[1]) * i / (num_colors // 2 - 1)),
                int(additional_rgb[2] + (end_rgb[2] - additional_rgb[2]) * i / (num_colors // 2 - 1))
            ))
            for i in range(num_colors // 2)
        ])

    return color_gradient

def sankey(df,cols,val,color='not_gray'):

    
    total=df[val].sum()
    df_viz=df.groupby(cols).agg(
                                UNITS=(val,'sum')).reset_index()

    df_agg=pd.DataFrame()
    for idx, col in enumerate(cols):
        if idx < len(cols)-1:
            df_tmp=df_viz.groupby([cols[idx],cols[idx+1]]).agg(SIZE=('UNITS','sum')).reset_index()
            df_tmp.columns=['SOURCE','TARGET','SIZE']
            df_tmp['SOURCE_LEVEL'] = idx
            df_tmp['TARGET_LEVEL'] = idx+1
            df_agg=pd.concat([df_agg,df_tmp])
    try:
        unique_values = df_agg['SOURCE']._append(df_agg['TARGET']).unique()
    except:
        unique_values = df_agg['SOURCE'].append(df_agg['TARGET']).unique()
    value_to_key = {value: key for key, value in enumerate(unique_values, start=0)}
    df_agg['SOURCE_ID'] = df_agg['SOURCE'].map(value_to_key)
    df_agg['TARGET_ID'] = df_agg['TARGET'].map(value_to_key)


    start_color='#FF6600'
    start_color='#FF13F0'
    end_color='#55E0FF'
    end_color='#02B7DD'
    color_list = generate_gradient(start_color, end_color, len(cols))

    if color=='gray':
        color_list=['lightgray','lightgray','lightgray','lightgray','lightgray','lightgray','lightgray','lightgray','lightgray','lightgray','lightgray','lightgray']

    color_dict={index: value for index, value in enumerate(color_list)}


    df_agg['SOURCE_COLOR'] = df_agg['SOURCE_LEVEL'].map(color_dict)
    df_agg['TARGET_COLOR'] = df_agg['TARGET_LEVEL'].map(color_dict)
    df_agg.reset_index(drop=True,inplace=True)

    df_agg['SIZE_PCT_BY_SOURCE'] = df_agg.groupby(['SOURCE'])['SIZE'].transform(lambda x: x/x.sum())
    df_agg['SIZE_PCT_BY_TARGET'] = df_agg.groupby(['TARGET'])['SIZE'].transform(lambda x: x/x.sum())
    df_agg['SIZE_PCT_TOTAL'] = df_agg['SIZE'] / total


    color_dict=df_agg.set_index('SOURCE')['SOURCE_COLOR'].to_dict() | df_agg.set_index('TARGET')['TARGET_COLOR'].to_dict()


    node_df1=df_agg.groupby(['SOURCE']).agg(idx=('SOURCE_ID','min'),
                                            LEVEL=('SOURCE_LEVEL','min'),
                                            COLOR=('SOURCE_COLOR','min'),
                                        SIZE=('SIZE','sum')).sort_values('idx',ascending=True).reset_index()
    node_df1.columns=['LABEL','IDX','LEVEL','COLOR','SIZE']

    node_df2=df_agg.groupby(['TARGET']).agg(idx=('TARGET_ID','min'),
                                            LEVEL=('TARGET_LEVEL','min'),
                                            COLOR=('TARGET_COLOR','min'),
                                            SIZE=('SIZE','sum')).sort_values('idx',ascending=True).reset_index()
    node_df2.columns=['LABEL','IDX','LEVEL','COLOR','SIZE']

    node_df=pd.concat([node_df1,node_df2])
    node_df=node_df.drop_duplicates().reset_index(drop=True)
    node_df=node_df.groupby(['LABEL','IDX','LEVEL','COLOR']).agg(SIZE=('SIZE','max')).sort_values('IDX',ascending=True).reset_index()

    node_df['SIZE_PCT_BY_LEVEL'] = node_df.groupby(['LEVEL'])['SIZE'].transform(lambda x: x/x.sum())
    df_agg['LINK_COLOR'] = 'lightgray'
    node_df['LABEL_PCT'] = node_df['LABEL'] + ' - ' + (100*node_df['SIZE_PCT_BY_LEVEL']).apply('{:.2f}%'.format)

    fig=go.Figure(
        data=[go.Sankey(
        arrangement='perpendicular',
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black'),
                label=node_df['LABEL_PCT'],
                color=node_df['COLOR'],
                customdata=node_df['SIZE_PCT_BY_LEVEL'],
                hovertemplate= "%{label}<br>%{value} Units -- %{customdata:.1%}" + "<extra></extra>",
                
                
            ),
            link=dict(
                source=df_agg['SOURCE_ID'],
                target=df_agg['TARGET_ID'],
                value=df_agg['SIZE'],
                color=df_agg['LINK_COLOR'],
                customdata=df_agg[['SIZE_PCT_BY_SOURCE','SIZE_PCT_BY_TARGET','SIZE_PCT_TOTAL']],
                hovertemplate= "From %{source.label} -> %{target.label}<br> %{value} Units -- %{customdata[2]:.1%} of Total<br>%{customdata[0]:.1%} of %{source.label} ---- %{customdata[1]:.1%} of %{target.label}" + "<extra></extra>"
            )
        )
            ]
    )






    # This adds headers
    for x_coordinate, column_name in enumerate(cols):

        
        fig.add_annotation(
                x=x_coordinate,
                y=1.05,
                xref="x",
                yref="paper",
                text=column_name,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                opacity = .8,
                font=dict(
                    # family="Courier New, monospace",
                    size=12,
                    color="black",
                    
                    ),
                align="center",
                )


    fig.update_layout(
        height=800,
        width=1200,
        font_family="Futura",
        font_color="black",
    xaxis={
    'showgrid': False, # thin lines in the background
    'zeroline': False, # thick line at x=0
    'visible': False,  # numbers below
    },
    yaxis={
    'showgrid': False, # thin lines in the background
    'zeroline': False, # thick line at x=0
    'visible': False,  # numbers below
    }, plot_bgcolor='rgba(0,0,0,0)', 
    font_size=10
    
    
    
    )

    st.plotly_chart(fig, use_container_width=True)



def rca_over_time(d,attr,val,y_title,y_tick):
    if d.index.size <= 75:
        marker_size=12
    elif d.index.size <= 300:
        marker_size=8
    else:
        marker_size=6

    fig=px.scatter(d,
            x='DATE_AGG',
            y='ROLLING_AVG',
            color=attr,
            render_mode='svg',
            template='simple_white',
            )
    fig.update_traces(
        mode='lines',
        line_shape='spline',
        line=dict(width=4),
        marker=dict(size=12,opacity=.9,
            line=dict(width=1,color='DarkSlateGrey')
                )
    )
    fig.update_layout(
                font_family="Futura",
                height=800,
        font_color="black",
    )

    fig.update_yaxes(
        title=y_title,
    tickformat =y_tick
                        )
    fig.update_xaxes(
            title='Date',
            
                        )
    d['DAY_OF_WEEK'] = pd.to_datetime(d['DATE_AGG']).dt.day_name()
    # st.write(d)
    fig2=px.scatter(d,
            x='DATE_AGG',
            y=val,
            hover_data=['DAY_OF_WEEK'],
                color=attr,
                render_mode='svg',
                template='simple_white',
                )
    fig2.update_traces(
        mode='markers',
        line_shape='spline',
        line=dict(width=3),
        marker=dict(size=marker_size,opacity=.35,
            line=dict(width=1,color='DarkSlateGrey')
                )
    )

    for i in range(len(fig2.data)):
        fig.add_trace(fig2.data[i])

    st.plotly_chart(fig, use_container_width=True)

def app():
    df = load_data()

    c1,c2=st.columns(2)
    c3,c4=st.columns(2)

    all_cols=df.columns
    num_cols = ['game_count']
    default_cols=['league_name','spread_covered_cat','total_covered_cat','total_cat']
    ltmp = [x for x in all_cols if x not in num_cols]
    non_default_cols = [x for x in ltmp if x not in default_cols]

    with c1.expander('Hide / show attributes / RCAs'):
        with st.form(key='attribute_form'):
            items = [
                {'header': 'Viewable Attributes', 'items': default_cols
                                                              },
                {'header': 'Hidden Attributes', 'items':     non_default_cols
                                                                },
                                                                ]
            sorted_items = sort_items(items, multi_containers=True)
            cols=sorted_items[0]['items']

            attr_submit_button = st.form_submit_button(label='Submit')
            

    with c2.expander('Filter data:'):
                with st.form(key='filter_form'):
                    for col in cols:
                        col_values=list(set(df[col].unique().tolist()))
                        filter_list = st.multiselect('Filter by '+col, col_values,col_values)
                        df=df.loc[df[col].isin(filter_list)]
                    submit_button = st.form_submit_button(label='Submit')
                        

    
    val = 'game_count'

    sankey(df,cols,val)





    ## RCA OVER TIME

    c1,c2,c3,c4=st.columns(4)
    attr=c1.selectbox('Select an RCA:',cols)
    rolling_avg=c4.slider('Select how far back for the rolling avg',
                          min_value=1,
                          max_value=90,
                          value=1)
    val_select=c2.selectbox('% or # of Units',['% of Games','# of Games'])

    if val_select == '% of Games':
        val = 'game_pct'
        y_title= '% Games'
        y_tick=',.1%'
        text_auto=',.1%'
    if val_select == '# of Games':
        val= 'game_count'
        y_title= '# Games'
        y_tick='.3s'
        text_auto='.3s'


    df['week'] = pd.to_datetime(df['week'])

    date_agg_select=c3.selectbox('Date Rollup',[
        # 'Daily',
        'Weekly','Monthly',
        # 'Quarterly'
        ])
    if date_agg_select == 'Daily':
        df['DATE_AGG'] = df['week']
    elif date_agg_select == 'Weekly':
        df['DATE_AGG'] = df['week'] - pd.to_timedelta(df['week'].dt.dayofweek, unit='D')
    elif date_agg_select == 'Monthly':
        df['DATE_AGG'] = df['week'] - pd.to_timedelta(df['week'].dt.day - 1, unit='D')
    elif date_agg_select == 'Quarterly':
        df['DATE_AGG'] = df['week'] - pd.to_timedelta((df['week'].dt.quarter - 1) * 3 + (df['DATE'].dt.month - 1) % 3, unit='M')

    d=df.groupby(['DATE_AGG',attr]).agg(
            game_count=('game_count','sum'),
            
            ).reset_index()
    d['game_pct']=d['game_count'] / d.groupby(['DATE_AGG'])['game_count'].transform('sum')

    d['ROLLING_AVG'] = d.groupby(attr)[val].transform(lambda x: x.rolling(rolling_avg).mean())

    rca_over_time(d,attr,val,y_title,y_tick)





if __name__ == "__main__":
    #execute
    app()
