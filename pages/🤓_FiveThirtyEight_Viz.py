import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import datetime

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog'
    )

df_color=pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/bovada/master/color_map.csv')
trans_df = df_color[['team','primary_color']].set_index('team').T
color_map=trans_df.to_dict('index')['primary_color']
del df_color
del trans_df

non_pct = ['ELO','SPI','Proj Point Diff','Proj Goal Diff','Proj Points','Points']


# https://raw.githubusercontent.com/aaroncolesmith/538_viz/main/538_champions_league.csv

@st.cache(ttl=3600)
def load_data_sources():
    df_files = pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/538_viz/main/last_updated.csv')
    df_files['hours_since'] = (datetime.datetime.utcnow()-pd.to_datetime(df_files.updated))/np.timedelta64(1, 'h')
    df_files['title'] = df_files['data_sources'] + ' - updated '+ round(df_files['hours_since'],2).astype('str') + ' hours ago'
    df_files['file_names'] = df_files['file_names'].str.replace('./','https://raw.githubusercontent.com/aaroncolesmith/538_viz/main/')
    data_sources=dict(zip(df_files.title, df_files.file_names))
    del df_files
    return data_sources

def app():
    st.title('FiveThirtyEight Viz')

    data_sources = load_data_sources()

    sel = st.radio("Select sport/competition:", list(data_sources.keys()))

    print('AARONLOG - 538 selection ' + sel)

    data = data_sources[sel]
    d=pd.read_csv(data)
    d=d.sort_values(['updated','team'], ascending=(True, True))

    columns=d.columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_columns = d.select_dtypes(include=numerics).columns.tolist()

    title_columns = [x.title() for x in num_columns]
    title_columns = [x.replace('_',' ') for x in title_columns]
    title_columns = [x.replace('Ucl','UCL') for x in title_columns]
    title_columns = [x.replace('Elo','ELO') for x in title_columns]
    title_columns = [x.replace('Spi','SPI') for x in title_columns]

    col_dict = dict(zip(title_columns, num_columns))
    a=np.insert(list(col_dict.keys()),0,'')

    col_select=st.selectbox('Select a metric -', a)

    if len(col_select) > 0:
        val = col_dict[col_select]
        d[val] = pd.to_numeric(d[val])

        fig=px.scatter(d,
                   x='updated',
                   y=val,
                   color='team',
                   title='Odds Over Time',
                   render_mode='svg',
                   color_discrete_map=color_map)
        fig.update_traces(mode='lines',
                            line_shape='spline',
                            opacity=.75,
                            marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                            line = dict(width=4))

        if col_select in non_pct:
            fig.update_yaxes(title=val.replace('_',' ').title())
        else:
            fig.update_yaxes(tickformat = ',.0%',title=val.replace('_',' ').title())
        fig.update_xaxes(title='Date')
        st.plotly_chart(fig, use_container_width=True)


        d[val+'_pct_change_initial']=d.groupby('team')[val].transform(lambda x: (x-x.iloc[0]))
        fig=px.scatter(d,
                   x='updated',
                   y=val+'_pct_change_initial',
                   title='Change Vs. Initial Odds',
                   color='team',
                   color_discrete_map=color_map,
                   render_mode='svg')
        fig.update_traces(mode='lines',
                            line_shape='spline',
                            opacity=.75,
                            marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                            line = dict(width=4))
        if col_select in non_pct:
            fig.update_yaxes(title=val.replace('_',' ').title())
        else:
            fig.update_yaxes(tickformat = ',.0%',title=val.replace('_',' ').title())
        fig.update_xaxes(title='Date')
        st.plotly_chart(fig, use_container_width=True)


        fig=px.bar(
            d.loc[pd.to_datetime(d.updated) == pd.to_datetime(d.updated).max()].sort_values(val,ascending=False),
            x='team',
            y=val,
            title='Latest Odds by Team',
            color='team',
            color_discrete_map=color_map,
            )
        if col_select in non_pct:
            fig.update_yaxes(title=val.replace('_',' ').title())
        else:
            fig.update_yaxes(tickformat = ',.0%',title=val.replace('_',' ').title())
        fig.update_xaxes(title='Team')

        del d[val+'_pct_change_initial']
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    #execute
    app()
