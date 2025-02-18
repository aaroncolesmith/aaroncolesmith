import pandas as pd
import requests
from pandas.io.json import json_normalize
import datetime
import io
from io import StringIO
# import boto3
import streamlit as st
import numpy as np
import plotly_express as px
import sys
import pyarrow as pa
import pyarrow.parquet as pq

color_discrete_sequence=['#FF1493','#120052','#652EC7','#00C2BA','#82E0BF','#55E0FF','#002BFF','#FF911A']


# def max_minus_min(x):
#     return max(x) - min(x)

# def last_minus_avg(x):
#     return last(x) - mean(x)

@st.cache(ttl=43200, suppress_st_warning=True)
def load_file():
    # df = pd.read_csv('./data/bovada.csv')
    # df=pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/bovada/master/bovada_new.csv')
    # df=pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/bovada/master/bovada_data.csv')
    df=pd.read_parquet('https://github.com/aaroncolesmith/bovada/blob/master/bovada_data.parquet?raw=true', engine='pyarrow')
    df['date'] = pd.to_datetime(df['date'])
    df['seconds_ago']=(pd.to_numeric(datetime.datetime.utcnow().strftime("%s")) - pd.to_numeric(df['date'].apply(lambda x: x.strftime('%s'))))
    df['seconds_ago']=(pd.to_numeric(datetime.datetime.utcnow().strftime("%s")) - pd.to_numeric(df['date'].apply(lambda x: x.strftime('%s'))))
    df['minutes_ago'] = round(df['seconds_ago']/60,2)
    df['Prev_Probability']=df.groupby(['title','description'])['Implied_Probability'].transform(lambda x: x.shift(1))
    df['Implied_Probability'] = round(df['Implied_Probability'],4)
    df['Prev_Probability'] = round(df['Prev_Probability'],4)

    

    return df

# @st.cache(suppress_st_warning=True)
def load_scatter_data():
    df=pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/bovada/master/bovada_scatter.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['seconds_ago']=(pd.to_numeric(datetime.datetime.utcnow().strftime("%s")) - pd.to_numeric(df['date'].apply(lambda x: x.strftime('%s'))))
    df['minutes_ago'] = round(df['seconds_ago']/60,2)
    df['hours_ago'] = round(df['minutes_ago']/60,2)

    last_update = (datetime.datetime.utcnow() - df['date'].max()).total_seconds()

    if last_update <60:
        st.write('Last update: '+str(round(last_update,2))+' seconds ago')
    elif last_update <3600:
        st.write('Last update: '+str(round(last_update/60,2))+' minutes ago')
    else:
        st.write('Last update: '+str(round(last_update/3600,2))+' hours ago')

    return df

# @st.cache(suppress_st_warning=True)
# def get_s3_data(bucket, key):
#     s3 = boto3.client('s3')
#     obj = s3.get_object(Bucket=bucket, Key=key)
#     df = pd.read_csv(io.BytesIO(obj['Body'].read()))
#     df['date'] = pd.to_datetime(df['date'])
#     df['seconds_ago']=(pd.to_numeric(datetime.datetime.utcnow().strftime("%s")) - pd.to_numeric(df['date'].apply(lambda x: x.strftime('%s'))))
#     return df

# def save_to_s3(df, bucket, key):
#     s3_resource = boto3.resource('s3')
#     csv_buffer = StringIO()
#     df.to_csv(csv_buffer,index=False)
#     s3_resource.Object(bucket, key).put(Body=csv_buffer.getvalue())

def get_select_options(df, track_df):
    b=track_df.groupby(['selection']).agg({'count':'sum'}).sort_values(['count'], ascending=False).reset_index(drop=False)
    a=pd.merge(df, b, left_on='title',right_on='selection', how = 'left').sort_values(['count'], ascending=False)
    a['date'] = pd.to_datetime(a['date'])
    a=a.groupby('title').agg({'date':'max','count':'first'}).reset_index().sort_values('count',ascending=False)
    a['date'] = a['date'].dt.floor('Min').astype('str').str[:16].str[5:]
    a=a['title'] + ' | ' + a['date']
    a=a.to_list()
    a=np.insert(a,0,'')
    return a

def table_output(df):
    st.write(df.groupby(['Winner']).agg({'Date':'max','Price': ['last','mean','max','min',max_minus_min,'count']}).sort_values([('Price', 'mean')], ascending=True))

def line_chart(df, option, color_map):
    g=px.line(df,
    x='Date',
    y='Price',
    color='Winner',
    render_mode='svg',
    color_discrete_map=color_map,
    # animation_frame='Date',
    color_discrete_sequence=color_discrete_sequence,
    title='Betting Odds Over Time <br><sup>'+option+' </sup>')
    g.update_traces(mode='lines',
                    line_shape='spline',
                    opacity=.75,
                    marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                    line = dict(width=4),
                    hovertemplate='%{x} - %{y}')
    g.update_yaxes(title='Implied Probability',
                   showgrid=False,
                   # gridwidth=1,
                   # gridcolor='#D4D4D4'
                  )
    # g.update_layout(plot_bgcolor='white')
    g.update_xaxes(title='Date',
                  showgrid=False,
                  # gridwidth=1,
                  # gridcolor='#D4D4D4'
                  )
    # g=color_update(g)
    st.plotly_chart(g,use_container_width=True)

def line_chart_probability(df,option,color_map):
    g=px.line(df,
    x='Date',
    y='Implied_Probability',
    color='Winner',
    render_mode='svg',
    color_discrete_map=color_map,
    color_discrete_sequence=color_discrete_sequence,
    # title='Implied Probability Over Time'
    title="Implied Probability Over Time <br><sup>"+option+" </sup>")
    
    g.update_traces(mode='lines',
                    line_shape='spline',
                    opacity=.75,
                    marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                    line = dict(width=4),
                    hovertemplate='%{x} - %{y}')
    g.update_yaxes(
    #              range=[0, 1],
                   title='Implied Probability',
                   showgrid=False,
                   # gridwidth=1,
                   # gridcolor='#D4D4D4',
                   tickformat = ',.0%'
                  )
    # g.update_layout(plot_bgcolor='white')
    g.update_xaxes(title='Date',
                  showgrid=False,
                  # gridwidth=1,
                  # gridcolor='#D4D4D4'
                  )
    # g=color_update(g)
    st.plotly_chart(g,use_container_width=True)

def line_chart_probability_initial(df,option,color_map):
    df['Implied_Probability_Initial_Change']=df.groupby('Winner')['Implied_Probability'].transform(lambda x: (x-x.iloc[0]))
    g=px.line(df,
    x='Date',
    y='Implied_Probability_Initial_Change',
    color='Winner',
    render_mode='svg',
    color_discrete_map=color_map,
    color_discrete_sequence=color_discrete_sequence,
    title='Implied Probability - Change Since Initial Odds <br><sup>'+option+'</sup>')
    g.update_traces(mode='lines',
                    line_shape='spline',
                    opacity=.75,
                    marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')),
                    line = dict(width=4),
                    hovertemplate='%{x} - %{y}')
    g.update_yaxes(
    #              range=[0, 1],
                   title='Implied Probability',
                   showgrid=False,
                   # gridwidth=1,
                   # gridcolor='#D4D4D4',
                   tickformat = ',.0%'
                  )
    # g.update_layout(plot_bgcolor='white')
    g.update_xaxes(title='Date',
                  showgrid=False,
                  # gridwidth=1,
                  # gridcolor='#D4D4D4'
                  )
    # g=color_update(g)
    st.plotly_chart(g,use_container_width=True)

def get_color_map():
  df=pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/bovada/master/color_map.csv')
  trans_df = df[['team','primary_color']].set_index('team').T
  color_map=trans_df.to_dict('index')['primary_color']

  # word_freq.update({'before': 23})
  return color_map

def ga(event_category, event_action, event_label):
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec='+event_category+'&ea='+event_action+'&el='+event_label+'">',unsafe_allow_html=True)

def recent_updates():
    # d=df.loc[(df.Pct_Change.abs() > .01) & (df.date >= df.date.max() - pd.Timedelta(hours=4)) & (df.Pct_Change.notnull())].sort_values('Pct_Change',ascending=False).reset_index(drop=False)
    d=load_scatter_data()
    fig=px.scatter(d,
               y='Pct_Change',
               title='Recent Updates - Wagers Rising / Falling',
               hover_data=['title','description','Implied_Probability','Prev_Probability', 'hours_ago'])
    fig.update_traces(opacity=.75,
                    marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey'),
                                color=np.where(d['Pct_Change'] > 0,'green',np.where(d['Pct_Change'] < 0,'red','red'))
                                )
                    )
    fig.update_yaxes(
                    title='Percent Change',
                    showgrid=False,
                    tickformat = ',.0%'
                  )

    fig.update_layout(xaxis_type = 'category')

    fig.update_xaxes(
                    title='',
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    categoryorder='total descending'
                  )

    fig.update_traces(hovertemplate='Bet Title: %{customdata[0]}<br>Bet Wager: %{customdata[1]}<br>Probability Change: %{customdata[3]:.2%} > %{customdata[2]:.2%}<br>Pct Change: %{y:.1%}<br>Last Update: %{customdata[4]} hrs ago')

    st.plotly_chart(fig)

    d['date']=d['date'].astype('str').str[:16].str[5:]
    d['title']=d['title'] + ' | ' + d['date']
    
    return d['title'].unique()


def app():

    color_map = get_color_map()

    st.title('Bovada Odds Over Time')
    st.markdown('Welcome to Bovada Scrape!!! Select an option below and see how the betting odds have tracked over time!')

    recent_list= recent_updates()
    recent_list = recent_list.tolist()

    df = load_file()

    ga('bovada','get_data',str(df.index.size))

    a=df.groupby('title').agg({'date':['max','size','nunique']}).reset_index()
    a.columns = ['title','date','count','unique']
    a['date_sort'] = a['date'].astype('datetime64[D]')
    a=a.sort_values(['date_sort','unique','count'],ascending=(False,False,False))
    del a['date_sort']
    a['date']=a['date'].astype('str').str[:16].str[5:]
    a=a['title'] + ' | ' + a['date']
    a=a.to_list()
    a = recent_list + a
    tmp_list = []
    [tmp_list.append(x) for x in a if x not in tmp_list]
    a=tmp_list
    del tmp_list
    a=np.insert(a,0,'')

    option=st.selectbox('Select a bet -', a)
    # option = option[:-14]

    if len(option) > 0:
            print('AARONLOG - Bovada selection ' + option)
            o = st.radio( "Show all or favorites only?",('Show All', 'Favorites'))
            try:
                option = option.split(' |')[0]
            except:
                pass
            # st.markdown('#### '+option)
            
            if o == 'Show All':
                filtered_df = df.loc[df.title == option]
                filtered_df = filtered_df[['date','title','description','price.american','Implied_Probability']].reset_index(drop=True)
                filtered_df.columns = ['Date','Title','Winner','Price','Implied_Probability']
                filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

            if o == 'Favorites':
                filtered_df = df.loc[df.title == option]
                filtered_df = filtered_df[['date','title','description','price.american','Implied_Probability']].reset_index(drop=True)
                filtered_df.columns = ['Date','Title','Winner','Price','Implied_Probability']
                filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
                f=filtered_df.groupby(['Winner']).agg({'Date':'max','Price': ['last','mean','max','min','count']}).sort_values([('Price', 'mean')], ascending=True).reset_index(drop=False).head(10)
                f=f['Winner']
                filtered_df=filtered_df.loc[filtered_df.Winner.isin(f)]
            line_chart_probability(filtered_df,option,color_map)
            line_chart_probability_initial(filtered_df,option,color_map)
            line_chart(filtered_df,option,color_map)
            # table_output(filtered_df)
            ga('bovada','view_option',option)



if __name__ == "__main__":
    #execute
    app()
