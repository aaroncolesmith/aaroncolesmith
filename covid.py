import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import streamlit as st
import datetime
from covid_functions import load_data_us, load_data_global, bar_graph, bar_graph_dimension, rolling_avg, header, rolling_avg_pct_change, ga

def main_dash(df, report_date, days_back):

    header(report_date)

    metric = 'Confirmed_Growth'
    dimension='Country'
    width=800
    bar_graph(df,days_back,metric,800,'Daily Growth in COVID Cases')
    bar_graph_dimension(df,days_back,metric,'Country',800,'Daily Growth in COVID Cases by Country')
    rolling_avg(df,days_back,metric,800,'7 Day Rolling Avg of COVID Cases')
    rolling_avg_pct_change(df, metric, dimension, days_back,800,'Rolling Avg. vs. % Change in COVID Cases')

    metric = 'Deaths_Growth'
    bar_graph(df,days_back,metric,800,'Daily Growth in COVID Deaths')
    bar_graph_dimension(df,days_back,metric,'Country',800,'Daily Growth in COVID Deaths by Country')
    rolling_avg(df,days_back,metric,800,'7 Day Rolling Avg of COVID Deaths')
    rolling_avg_pct_change(df, metric, dimension, days_back,800,'Rolling Avg. vs. % Change in COVID Deaths')

def country(df, report_date, days_back):

    header(report_date)

    dimension='Country'
    a=df[dimension].unique()
    a=np.insert(a,0,'')

    selection=st.selectbox('Select a Country to view data', a)

    if len(selection) > 0:
        st.write('# '+selection)
        filter_view(df, dimension, selection, days_back)

def state(df, report_date, days_back):

    header(report_date)

    metric = 'Confirmed_Growth'
    dimension='State'
    width=800

    bar_graph_dimension(df,days_back,metric,dimension,800,'Daily Growth in COVID Cases by US State')
    rolling_avg_pct_change(df, metric, dimension, days_back,800,'Rolling Avg. vs. % Change in COVID Cases')

    metric = 'Deaths_Growth'
    bar_graph_dimension(df,days_back,metric,dimension,800,'Daily Growth in COVID Deaths by US State')
    rolling_avg_pct_change(df, metric, dimension, days_back,800,'Rolling Avg. vs. % Change in COVID Deaths')

    days_back=90
    dimension='State'
    a=df[dimension].unique()
    a=np.insert(a,0,'')

    selection=st.selectbox('Select a State to view data', a)

    if len(selection) > 0:
        st.write('# '+selection)
        filter_view(df, dimension, selection, days_back)

def county(df, report_date, days_back):

    header(report_date)
    width=800

    metric = 'Confirmed_Growth'
    dimension = 'Combined_Key'
    rolling_avg_pct_change(df.loc[df[metric] > 100], metric, dimension, days_back,800,'Rolling Avg. vs. % Change in COVID Cases')

    dimension='State'
    a=df[dimension].unique()
    a=np.insert(a,0,'')

    selection=st.selectbox('Select a State to view data', a)

    if len(selection) > 0:
        st.write('# '+selection)
        df=df.loc[df[dimension] == selection]

        metric = 'Confirmed_Growth'
        dimension = 'Combined_Key'
        bar_graph_dimension(df,days_back,metric,dimension,800,'Daily Growth in COVID Cases by '+selection+'\'s County')
        rolling_avg_pct_change(df, metric, dimension, days_back,800,'Rolling Avg. vs. % Change in COVID Cases')

        metric = 'Deaths_Growth'
        bar_graph_dimension(df,days_back,metric,dimension,800,'Daily Growth in COVID Deaths by '+selection+'\'s County')
        rolling_avg_pct_change(df, metric, dimension, days_back,800,'Rolling Avg. vs. % Change in COVID Deaths')

        b=df[dimension].unique()
        b=np.insert(b,0,'')

        county_selection=st.selectbox('Select a County to view data', b)
        if len(county_selection) > 0:
            filter_view(df, dimension, county_selection, days_back)


def filter_view(df, dimension, selection, days_back):

    metric = 'Confirmed_Growth'
    df=df.loc[df[dimension] == selection]
    bar_graph(df, days_back, metric, 800, 'Daily Growth in COVID Cases')
    rolling_avg(df,days_back,metric,800,'7 Day Rolling Avg of COVID Cases')
    metric = 'Deaths_Growth'
    bar_graph(df,days_back,metric,800,'Daily Growth in COVID Deaths')
    rolling_avg(df,days_back,metric,800,'7 Day Rolling Avg of COVID Deaths')

    ga('Coronavirus-Viz',dimension, selection)

def app():

    # df_us = load_data_us()
    df_all = load_data_global()
    report_date = df_all.Date.dt.date.max()
    ga('Coronavirus-Viz','Page Load', 'Page Load')

    days_back = (datetime.datetime.now() - df_all.Date.min()).days
    st.sidebar.markdown('## COVID-Viz Navigation')
    radio_selection = st.sidebar.radio('Select a page:',['Main Dashboard','Breakdown by Country'])
    days_back = st.sidebar.slider('How many days back',30,days_back,90)

    if radio_selection == 'Main Dashboard':
        main_dash(df_all, report_date, days_back)
    if radio_selection == 'Breakdown by Country':
        ga('Coronavirus-Viz','Page Load', radio_selection)
        country(df_all, report_date, days_back)
    if radio_selection == 'Breakdown by US State':
        ga('Coronavirus-Viz','Page Load', radio_selection)
        state(df_us, report_date, days_back)
    if radio_selection == 'Breakdown by US County':
        ga('Coronavirus-Viz','Page Load', radio_selection)
        county(df_us, report_date, days_back)

if __name__ == "__main__":
    #execute
    main()
