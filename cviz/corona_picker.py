import pandas as pd
import numpy as np
import plotly_express as px
import streamlit as st

@st.cache
def load_data_us():
    df = pd.read_csv('./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    df = df.drop(df.columns[0:5], axis=1)
    df=df.set_index(['Admin2','Province_State','Country_Region','Combined_Key','Lat','Long_']).stack().reset_index()
    df.columns=['Admin2','Province_State','Country_Region','Combined_Key','Lat','Long_','Date','Confirmed']

    df['Date']=pd.to_datetime(df.Date)
    tmp = df.groupby(['Combined_Key']).agg({'Confirmed':'sum'}).reset_index().sort_values('Confirmed',ascending=False).head(30)
    l = tmp.Combined_Key.unique()
    df.loc[df.Combined_Key.isin(l),'top_city'] = df.Combined_Key
    df['top_city']=df.top_city.fillna('Other')
    df['Confirmed_Growth'] = df['Confirmed']-df['Confirmed'].shift(1)
    df.loc[df.Combined_Key != df.Combined_Key.shift(1),'Confirmed_Growth'] = 0
    df['Confirmed_Growth_Pct'] = (df['Confirmed']-df['Confirmed'].shift(1))/df['Confirmed'].shift(1)
    df.loc[df.Combined_Key != df.Combined_Key.shift(1),'Confirmed_Growth_Pct'] = 0
    df['Confirmed_Growth_Pct'] = df.Confirmed_Growth_Pct.fillna(0)

    deaths_us = pd.read_csv('./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
    deaths_us = deaths_us.drop(deaths_us.columns[0:5], axis=1)
    deaths_us=deaths_us.set_index(['Admin2','Province_State','Country_Region','Combined_Key','Lat','Long_']).stack().reset_index()
    deaths_us.columns=['Admin2','Province_State','Country_Region','Combined_Key','Lat','Long_','Date','Deaths']
    us_population = deaths_us.loc[deaths_us.Date == 'Population']
    deaths_us = deaths_us.loc[deaths_us.Date != 'Population']

    deaths_us['Date']=pd.to_datetime(deaths_us.Date)
    tmp = deaths_us.groupby(['Combined_Key']).agg({'Deaths':'sum'}).reset_index().sort_values('Deaths',ascending=False).head(30)
    l = tmp.Combined_Key.unique()
    deaths_us.loc[deaths_us.Combined_Key.isin(l),'top_city'] = deaths_us.Combined_Key
    deaths_us['top_city']=deaths_us.top_city.fillna('Other')
    deaths_us['Deaths_Growth'] = deaths_us['Deaths']-deaths_us['Deaths'].shift(1)
    deaths_us.loc[deaths_us.Combined_Key != deaths_us.Combined_Key.shift(1),'Deaths_Growth'] = 0
    deaths_us['Deaths_Growth_Pct'] = (deaths_us['Deaths']-deaths_us['Deaths'].shift(1))/deaths_us['Deaths'].shift(1)
    deaths_us.loc[deaths_us.Combined_Key != deaths_us.Combined_Key.shift(1),'Deaths_Growth_Pct'] = 0
    deaths_us['Deaths_Growth_Pct'] = deaths_us.Deaths_Growth_Pct.fillna(0)

    df_us = pd.merge(df, deaths_us, left_on=['Combined_Key','Date'],right_on=['Combined_Key','Date'])
    df_us = df_us[['Admin2_x','Province_State_x','Country_Region_x','Combined_Key','top_city_x','Lat_x','Long__x','Date','Confirmed','Confirmed_Growth','Confirmed_Growth_Pct','Deaths','Deaths_Growth','Deaths_Growth_Pct']]
    df_us.columns = ['Admin','Province_State','Country_Region','Combined_Key','Top_City','Lat','Long','Date','Confirmed','Confirmed_Growth','Confirmed_Growth_Pct','Deaths','Deaths_Growth','Deaths_Growth_Pct']

    return df_us

@st.cache
def load_data_global():
    confirmed_all = pd.read_csv('./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    confirmed_all=confirmed_all.set_index(['Province/State','Country/Region','Lat','Long']).stack().reset_index()
    confirmed_all['Combined_Key'] = confirmed_all['Province/State'].fillna('None') + ', ' + confirmed_all['Country/Region']
    confirmed_all.columns=['Province/State','Country/Region','Lat','Long','Date','Confirmed','Combined_Key']
    confirmed_all = confirmed_all[['Province/State','Country/Region','Combined_Key','Lat','Long','Date','Confirmed']]

    confirmed_all['Date']=pd.to_datetime(confirmed_all.Date)

    tmp = confirmed_all.groupby(['Combined_Key']).agg({'Confirmed':'sum'}).reset_index().sort_values('Confirmed',ascending=False).head(30)
    l = tmp.Combined_Key.unique()
    confirmed_all.loc[confirmed_all.Combined_Key.isin(l),'top_city'] = confirmed_all.Combined_Key
    confirmed_all['top_city']=confirmed_all.top_city.fillna('Other')

    confirmed_all['Confirmed_Growth'] = confirmed_all['Confirmed']-confirmed_all['Confirmed'].shift(1)
    confirmed_all.loc[confirmed_all.Combined_Key != confirmed_all.Combined_Key.shift(1),'Confirmed_Growth'] = 0
    confirmed_all['Confirmed_Growth_Pct'] = (confirmed_all['Confirmed']-confirmed_all['Confirmed'].shift(1))/confirmed_all['Confirmed'].shift(1)
    confirmed_all.loc[confirmed_all.Combined_Key != confirmed_all.Combined_Key.shift(1),'Confirmed_Growth_Pct'] = 0
    confirmed_all['Confirmed_Growth_Pct']=confirmed_all['Confirmed_Growth_Pct'].fillna(0)

    deaths_all = pd.read_csv('./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    deaths_all=deaths_all.set_index(['Province/State','Country/Region','Lat','Long']).stack().reset_index()
    deaths_all['Combined_Key'] = deaths_all['Province/State'].fillna('None') + ', ' + deaths_all['Country/Region']
    deaths_all.columns=['Province/State','Country/Region','Lat','Long','Date','Deaths','Combined_Key']
    deaths_all = deaths_all[['Province/State','Country/Region','Combined_Key','Lat','Long','Date','Deaths']]

    deaths_all['Date']=pd.to_datetime(deaths_all.Date)
    tmp = deaths_all.groupby(['Combined_Key']).agg({'Deaths':'sum'}).reset_index().sort_values('Deaths',ascending=False).head(30)
    l = tmp.Combined_Key.unique()
    deaths_all.loc[deaths_all.Combined_Key.isin(l),'top_city'] = deaths_all.Combined_Key
    deaths_all['top_city']=deaths_all.top_city.fillna('Other')
    deaths_all['Deaths_Growth'] = deaths_all['Deaths']-deaths_all['Deaths'].shift(1)
    deaths_all.loc[deaths_all.Combined_Key != deaths_all.Combined_Key.shift(1),'Deaths_Growth'] = 0
    deaths_all['Deaths_Growth_Pct'] = (deaths_all['Deaths']-deaths_all['Deaths'].shift(1))/deaths_all['Deaths'].shift(1)
    deaths_all.loc[deaths_all.Combined_Key != deaths_all.Combined_Key.shift(1),'Deaths_Growth_Pct'] = 0
    deaths_all['Deaths_Growth_Pct'] = deaths_all.Deaths_Growth_Pct.fillna(0)

    df_all = pd.merge(confirmed_all, deaths_all, left_on=['Combined_Key','Date'],right_on=['Combined_Key','Date'])
    df_all = df_all[['Province/State_x','Country/Region_x','Combined_Key','top_city_x','Lat_x','Long_x','Date','Confirmed','Confirmed_Growth','Confirmed_Growth_Pct','Deaths','Deaths_Growth','Deaths_Growth_Pct']]
    df_all.columns = ['Province_State','Country_Region','Combined_Key','Top_City','Lat','Long','Date','Confirmed','Confirmed_Growth','Confirmed_Growth_Pct','Deaths','Deaths_Growth','Deaths_Growth_Pct']

    return df_all

def main():
    df_global = load_data_global()
    df_us = load_data_us()
    dataset = st.sidebar.selectbox(
    'Which dataset?',
    ('','Global','US')
    )

    if dataset == 'Global':
        df = df_global
    else:
        df = df_us

    metric1 = st.sidebar.selectbox(
    'Select a metric -- something like confirmed cases or deaths',
    ('','Confirmed','Confirmed_Growth','Confirmed_Growth_Factor','Deaths','Deaths_Growth')
    )

    metric2 = st.sidebar.selectbox(
    'Select another metrics [Optional]',
    ('','Confirmed','Confirmed_Growth','Confirmed_Growth_Factor','Deaths','Deaths_Growth')
    )

    dimension = st.sidebar.selectbox(
    'Select a dimension -- like Date or Days since 100th Case',
    ('','Date','Days Since 100th Case')
    )

    if (len(dataset) > 0 and len(metric1) > 0 and len(dimension) > 0):
        if st.sidebar.button('Run'):
            st.write('Run' + dataset + metric1 + metric2 + dimension)
            st.write(df.head(5))
            bar_graph(df, metric1, dimension)
        else:
            st.sidebar.markdown('Select parameters and run to see some visualizations')

def bar_graph(df, metric, dimension):
    fig=px.bar(df.groupby([dimension]).agg({metric:'sum'}).reset_index(),x=dimension,y=metric)
    st.plotly_chart(fig)


if __name__ == "__main__":
    #execute
    main()
