import pandas as pd
import numpy as np
import plotly_express as px
import streamlit as st

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

@st.cache
def load_data_us():
    try:
        df = pd.read_csv('./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    except:
        df = pd.read_csv('./coronavirus_viz/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
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

    try:
        deaths_us = pd.read_csv('./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
    except:
        deaths_us = pd.read_csv('./coronavirus_viz/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
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

def load_data_global():
    try:
        confirmed_all = pd.read_csv('./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    except:
        confirmed_all = pd.read_csv('./coronavirus_viz/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
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

    try:
        deaths_all = pd.read_csv('./csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    except:
        deaths_all = pd.read_csv('./coronavirus_viz/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
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

def daily_growth_all(df):
    a=px.bar(df.loc[df.Date > df.Date.max() - pd.to_timedelta(90, unit='d')].groupby(['Date']).agg({'Confirmed_Growth':'sum'}).reset_index(),
    x='Date',
    y='Confirmed_Growth',
    title='Daily Growth in COVID Cases')
    a.update_layout(showlegend=True)
    st.plotly_chart(a)

def bar_graph_all(df):
    a=px.bar(df.groupby(['Date']).agg({'Confirmed':'sum'}).reset_index(),x='Date',y='Confirmed',title='Confirmed Cases Over Time', width=1400, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("=",": ")))
    st.plotly_chart(a)

def bar_graph_country(df):
    #df['Country_Region'] = df.Country_Region.str[:15]
    df['Country'] = df.Country_Region.str[:15]
    a=px.bar(df.loc[df.Date > df.Date.max() - pd.to_timedelta(90, unit='d')].groupby(['Date','Country']).agg({'Confirmed_Growth':'sum'}).reset_index().sort_values('Confirmed_Growth',ascending=False),x='Date',y='Confirmed_Growth',color='Country', title = 'Daily Growth in COVID Cases by Country')
    a.update_layout(showlegend=True)
    #a.update_layout(width=800,paper_bgcolor="LightSteelBlue",autosize=False)
    #legend=dict(x=2.2, y=1.0))
    #legend=dict(xanchor='center',yanchor='top',x=0.5,y=-1.3)
    st.plotly_chart(a)

def bar_graph_confirmed_growth(df):
    a=px.bar(df.groupby(['Date','Country_Region']).agg({'Confirmed_Growth':'sum'}).reset_index().sort_values('Confirmed_Growth',ascending=False),
       x='Date',y='Confirmed_Growth',color='Country_Region', title = 'Confirmed Growth (Day over Day) by Country',width=1200, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("=",": ")))
    st.plotly_chart(a)

def bar_graph_deaths(df):
    a=px.bar(df.groupby(['Date']).agg({'Deaths':'sum'}).reset_index(),x='Date',y='Deaths',title='Deaths Over Time',width=1400, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("=",": ")))
    st.plotly_chart(a)

def bar_graph_deaths_growth(df):
    a=px.bar(df.groupby(['Date','Country_Region']).agg({'Deaths_Growth':'sum'}).reset_index().sort_values('Deaths_Growth',ascending=False),
       x='Date',y='Deaths_Growth',color='Country_Region', title = 'Death Growth by Country',width=1400, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("=",": ")))
    st.plotly_chart(a)

def bar_graph_confirmed_state(df):
    a=px.bar(df.loc[df['Date'] > '2020-03-15'].groupby(['Date','Province_State']).agg({'Confirmed':'sum'}).reset_index().sort_values('Confirmed',ascending=False),
       x='Date',y='Confirmed',color='Province_State', title = 'Confirmed by US State',width=1400, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("=",": ")))
    st.plotly_chart(a)

def bar_graph_confirmed_growth_state(df):
    a=px.bar(df.loc[df['Date'] > '2020-03-15'].groupby(['Date','Province_State']).agg({'Confirmed_Growth':'sum'}).reset_index().sort_values('Confirmed_Growth',ascending=False),
       x='Date',y='Confirmed_Growth',color='Province_State', title = 'Confirmed Growth (Day over Day) by State',width=1400, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("=",": ")))
    st.plotly_chart(a)

def scatter_deaths_confirmed(df):
    p=df.groupby(['Date','Country_Region']).agg({'Deaths':'sum','Confirmed':'sum'}).reset_index()
    a=px.scatter(p.groupby(['Country_Region']).agg({'Deaths':'last','Confirmed':'last'}).reset_index(),x='Confirmed',y='Deaths',color='Country_Region',title='Confirmed Cases vs. Deaths by Country')
    a.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    st.plotly_chart(a)

def rolling_avg(df):
    d=df.loc[df.Date > df.Date.max() - pd.to_timedelta(90, unit='d')].groupby(['Date']).agg({'Confirmed_Growth':'sum'}).reset_index()
    d['Rolling_Avg'] = d['Confirmed_Growth'].rolling(window=7).mean()
    d1 = d.melt(id_vars=['Date']+list(d.keys()[5:]), var_name='val')
    fig=px.line(d1, x='Date', y='value', color='val',title='Daily COVID Cases vs. 7 Day Rolling Average', width=800)
    #fig.update_traces(mode='lines')
    fig.update_traces(mode='lines+markers',
                      marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    st.plotly_chart(fig)

def rolling_avg_deaths(df):
    d=df.loc[(df.Date > df.Date.max() - pd.to_timedelta(90, unit='d'))].groupby(['Date']).agg({'Deaths_Growth':'sum'}).reset_index()
    d['Rolling_Avg'] = d['Deaths_Growth'].rolling(window=7).mean()
    d1 = d.melt(id_vars=['Date']+list(d.keys()[5:]), var_name='val')
    fig=px.line(d1, x='Date', y='value', color='val',title='Daily COVID Deaths vs. 7 Day Rolling Average')
    #fig.update_traces(mode='lines')
    fig.update_traces(mode='lines+markers',
                      marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    st.plotly_chart(fig)

def state_rolling_avg(df, state):
    d=df.loc[(df.Date > df.Date.max() - pd.to_timedelta(90, unit='d')) & (df.Province_State == state)].groupby(['Date']).agg({'Confirmed_Growth':'sum'}).reset_index()
    d['Rolling_Avg'] = d['Confirmed_Growth'].rolling(window=7).mean()
    d1 = d.melt(id_vars=['Date']+list(d.keys()[5:]), var_name='val')
    fig=px.line(d1, x='Date', y='value', color='val',title='Daily COVID Cases vs. 7 Day Rolling Average for '+state)
    #fig.update_traces(mode='lines')
    fig.update_traces(mode='lines+markers',
                      marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    st.plotly_chart(fig)

def mortality_rate(df):
    d=df.groupby(['Date']).agg({'Deaths':'sum','Confirmed':'sum'}).reset_index()
    d['Mortality_Rate'] = (d.Deaths / d.Confirmed)*100

    fig=px.scatter(d.tail(30),x='Date',y='Mortality_Rate',width=800,title='Mortality Rate Over Time (Last 30 Days)',text='Mortality_Rate')
    fig.update_traces(textposition='bottom center',
                      texttemplate = '%{text:.2f}',
                      textfont_size=10,
                      marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    st.plotly_chart(fig)


def state_deaths_rolling_avg(df, state):

    d=df.loc[(df.Date > df.Date.max() - pd.to_timedelta(60, unit='d')) & (df.Province_State == state)].groupby(['Date']).agg({'Deaths_Growth':'sum'}).reset_index()
    d['Rolling_Avg'] = d['Deaths_Growth'].rolling(window=7).mean()
    d1 = d.melt(id_vars=['Date']+list(d.keys()[5:]), var_name='val')
    fig=px.line(d1, x='Date', y='value', color='val',title='Daily COVID Deaths vs. 7 Day Rolling Average for '+state)
    #fig.update_traces(mode='lines')
    fig.update_traces(mode='lines+markers',
                      marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    st.plotly_chart(fig)

def coronavirus_viz():
    main()

def main():
    #_max_width_()
    st.title("Coronavirus-Viz")
    st.markdown("Feel free to explore the data. Click on an item in the legend to filter it out -- double-click an item to filter down to just that item. Or click and drag to filter the view so that you only see the range you are looking for.")
    st.write("If you have any feedback or questions, feel free to get at me on the [Twitter] (https://www.twitter.com/aaroncolesmith) machine")

    df_us = load_data_us()
    df_all = load_data_global()
    #Daily Growth in COVID Cases
    daily_growth_all(df_all)
    #Daily Growth in COVID Cases by Country
    bar_graph_country(df_all)
    #Yesterday's Top Growth Factor by Country
    rolling_avg(df_all)
    rolling_avg_deaths(df_all)
    mortality_rate(df_all)

    a=df_us['Province_State'].unique()
    a=np.insert(a,0,'')
    option=st.selectbox('Select a State to view data', a)
    if len(option) > 0:
        state=option
        state_rolling_avg(df_us, state)
        state_deaths_rolling_avg(df_us, state)

if __name__ == "__main__":
    #execute
    main()






# a = px.bar(df.groupby(['observationdate','country_region']).agg({'confirmed':sum}).reset_index(drop=False).sort_values(['confirmed'],ascending=False), x='observationdate',y='confirmed',color='country_region', title='Observations Over Time',width=1400, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("country_region=","")))
# st.plotly_chart(a)
#
# b = px.bar(df.groupby(['observationdate','country_region']).agg({'deaths':sum}).reset_index(drop=False).sort_values(['deaths'],ascending=False), x='observationdate',y='deaths',color='country_region', title='Deaths Over Time',width=1400, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("country_region=","")))
# st.plotly_chart(b)
#
# c = px.scatter(df.groupby(['observationdate','country_region']).agg({'confirmed':sum,'deaths':sum}).reset_index(drop=False).groupby(['country_region']).agg({'confirmed':'last','deaths':'last'}).reset_index(),x='confirmed',y='deaths',color='country_region', title='Confirmed Cases vs. Deaths by Country').for_each_trace(lambda t: t.update(name=t.name.replace("country_region=","")))
# c.update_traces(marker=dict(size=12,
#                               line=dict(width=2,
#                                         color='DarkSlateGrey')),
#                   selector=dict(mode='markers'))
# st.plotly_chart(c)
#
# d=px.bar(df[(df.country_region == 'US') & (df.observationdate > '2020-02-28')].groupby(['observationdate','province_state']).agg({'confirmed':sum}).reset_index(drop=False).sort_values(['confirmed'],ascending=False), x='observationdate',y='confirmed',color='province_state', title='US Observations by State')
# st.plotly_chart(d)
#
# dfg=df.groupby(['observationdate','country_region']).agg({'confirmed':sum}).reset_index(drop=False)
#
# dfg.loc[dfg.confirmed >= 100, 'over_100'] = 1
# dfg.loc[dfg.confirmed < 100, 'over_100'] = 0
#
# g=pd.concat([dfg.loc[dfg.country_region == 'Mainland China'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'US'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'South Korea'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'Japan'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'Italy'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'Iran'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'Singapore'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'France'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'Japan'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'Germany'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'Spain'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'Hong Kong'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index(),
#          dfg.loc[dfg.country_region == 'UK'].groupby(['observationdate','country_region','confirmed']).sum().cumsum().reset_index()
#            .reset_index()], sort=False)
#
#
#
# t=px.line(g.loc[g.over_100 > 0],x='over_100',y='confirmed',color='country_region', title='Confirmed Cases Since the 100th Observation',width=1400, height=600).for_each_trace(lambda t: t.update(name=t.name.replace("country_region=","")))
# t.update_xaxes(title='Days Since 100th Observation')
# t.update_yaxes(title='Confirmed Cases')
# t.update_traces(mode='lines+markers')
# st.plotly_chart(t)
#
#
#
