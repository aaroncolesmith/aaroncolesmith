import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go


def density_map_agg(d):

    fig = px.density_mapbox(d, 
                        lat='LATITUDE', 
                        lon='LONGITUDE', 
                        z='COUNT',
                        radius=25,
                        center=dict(lat=pd.to_numeric(d['LATITUDE'],errors='coerce').mean(), lon=pd.to_numeric(d['LONGITUDE'],errors='coerce').mean()), 
                        zoom=10,
                        opacity=.90, 
                        height=600,
                        hover_data=['ADDRESS','CRIME','LAST_DATE'],
                        mapbox_style="stamen-terrain")
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br><br>Crime Count: %{z}<br>Coordinates: (%{lat},%{lon})<br>Address: %{customdata[0]}<br>Last date: %{customdata[2]|%-m/%-d %-I:%M%p}')
    st.plotly_chart(fig)

def density_map_day(d):

    fig = px.density_mapbox(d, 
                            lat='LATITUDE', 
                            lon='LONGITUDE', 
                            hover_name='CRIME',
                            hover_data=['ADDRESS','COUNT','LAST_DATE'],
                            animation_frame=d['DAY'].astype('str'),
                            zoom=10,
                            radius=25,
                            opacity=.90, 
                            center=dict(lat=pd.to_numeric(d['LATITUDE'],errors='coerce').mean(), lon=pd.to_numeric(d['LONGITUDE'],errors='coerce').mean()),
                            height=600)
    fig.update_layout(mapbox_style="stamen-terrain")
    custom_template = '<b>%{hovertext}</b><br><br>Crime Count: %{customdata[1]}<br>Coordinates: (%{lat},%{lon})<br>Address: %{customdata[0]}<br>Last date: %{customdata[2]|%-m/%-d %-I:%M%p}'
    fig.update_traces(hovertemplate=custom_template)
    for frame in fig.frames:
        frame.data[0].hovertemplate = custom_template
    fig['layout']['sliders'][0]['currentvalue']['prefix'] = 'Date: '
    st.plotly_chart(fig)

def scatter_map_agg(d):

    fig = px.scatter_mapbox(d, 
                            lat='LATITUDE', 
                            lon='LONGITUDE', 
                            hover_name='CRIME',
                            hover_data=['ADDRESS','COUNT','LAST_DATE'],
                            size='COUNT_SCALED',
                            color_discrete_sequence=["fuchsia"],
                            opacity=.6,
                            size_max=50, 
                            center=dict(lat=pd.to_numeric(d['LATITUDE'],errors='coerce').mean(), lon=pd.to_numeric(d['LONGITUDE'],errors='coerce').mean()),
                            zoom=10, 
                            height=600)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br><br>Crime Count: %{customdata[1]}<br>Coordinates: (%{lat},%{lon})<br>Address: %{customdata[0]}<br>Last date: %{customdata[2]|%-m/%-d %-I:%M%p}')
    st.plotly_chart(fig)

def scatter_map_day(d):

    fig = px.scatter_mapbox(d, 
                            lat='LATITUDE', 
                            lon='LONGITUDE', 
                            hover_name='CRIME',
                            hover_data=['ADDRESS','COUNT','LAST_DATE'],
                            size='COUNT_SCALED',
                            animation_group='LAT_LON',
                            animation_frame=d['DAY'].astype('str'),
                            color_discrete_sequence=["fuchsia"],
                            opacity=.6,
                            center=dict(lat=pd.to_numeric(d['LATITUDE'],errors='coerce').mean(), lon=pd.to_numeric(d['LONGITUDE'],errors='coerce').mean()),
                            size_max=100, 
                            zoom=10, 
                            height=600)
    fig.update_layout(mapbox_style="stamen-terrain")
    custom_template = '<b>%{hovertext}</b><br><br>Crime Count: %{customdata[1]}<br>Coordinates: (%{lat},%{lon})<br>Address: %{customdata[0]}<br>Last date: %{customdata[2]|%-m/%-d %-I:%M%p}'
    fig.update_traces(hovertemplate=custom_template)
    for frame in fig.frames:
        frame.data[0].hovertemplate = custom_template
    fig['layout']['sliders'][0]['currentvalue']['prefix'] = 'Date: '
    st.plotly_chart(fig)

def group_data_agg(df):
    d=df.groupby(['LATITUDE','LONGITUDE','ADDRESS']).agg({'DATE_CRIME': lambda x: '<br>'.join(x),
                                            'ID': 'size',
                                            'DATE':'max'}).reset_index()

    d.columns = ['LATITUDE','LONGITUDE','ADDRESS','CRIME','COUNT','LAST_DATE']
    d['LATITUDE'] = pd.to_numeric(d['LATITUDE'])
    d['LONGITUDE'] = pd.to_numeric(d['LONGITUDE'])
    d['COUNT_SCALED'] = d['COUNT']*3

    d['LAT_LON'] = d['LATITUDE'].astype('str') + ', ' +  d['LONGITUDE'].astype('str')

    # if the crime desciption is greater than 500 characters, cut it off at 500 characters
    d['CRIME'] = d['CRIME'].apply(lambda x: x[:1500] + '...' if len(x) > 1500 else x)

    return d

def group_data_day(df):
    d=df.groupby(['LATITUDE','LONGITUDE','ADDRESS','DAY']).agg({'DATE_CRIME': lambda x: '<br>'.join(x),
                                            'ID': 'size',
                                            'DATE':'max'}).reset_index()
    d.columns = ['LATITUDE','LONGITUDE','ADDRESS','DAY','CRIME','COUNT','LAST_DATE']
    d['LATITUDE'] = pd.to_numeric(d['LATITUDE'])
    d['LONGITUDE'] = pd.to_numeric(d['LONGITUDE'])
    d['LAT_LON'] = d['LATITUDE'].astype('str') + ', ' +  d['LONGITUDE'].astype('str')

    # if the crime desciption is greater than 500 characters, cut it off at 500 characters
    d['CRIME'] = d['CRIME'].apply(lambda x: x[:1500] + '...' if len(x) > 1500 else x)
    d['DAY']=d['DAY'].dt.tz_localize(None).dt.to_pydatetime()
    d['COUNT_SCALED'] = d['COUNT']*5

    d=d.sort_values('DAY',ascending=True).reset_index(drop=True)

    return d

def crime_cnt_rolling_avg(df):
    df2=df.groupby('HOUR').size().to_frame('CRIME_CNT').reset_index()
    df2['ROLLING_24'] = df2['CRIME_CNT'].rolling(window=24).mean()

    fig=go.Figure()
    fig.add_trace(
        go.Scatter(x=df2.HOUR,
                y=df2.ROLLING_24,
                name='Rolling Avg',
                mode='lines',
                line_shape='spline',
                marker_color='#626EF6',
                marker=dict(
                    size=4,
                    line=dict(
                        width=1,
                        color='#1320B2'
                        )
                    )
                )
        )

    fig.add_trace(
        go.Scatter(x=df2.HOUR,
                y=df2.CRIME_CNT,
                name='Crime Count',
                mode='markers',
                marker_color='#626EF6',
                opacity=.5,
                marker=dict(
                    size=8,
                    line=dict(
                        width=1,
                        color='#1320B2'
                        )
                    )
                )
        )
    st.plotly_chart(fig)

def app():
    df = pd.read_csv('https://raw.githubusercontent.com/aaroncolesmith/portland_crime_map/main/data.csv')
    df['DATE'] = pd.to_datetime(df['DATE'],utc=True)

    # Convert date from UTC to PST
    df['DATE'] = df['DATE'].dt.tz_convert('US/Pacific')
    df['HOUR'] = df['DATE'].dt.floor('h')
    df['DAY'] = df['DATE'].dt.floor('d')
    df['DATE_CRIME'] = df['DATE'].dt.strftime('%-m/%-d %-I:%M%p').astype('str') + ' - ' + df['CRIME']

    st.title('Portland Crime Map')
    st.markdown('Updated as of: ' + str(df['DATE'].max().strftime('%-m/%-d %-I:%M%p')))
    st.markdown('This app is a Streamlit dashboard that shows the number of crimes in Portland, Oregon.')

    st.subheader('Crime Map')

    c1, c2 = st.columns(2)

    # Selector to view aggregate data or by day
    view_type = c1.selectbox('View by', ['Day', 'All-Time'])

    # Selector to view density_map or scatter_map
    map_type = c2.selectbox('Map Type', ['Scatter Map', 'Density Map'])

    if view_type == 'All-Time':
        d=group_data_agg(df)
        if map_type == 'Density Map':
            density_map_agg(d)
        elif map_type == 'Scatter Map':
            scatter_map_agg(d)

    if view_type == 'Day':
        d=group_data_day(df)
        if map_type == 'Density Map':
            density_map_day(d)
        elif map_type == 'Scatter Map':
            scatter_map_day(d)

    # Scatter plot of crime counts by hour
    # df['HOUR']=df['DATE'].dt.floor('h')
    # fig = px.scatter(df.groupby('HOUR').size().to_frame('COUNT').reset_index(), x='HOUR', y='COUNT')
    # st.plotly_chart(fig)

    crime_cnt_rolling_avg(df)


    # Stacked bar chart over by by crime type
    fig = px.bar(df.groupby(['HOUR','CRIME']).size().to_frame('COUNT').reset_index(), x='HOUR', y='COUNT', color='CRIME')
    st.plotly_chart(fig)  

    st.write(df.tail(5))




if __name__ == "__main__":
    #execute
    app()