#UPDTATED VERSION

import os
import streamlit as st
import pandas as pd
import requests
import plotly_express as px

APP_URL = os.environ["APP_URL"]
STRAVA_CLIENT_ID = os.environ["STRAVA_CLIENT_ID"]
STRAVA_CLIENT_SECRET = os.environ["STRAVA_CLIENT_SECRET"]

@st.cache(show_spinner=False, suppress_st_warning=True)
def get_strava_auth(query_params):
    authorization_code = query_params.get("code", [None])[0]

    payload = {
    'client_id': STRAVA_CLIENT_ID,
    'client_secret': STRAVA_CLIENT_SECRET,
    'code': authorization_code,
    "grant_type": "authorization_code"
    }

    r=requests.post('https://www.strava.com/oauth/token',params=payload)
    return r.json()

    auth=response.json()
    return auth

@st.cache(show_spinner=False, suppress_st_warning=True)
def load_strava_data(auth):
    with st.spinner('Loading your activities...'):
        df=pd.DataFrame()
        header = {'Authorization': 'Bearer ' + auth['access_token']}
        activites_url = "https://www.strava.com/api/v3/athlete/activities"
        i=1
        size1=0
        size2=1
        while size1!=size2:
            size1=df.index.size
            param = {'per_page': 200, 'page': i}
            data = requests.get(activites_url, headers=header, params=param).json()
            df=pd.concat([df,pd.DataFrame(data)])
            size2=df.index.size
            i+=1
        df['distance_miles'] = df['distance']/1609
        df['elapsed_time_min']=pd.to_timedelta(df['elapsed_time']).astype('timedelta64[s]').astype(int)/60
        df['moving_time_min']=pd.to_timedelta(df['moving_time']).astype('timedelta64[s]').astype(int)/60
        df['date'] = pd.to_datetime(df['start_date']).dt.date
        df['elapsed_time_hours'] = round(df['elapsed_time'] / 3600,2)
        return df






def app():

    st.title('Strava Viz')
    st.markdown('##### Welcome to Strava Viz -- login to your Strava account to see a dashboard visualizing all of your Strava activities!')

    query_params = st.experimental_get_query_params()
    st.write(query_params)
    if query_params:
        auth = get_strava_auth(query_params)
        st.write('Welcome '+auth['athlete']['firstname'])

        df = load_strava_data(auth)

        st.write('You have a total of '+str(df.index.size)+' activities in your Strava account!')

        st.write(df.head(5))

        d=df.groupby([df['date'].values.astype('datetime64[M]'),df['type']]).size().to_frame('activities').reset_index().copy()
        d.columns = ['month','type','workouts']

        fig=px.bar(d,
            x='month',
            y='workouts',
            color='type',
            title='Total Number of Activities by Month',
            )
        fig.update_traces(
            hovertemplate='%{x} - %{y} activities'
        )
        fig.update_yaxes(title='# of Activities',
                       showgrid=False,
                      )
        fig.update_layout(plot_bgcolor='white')
        fig.update_xaxes(title='Date',
                      showgrid=False,
                      )
        st.plotly_chart(fig)
        del d



        fig=px.scatter(df.loc[df.distance>0],
                  x='start_date',
                  y='distance_miles',
                  color='type',
                  title='Workout Distance by Workout Type',
                  hover_data=['type','name','date','id'])
        fig.update_traces(mode='markers',
                          marker=dict(size=8,
                                      line=dict(width=1,
                                                color='DarkSlateGrey')))
        fig.update_xaxes(title='Date',showgrid=False)
        fig.update_yaxes(title='Workout Distance (Miles)',showgrid=False)
        fig.update_layout(plot_bgcolor='white')
        fig.update_traces(
            hovertemplate='%{x} - %{customdata[0]}<br>%{customdata[1]}<br>%{y:.1f} miles<extra></extra>'
        )
        st.plotly_chart(fig)



        st.markdown('### Build Your Own Scatter Plot')
        columns=df.columns

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        num_columns = df.select_dtypes(include=numerics).columns.tolist()

        workout_options = df['type'].unique()
        selected_workouts = st.multiselect('Select Workouts', workout_options, workout_options)
        df = df[df['type'].isin(selected_workouts)]

        x_axis = st.selectbox('Select values on the X Axis',
            columns,
            58)

        y_axis = st.selectbox('Select values on the Y Axis',
            columns,
            36)

        hover_data = st.multiselect('Select values on that will appear on hover',
            columns)

        print('X Axis: ' + x_axis)
        print('Y Axis: ' + y_axis)

        fig=px.scatter(df.loc[df.distance>0],
                  x=x_axis,
                  y=y_axis,
                  color='type',
                  title='Build Your Own Scatterplot',
                  hover_data=hover_data)
        fig.update_traces(mode='markers',
                          marker=dict(size=8,
                                      line=dict(width=1,
                                                color='DarkSlateGrey')))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(plot_bgcolor='white')
        # fig.update_traces(
        #     hovertemplate='%{x} - %{customdata[0]}<br>%{customdata[1]}<br>%{y:.1f} miles<extra></extra>'
        # )
        st.plotly_chart(fig)


        st.write(num_columns)
        st.write(columns)

        # distance, moving_time, elapsed_time, total_elevation_gain, start_date, achievement_count, kudos_count, average_speed, max_speed, average_cadence, average_temp, average_heartrate, max_heartrate, elev_high, elev_low, pr_count, average_watts, kilojoules, distance_miles, elapsed_time_hours





    else:

        payload={"client_id": STRAVA_CLIENT_ID,"redirect_uri": APP_URL,"response_type": "code","approval_prompt": "auto","scope": "activity:read_all"}
        r=requests.post('https://www.strava.com/oauth/authorize',params=payload)
        st.markdown("<a href=\""+str(r.url)+"\">Click here to login to Strava!</a>",
        unsafe_allow_html=True,)


if __name__ == "__main__":
    #execute
    app()

