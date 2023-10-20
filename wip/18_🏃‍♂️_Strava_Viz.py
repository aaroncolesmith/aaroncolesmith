#UPDTATED VERSION

import os
import streamlit as st
import pandas as pd
import requests
import plotly_express as px
import extra_streamlit_components as stx

# APP_URL = os.environ["APP_URL"]
# STRAVA_CLIENT_ID = os.environ["STRAVA_CLIENT_ID"]
# STRAVA_CLIENT_SECRET = os.environ["STRAVA_CLIENT_SECRET"]

APP_URL = 'http://localhost/'
APP_URL = 'http://localhost:8502/Strava_Viz'
STRAVA_CLIENT_ID = '31759'
STRAVA_CLIENT_SECRET = '2598b3d943c65ea4ba6fccd1be1c4c20d246f534'


@st.cache(allow_output_mutation=True)
def get_manager():
    return stx.CookieManager()

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

@st.cache(show_spinner=False, suppress_st_warning=True)
def load_strava_data(auth):
    with st.spinner('Loading your activities...'):
        df=pd.DataFrame()
        header = {'Authorization': 'Bearer ' + auth['access_token']}
        st.write(auth['access_token'])
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
            st.write(size1)

        df['distance_miles'] = df['distance']/1609
        df['elapsed_time_min']=df['elapsed_time']/60
        df['moving_time_min']=df['moving_time']/60
        df['date'] = pd.to_datetime(df['start_date']).dt.date
        df['elapsed_time_hours'] = round(df['elapsed_time'] / 3600,2)
        df['min_per_mile'] = df['moving_time_min']/df['distance_miles']

        return df


def strava_login(cookie_manager):
    payload={"client_id": STRAVA_CLIENT_ID,"redirect_uri": APP_URL,"response_type": "code","approval_prompt": "auto","scope": "activity:read_all"}
    r=requests.post('https://www.strava.com/oauth/authorize',params=payload)
    st.markdown("<a href=\""+str(r.url)+"\" target = \"_self\">Click here to login to Strava!</a>",
    unsafe_allow_html=True,)

    query_params = st.experimental_get_query_params()
    st.write(query_params)

    if query_params.get("code"):

        # authorization_code = query_params.get("code", [None])[0]
        authorization_code = query_params['code'][0]
        st.write(authorization_code)

        payload = {
        'client_id': STRAVA_CLIENT_ID,
        'client_secret': STRAVA_CLIENT_SECRET,
        'code': authorization_code,
        "grant_type": "authorization_code"
        }
        st.write(payload)
        r=requests.post('https://www.strava.com/oauth/token',params=payload)
        st.write(r.url)
        st.write('this is request post json')
        st.write(r.json())




        auth=r.json()
        st.write('this is auth')
        st.write(auth)

        access_token=r.json()['access_token']
        st.write(f'this is access token {access_token}')

        if query_params:
            cookie_manager.set('strava_auth_code', authorization_code, key='0')
            cookie_manager.set('strava_auth', auth, key='1')

            return cookie_manager







def app():

    st.title('Strava Viz')
    st.markdown('##### Welcome to Strava Viz -- login to your Strava account to see a dashboard visualizing all of your Strava activities!')

    cookie_manager = get_manager()
    cookies = cookie_manager.get_all()

    st.write(cookies)

    if st.button("Log Out"):
        cookie_manager.delete('strava_auth_code',key='0')
        cookie_manager.delete('strava_auth',key='1')

        st.experimental_set_query_params()

        st.write("You are logged out!")

    try:
        cookies['strava_auth_code']
        st.write('Auth exists')

        auth=cookies['strava_auth']
        # try:
        #     ## If errors exists, clear cookies and query params
        #     st.write(cookies['strava_auth']['errors'])
        #     cookie_manager.delete('strava_auth_code',key='0')
        #     cookie_manager.delete('strava_auth',key='1')

        #     st.experimental_set_query_params()
        # except:
        #     pass
        st.write('Welcome '+cookies['strava_auth']['athlete']['firstname'])
        st.write(auth['access_token'])


        # header = {'Authorization': 'Bearer ' + auth['access_token']}
        # activites_url = "https://www.strava.com/api/v3/athlete/activities"
        # param = {'per_page': 200, 'page': 1}

        # df=pd.DataFrame()
        # d=requests.get(activites_url, headers=header, params=param).json()
        # df=pd.concat([df,pd.DataFrame(d)])
        # st.write(df.head(5))

        # df=pd.DataFrame()
        # header = {'Authorization': 'Bearer ' + auth['access_token']}
        # st.write(auth['access_token'])
        # activites_url = "https://www.strava.com/api/v3/athlete/activities"
        # i=1
        # size1=0
        # size2=400
        
        # while size1!=size2:
        #     size1=df.index.size
        #     param = {'per_page': 200, 'page': i}
        #     data = requests.get(activites_url, headers=header, params=param).json()
        #     df=pd.concat([df,pd.DataFrame(data)])
        #     size2=df.index.size
        #     i+=1
        #     st.write(size1)

        df = load_strava_data(auth)
        st.write('You have a total of '+str(df.index.size)+' activities in your Strava account!')

        fig=px.scatter(df.loc[df.distance>0],x='distance_miles',y='average_speed',color='type',title='Avg Speed vs. Total Miles by Type',hover_data=['name','date','id'])
        fig.update_traces(mode='markers',marker=dict(size=8,line=dict(width=1,color='DarkSlateGrey')))
        st.plotly_chart(fig,use_container_width=True)

        # st.write(df.head(5))

        # st.write(cookies)

    except Exception as e:
        st.write('Error')
        st.write(e)
        st.write('no auth')
        cookie_manager = strava_login(cookie_manager)

        # cookies = cookie_manager.get_all()

        # st.write(cookies)





    # try:
    #     cookies['strava_auth_code']
    #     st.write('Auth exists')
    #     auth=cookies['strava_auth']
    #     st.write('Welcome '+cookies['strava_auth']['athlete']['firstname'])

    #     # if st.button("Log Out"):
    #     #     cookie_manager.delete('strava_auth_code',key='0')
    #     #     cookie_manager.delete('strava_auth',key='1')

    #     #     st.write("You are logged out!")
    #     df = load_strava_data(auth)

    #     st.write('You have a total of '+str(df.index.size)+' activities in your Strava account!')

    #     st.write(df.head(5))




    # except:
    #     st.write('Cookies aren\'t setup')
    #     payload={"client_id": STRAVA_CLIENT_ID,"redirect_uri": APP_URL,"response_type": "code","approval_prompt": "auto","scope": "activity:read_all"}
    #     r=requests.post('https://www.strava.com/oauth/authorize',params=payload)
    #     st.markdown("<a href=\""+str(r.url)+"\" target = \"_self\">Click here to login to Strava!</a>",
    #     unsafe_allow_html=True,)

    #     query_params = st.experimental_get_query_params()
    #     st.write(query_params)

    #     if query_params.get("code"):

    #         authorization_code = query_params.get("code", [None])[0]

    #         payload = {
    #         'client_id': STRAVA_CLIENT_ID,
    #         'client_secret': STRAVA_CLIENT_SECRET,
    #         'code': authorization_code,
    #         "grant_type": "authorization_code"
    #         }
    #         r=requests.post('https://www.strava.com/oauth/token',params=payload)
    #         auth=r.json()
    #         st.write('this is auth')
    #         st.write(auth)
    #         if query_params:
    #             cookie_manager.set('strava_auth_code', authorization_code, key='0')
    #             cookie_manager.set('strava_auth', auth, key='1')





    # try:
    #     if cookies['strava_auth_code'] != 'null':
    #         st.write('Cookies are set!')
            
    #         auth=cookies['strava_auth']
    #         st.write('Welcome '+cookies['strava_auth']['athlete']['firstname'])

    #         if st.button("Log Out"):
    #             cookie_manager.delete('strava_auth_code',key='0')
    #             cookie_manager.delete('strava_auth',key='1')

    #             st.write("You are logged out!")
    #         df = load_strava_data(auth)

    #         st.write('You have a total of '+str(df.index.size)+' activities in your Strava account!')

    #         st.write(df.head(5))

    #         d=df.groupby([df['date'].values.astype('datetime64[M]'),df['type']]).size().to_frame('activities').reset_index().copy()
    #         d.columns = ['month','type','workouts']

    #         fig=px.bar(d,
    #             x='month',
    #             y='workouts',
    #             color='type',
    #             title='Total Number of Activities by Month',
    #             )
    #         fig.update_traces(
    #             hovertemplate='%{x} - %{y} activities'
    #         )
    #         fig.update_yaxes(title='# of Activities',
    #                         showgrid=False,
    #                         )
    #         fig.update_layout(plot_bgcolor='white')
    #         fig.update_xaxes(title='Date',
    #                         showgrid=False,
    #                         )
    #         st.plotly_chart(fig)
    #         del d



    #         fig=px.scatter(df.loc[df.distance>0],
    #                     x='start_date',
    #                     y='distance_miles',
    #                     color='type',
    #                     title='Workout Distance by Workout Type',
    #                     hover_data=['type','name','date','id'])
    #         fig.update_traces(mode='markers',
    #                             marker=dict(size=8,
    #                                         line=dict(width=1,
    #                                                 color='DarkSlateGrey')))
    #         fig.update_xaxes(title='Date',showgrid=False)
    #         fig.update_yaxes(title='Workout Distance (Miles)',showgrid=False)
    #         fig.update_layout(plot_bgcolor='white')
    #         fig.update_traces(
    #             hovertemplate='%{x} - %{customdata[0]}<br>%{customdata[1]}<br>%{y:.1f} miles<extra></extra>'
    #         )
    #         st.plotly_chart(fig)



    #         st.markdown('### Build Your Own Scatter Plot')
    #         columns=df.columns

    #         numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #         num_columns = df.select_dtypes(include=numerics).columns.tolist()

    #         workout_options = df['type'].unique()
    #         selected_workouts = st.multiselect('Select Workouts', workout_options, workout_options)
    #         df = df[df['type'].isin(selected_workouts)]

    #         x_axis = st.selectbox('Select values on the X Axis',
    #             columns,
    #             58)

    #         y_axis = st.selectbox('Select values on the Y Axis',
    #             columns,
    #             36)

    #         hover_data = st.multiselect('Select values on that will appear on hover',
    #             columns)

    #         print('X Axis: ' + x_axis)
    #         print('Y Axis: ' + y_axis)

    #         fig=px.scatter(df.loc[df.distance>0],
    #                     x=x_axis,
    #                     y=y_axis,
    #                     color='type',
    #                     title='Build Your Own Scatterplot',
    #                     hover_data=hover_data)
    #         fig.update_traces(mode='markers',
    #                             marker=dict(size=8,
    #                                         line=dict(width=1,
    #                                                 color='DarkSlateGrey')))
    #         fig.update_xaxes(showgrid=False)
    #         fig.update_yaxes(showgrid=False)
    #         fig.update_layout(plot_bgcolor='white')
    #         # fig.update_traces(
    #         #     hovertemplate='%{x} - %{customdata[0]}<br>%{customdata[1]}<br>%{y:.1f} miles<extra></extra>'
    #         # )
    #         st.plotly_chart(fig)


    #         st.write(num_columns)
    #         st.write(columns)

    #         # distance, moving_time, elapsed_time, total_elevation_gain, start_date, achievement_count, kudos_count, average_speed, max_speed, average_cadence, average_temp, average_heartrate, max_heartrate, elev_high, elev_low, pr_count, average_watts, kilojoules, distance_miles, elapsed_time_hours
    # except:
    #     # pass

    # # else:

    #     st.write("You are not logged in!")


    #     go_button = st.button('Go')

    #     if go_button:

    #     # if query_params.get("code"):

    #         authorization_code = query_params.get("code", [None])[0]

    #         payload = {
    #         'client_id': STRAVA_CLIENT_ID,
    #         'client_secret': STRAVA_CLIENT_SECRET,
    #         'code': authorization_code,
    #         "grant_type": "authorization_code"
    #         }

    #         r=requests.post('https://www.strava.com/oauth/token',params=payload)

    #         auth=r.json()
    #         st.write('this is auth')
    #         st.write(auth)
    #         if query_params:
    #             st.write(query_params)
    #             st.write(auth)


    #             cookie_manager.set('strava_auth_code', authorization_code, key='0')
    #             cookie_manager.set('strava_auth', auth, key='1')

    #             # st.write(cookies)











    # query_params = st.experimental_get_query_params()
    # if query_params.get("code", [None])[0]:
    #     auth = get_strava_auth(query_params)
    #     st.write('Welcome '+auth['athlete']['firstname'])

    #     df = load_strava_data(auth)

    #     st.write('You have a total of '+str(df.index.size)+' activities in your Strava account!')

    #     st.write(df.head(5))

    #     d=df.groupby([df['date'].values.astype('datetime64[M]'),df['type']]).size().to_frame('activities').reset_index().copy()
    #     d.columns = ['month','type','workouts']

    #     fig=px.bar(d,
    #         x='month',
    #         y='workouts',
    #         color='type',
    #         title='Total Number of Activities by Month',
    #         )
    #     fig.update_traces(
    #         hovertemplate='%{x} - %{y} activities'
    #     )
    #     fig.update_yaxes(title='# of Activities',
    #                    showgrid=False,
    #                   )
    #     fig.update_layout(plot_bgcolor='white')
    #     fig.update_xaxes(title='Date',
    #                   showgrid=False,
    #                   )
    #     st.plotly_chart(fig)
    #     del d



    #     fig=px.scatter(df.loc[df.distance>0],
    #               x='start_date',
    #               y='distance_miles',
    #               color='type',
    #               title='Workout Distance by Workout Type',
    #               hover_data=['type','name','date','id'])
    #     fig.update_traces(mode='markers',
    #                       marker=dict(size=8,
    #                                   line=dict(width=1,
    #                                             color='DarkSlateGrey')))
    #     fig.update_xaxes(title='Date',showgrid=False)
    #     fig.update_yaxes(title='Workout Distance (Miles)',showgrid=False)
    #     fig.update_layout(plot_bgcolor='white')
    #     fig.update_traces(
    #         hovertemplate='%{x} - %{customdata[0]}<br>%{customdata[1]}<br>%{y:.1f} miles<extra></extra>'
    #     )
    #     st.plotly_chart(fig)



    #     st.markdown('### Build Your Own Scatter Plot')
    #     columns=df.columns

    #     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #     num_columns = df.select_dtypes(include=numerics).columns.tolist()

    #     workout_options = df['type'].unique()
    #     selected_workouts = st.multiselect('Select Workouts', workout_options, workout_options)
    #     df = df[df['type'].isin(selected_workouts)]

    #     x_axis = st.selectbox('Select values on the X Axis',
    #         columns,
    #         58)

    #     y_axis = st.selectbox('Select values on the Y Axis',
    #         columns,
    #         36)

    #     hover_data = st.multiselect('Select values on that will appear on hover',
    #         columns)

    #     print('X Axis: ' + x_axis)
    #     print('Y Axis: ' + y_axis)

    #     fig=px.scatter(df.loc[df.distance>0],
    #               x=x_axis,
    #               y=y_axis,
    #               color='type',
    #               title='Build Your Own Scatterplot',
    #               hover_data=hover_data)
    #     fig.update_traces(mode='markers',
    #                       marker=dict(size=8,
    #                                   line=dict(width=1,
    #                                             color='DarkSlateGrey')))
    #     fig.update_xaxes(showgrid=False)
    #     fig.update_yaxes(showgrid=False)
    #     fig.update_layout(plot_bgcolor='white')
    #     # fig.update_traces(
    #     #     hovertemplate='%{x} - %{customdata[0]}<br>%{customdata[1]}<br>%{y:.1f} miles<extra></extra>'
    #     # )
    #     st.plotly_chart(fig)


    #     st.write(num_columns)
    #     st.write(columns)

    #     # distance, moving_time, elapsed_time, total_elevation_gain, start_date, achievement_count, kudos_count, average_speed, max_speed, average_cadence, average_temp, average_heartrate, max_heartrate, elev_high, elev_low, pr_count, average_watts, kilojoules, distance_miles, elapsed_time_hours





    # else:

    #     payload={"client_id": STRAVA_CLIENT_ID,"redirect_uri": APP_URL,"response_type": "code","approval_prompt": "auto","scope": "activity:read_all"}
    #     r=requests.post('https://www.strava.com/oauth/authorize',params=payload)
    #     st.markdown("<a href=\""+str(r.url)+"\" target = \"_self\">Click here to login to Strava!</a>",
    #     unsafe_allow_html=True,)


if __name__ == "__main__":
    #execute
    app()

