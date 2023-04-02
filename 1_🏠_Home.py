import pandas as pd
import numpy as np
import plotly_express as px
import plotly.io as pio
pio.templates.default = 'plotly_white'
import streamlit as st
import requests


def main():
    st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog'
    )

    print('AARONLOG - Main load ')

    # PAGES = {
    # "About": about,
    # "Experience": experience,
    # "Projects": projects,
    # "Bovada": bovada,
    # # "COVID-Viz": covid,
    # "CLSTR": clstr,
    # "FiveThirtyEight": fivethirtyeight_viz,
    # "NBA Clusters": nbaclusters,
    # "NBA Redraftables": nba_redraftables,
    # "NFL Mock Draft DB": nfl_mock_draft,
    # "Portland Crime Map": portland_crime_map,
    # "Stock Predictions": stock_prediction,
    # # "Strava Viz": strava_viz
    # }

    # st.sidebar.title('Navigation')
    # sel = st.sidebar.radio("Go to", list(PAGES.keys()))
    # page = PAGES[sel]

    # pages = list(PAGES.keys())
    # query_params = st.experimental_get_query_params()
    # print(query_params)
    # try:
    #     query_option = query_params['page'][0]
    # except:
    #     st.experimental_set_query_params(page=pages[0])
    #     query_params = st.experimental_get_query_params()
    #     query_option = query_params['page'][0]
    # page_selected = st.sidebar.selectbox('Pick option',
    #                                         pages,
    #                                         index=pages.index(query_option))
    # if page_selected:
    #     print('AARONLOG - page selected ' + page_selected)
    #     st.experimental_set_query_params(page=page_selected)
    #     PAGES[page_selected].app()

    st.write("""
    # Aaron Cole Smith
    Hi, I'm Aaron. I am a data-driven problem solver who believes that most any problem can be solved with creativity, technology, and hard work. I'm a passionate product manager with an emphasis on using data to inform my decision making and to make innovative products and features.

    From software that can build and train artificial intelligence solutions integrated with OCR and machine learning to power Olive's Robotic Process Automation (RPA) platform, to data-driven process discovery and process mining capabilities that power Olive's Pupil software - I'm passionate about building solutions that leverage data to make for better, more engaging customer experiences.
    """)

    st.image('./images/pupil.gif',caption='Pupil - a data-driven process discovery tool -- read more in the Projects section',use_column_width=True)

    st.write("""
    This passion also spills over into many of my side projects, mainly centered around data-driven apps related to topics of interest. Feel free to check out some of these initiatives on the sidebar, or continue below to see a quick preview of each.
    """)

    with st.expander("Bovada"):
        st.write("""
        Visualize betting odds scraped over time

        A very interesting set of data to see how implied probability of an event changes over time. Take a look at the 2020 Presidential Election, it is a pretty interesting visualization.
        """)
        st.image('./images/bovada.gif',use_column_width=True)

    # with st.expander("COVID-Viz"):
    #     st.write("""
    #     Visualize COVID cases & deaths over time, broken down by country

    #     My goal was to not only allow someone to view COVID cases and deaths over time, but also introduce some metrics that would give an indication whether a given country was on the rise or falling.
    #     Note: this one utlizes lots of data, so it may take some time to load
    #     """)
    #     st.image('./images/covid-viz.gif',use_column_width=True)

    with st.expander("CLSTR"):
        st.write("""
        Automatically cluster datasets -- choose from a couple presets or load a CSV to load your own data

        I've been very interested in unsupervised learning where you can just throw an algorithm a set of data and see the results. That was the intention with CLSTR which will allow you to upload your own data and see the results. This could be used for customer segmentation, recommendation engines, among many other applications.
        """)
        st.image('./images/clstr.gif',use_column_width=True)

    with st.expander("NBA Clusters"):
        st.write("""
        See how different NBA players/careers cluster based on similar stats

        A more specific clustering example, I wanted to see if I could take a data set with a number of different features and see the results. NBA Reference collects a number of different statistics ranging from scoring to rebounding to defensive metrics. It is very interesting to see how certain plays have similar statistical careers.
        """)
        st.image('./images/nba_clusters.gif',use_column_width=True)

    with st.expander("NBA Redraftables"):
        st.write("""
        How would NBA teams redraft based on a player's statistical career? Based on [The Ringer's Podcast Series](https://www.theringer.com/nba/2020/4/1/21202663/the-ringer-nba-redraftables-series)

        By combining draft results with a players career statistics, you can visualize whether a given draft pick was a good choice or a poor choice.
        """)
        st.image('./images/nba-redraftables.gif',use_column_width=True)

    with st.expander("Portland Crime Map"):
        st.write("""
        A map of Portland's crime data, broken down by neighborhood and visualized over time
        """)

    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.write("""If you have any questions / thoughts, feel free to reach out to me via [email](mailto:aaroncolesmith@gmail.com), [LinkedIn](https://linkedin.com/in/aaroncolesmith) or [Twitter](https://www.twitter.com/aaroncolesmith).""")

    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=portfolio&ea=about">',unsafe_allow_html=True)


def hide_footer():
    hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)

if __name__ == "__main__":
    #execute

    main()
    hide_footer()
