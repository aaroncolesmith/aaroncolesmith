import pandas as pd
import numpy as np
import plotly_express as px
import plotly.io as pio
# pio.templates.default = 'plotly_white'
import streamlit as st
import requests
import about
import experience
import projects
# import covid
import clstr
import nbaclusters
import nba_redraftables
import bovada
import nfl_mock_draft
import stock_prediction
import fivethirtyeight_viz
import portland_crime_map
import strava_viz

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog'
    )

def main():

    print('Main load -- test')

    PAGES = {
    "About": about,
    "Experience": experience,
    "Projects": projects,
    "Bovada": bovada,
    # "COVID-Viz": covid,
    "CLSTR": clstr,
    "FiveThirtyEight": fivethirtyeight_viz,
    "NBA Clusters": nbaclusters,
    "NBA Redraftables": nba_redraftables,
    "NFL Mock Draft DB": nfl_mock_draft,
    "Portland Crime Map": portland_crime_map,
    "Stock Predictions": stock_prediction,
    "Strava Viz": strava_viz
    }

    st.sidebar.title('Navigation')
    # sel = st.sidebar.radio("Go to", list(PAGES.keys()))
    # page = PAGES[sel]

    pages = list(PAGES.keys())
    query_params = st.experimental_get_query_params()
    try:
        query_option = query_params['page'][0]
    except:
        st.experimental_set_query_params(page=pages[0])
        query_params = st.experimental_get_query_params()
        query_option = query_params['page'][0]
    page_selected = st.sidebar.selectbox('Pick option',
                                            pages,
                                            index=pages.index(query_option))
    if page_selected:
        st.experimental_set_query_params(page=page_selected)
        PAGES[page_selected].app()


def hide_footer():
    hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)

if __name__ == "__main__":
    #execute
    hide_footer()
    main()
