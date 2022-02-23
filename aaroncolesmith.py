import pandas as pd
import numpy as np
import plotly_express as px
import plotly.io as pio
# pio.templates.default = 'plotly_white'
import streamlit as st
import geocoder
import requests
import about
import experience
import projects
import covid
import clstr
import nbaclusters
import nba_redraftables
import bovada
import nfl_mock_draft
import stock_prediction
import 538_viz

st.set_page_config(
    page_title='aaroncolesmith.com',
    page_icon='dog'
    )

def main():

    PAGES = {
    "About": about,
    "Experience": experience,
    "Projects": projects,
    "Bovada": bovada,
    "COVID-Viz": covid,
    "CLSTR": clstr,
    "FiveThirtyEight": 538_viz,
    "NBA Clusters": nbaclusters,
    "NBA Redraftables": nba_redraftables,
    "NFL Mock Draft DB": nfl_mock_draft,
    "Stock Predictions": stock_prediction
    }

    st.sidebar.title('Navigation')
    sel = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[sel]
    g = geocoder.ip('me')
    try:
        print('Aaronlytics - ' + sel + ' - ' + g.city + ', '+g.state + ' - ' + g.ip + ' - ' + g.hostname)
    except:
        pass
    page.app()


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
