import pandas as pd
import numpy as np
import plotly_express as px
import streamlit as st
from pandas.io.json import json_normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# try:
#     from coviz.coronavirus_viz import coronavirus_viz
# except:
#     from cviz.coronavirus_viz import coronavirus_viz
#from pandas import json_normalize
#from bs4 import BeautifulSoup
import requests
#import datetime
#import html
#from urllib.request import Request, urlopen
#from PIL import Image


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("",
    ('About Me',
    'Work Experience',
    'Projects'
    ))

    if selection == 'About Me':
        about()
    if selection == 'Work Experience':
        experience()
    if selection == 'Projects':
        projects()
    if selection == 'Data - Stocks':
        stocks()
    if selection == 'Price Tracker':
        price_tracker()
    if selection == 'Data - Coronavirus':
        coronavirus()
    if selection == 'Data - NBA Clusters':
        nba_clusters()

    st.sidebar.markdown('---')
    st.sidebar.markdown('# Data Products')
    st.sidebar.markdown("""
    * [Coronavirus Viz](http://coronavirus.aaroncolesmith.com)
    * [NBA Clusters](http://nbaclusters.aaroncolesmith.com)
    * [Bovada](http://bovada.aaroncolesmith.com)
    """)
    st.sidebar.image('./images/oscar_sticker.png',width=75)



def about():

    st.write("""
    # Aaron Cole Smith
    Hi, I'm Aaron. I am a data-driven problem solver who believes that any problem can be solved with hard work, creativity and technology. I have been deployed in a wide range of roles, but have recently excelled as a Product Manager focused on a high-tech product.


    I am currently a Product Manager at [SafeRide Health](https://saferidehealth.com) where we are building the technology platform to enable anyone to get the care they need.

    I'm also very active in building side projects, mainly centered around data-related apps. Feel free to check some of these out on the sidebar, like NBA Clusters where you can see how different NBA players / careers cluster based on similar stats.""")

    st.image('./images/nba_clusters.gif',caption='Preview of NBA Clusters -- feel free to check it out!',use_column_width=True)

    st.write("""If you have any questions / thoughts, feel free to reach out to me via [email](mailto:aaroncolesmith@gmail.com), [LinkedIn](https://linkedin.com/in/aaroncolesmith) or [Twitter](https://www.twitter.com/aaroncolesmith).

    """)
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=portfolio&ea=about">',unsafe_allow_html=True)

    st.markdown(
            """
    <style>
    canvas {
    max-width: 100%!important;
    height: auto!important;
    </style>
    """,
            unsafe_allow_html=True
        )

def experience():
    st.write("""
    # Work Experience

    ### [SafeRide Health](https://saferidehealth.com)
    #### Product Manager | Present
    - Built the technology platform to enable end-to-end patient transportation to care, supporting a number of different
    use cases supporting Health Plans, Health Networks, NEMT Providers & ridesharing solutions
    - Automated existing Operations’ processes to enable the organization to scale without having to increase team size
    - Introduced software development processes to enable a much more collaborative product / engineering culture

    ---
    ### [Olive](https://oliveai.com)
    #### Product Manager - AI & Process Automation Platform | 2017 - 2020
    - Built the software platform used to build and train our artificial intelligence solution, Olive, which allows an
    automation engineer to utilize a wide range RPA functionality and integrate with state of the art technology like CV,
    OCR and machine intelligence
    - Developed from the ground up a state of the art process mining / intelligence product to greatly reduce the time to
    analyze and document processes prior to building out automations
    - Increased automation build velocity by 500% by eliciting user feedback to implement usability enhancements

    #### Product Manager - Data & Analytics | 2016 - 2018
    - Built a data analytics solution to analyze our data across multiple data sources to provide the entire company with
    actionable intelligence and insights to influence product and business strategy
    - Shifted the company focus to make more data-driven decisions using the metrics and tools recently created

    #### Product Manager - Patient Registration & Internal Tools | 2015 - 2016
    - Developed the next-gen platform for our flagship product with a mindset of increasing scalability and engagement
    - Collaborated cross-functionally with leadership to ensure the internal tools developed would support all necessary
    job functions to allow CrossChx to grow at the rate it needed to meet investors’ demands

    ---
    ### [Cardinal Solutions Group](http://www.cardinalsolutions.com/)
    #### Senior Consultant – Product Management | 2014 - 2015
    - Hired as a Senior Consultant to the Agile Project Services helping mentor newer employees in both Agile/Scrum methodologies as well as working as a Business Analyst and Product Owner
    - Developed the requirements and the content model for key components to enable a new version of a product to be marketed and sold as an innovative educational solution
    - Collaborated with product stakeholders to build a comprehensive product backlog and roadmap to enable to the development of critical enhancements and cutting-edge features to enable approximately $1 billion in sales
    - Managed the workload for the development team to create incremental product updates towards the roadmap

    ---
    ### [Accenture](https://www.accenture.com/us-en)
    #### Consultant / Demand Planning Team Lead | 2011 - 2014
    - Analyzed, designed, and tested new functionality for DLA’s ERP solution which included enhancements to SAP (ECC 6.0) and JDA/Manugistics solutions as well as interfaces to Business Intelligence and Reporting systems
    - Managed the Demand Planning team to ensure all resources had a sufficient amount of work assigned
    - As the JDA Demand Planning subject matter expert, provided guidance on all new features being built to assess
    the potential impact to downstream data interfaces
    - Worked with key client contacts to maintain our current relationship while also working with our delivery leads to
    explore and expand our business relationship
    """)
    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=portfolio&ea=experience">',unsafe_allow_html=True)

def projects():
    st.write("""
    # Projects
    ### Olive - Process Mining / Process Mapping Product | September 2018 - July 2019
    **Problem:** Process discovery and process documentation were very difficult and time consuming for our team. We would go into meetings with customers and struggle to identify what were some good areas for automation and we would try to document existing processes, but interviews would only uncover a small set of the entire scope of the process.

    **Solution:** We built Pupil which was a Process Discover & Process Mapping product. Pupil would ingest process related data like system logs and using machine learning and process mining algorithms, it would cluster processes together. We could then display those processes in a 3d scatter visualization that would allow users to group similar sets of processes together. Finally, they could output those groups to a process map which could be handed off to the automation team to build an Olive.""")
    st.image('./images/pupil.gif',caption='Pupil was used for Process Mining & Process Discovery',use_column_width=True)

    st.write("""

    ---

    ### Olive - Process Automation Platform | July 2017 - May 2018
    **Problem:** As a company, we realized that we had product market fit, but we needed to build a platform to enable our team to efficiently scale operations of building and deploying Olive.

    **Solution:** We built a process automation platform, speficially for our internal team to build Olives. Built the platform specifically for our team and for our industry so that we could build healthcare automations as quickly, resiliently & easily as possible.""")
    st.image('./images/mimic.gif',caption='The OliveBuilder platform was used to build & deploy thousands of automated workers',use_column_width=True)

    st.write("""
    ---

    ### Olive - Data Analytics Project | July 2016 - February 2017
    **Problem:** Our team at Olive was not harnessing data to maximize our team's operations. We were making decisions based on gut feel or instinct solely as relying on data as a valuable input into the decision-making process.

    **Solution:** As I had a strong data background as well as a strong understanding of our internal data schema, our COO tasked me with harnessing our data to provide intelligence to our staff to operate more efficiently. This was done by connecting multiple datasets together and working with leaders from across the company to ensure everyone was using our data to most efficiently align their team's operations.
    """)
    st.image('./images/grow.png',caption='An example of a data dashboard used to visualize company performance metrics & goals',use_column_width=True)

    st.write("""
    ---

    ### Olive - Connect - Hospital Registration System | January 2016 - July 2016
    **Problem:** We had a legacy product that was built on old technology that was very buggy. It was also built on a single-tenant architecture which meant that costs for supporting the product were growing at an unsupportable rate.

    **Solution:** Went all-in on rewriting and redesigning our existing product so that we had a much more stable future. I led the product requirements and product design while working very closely with engineering leadership to ensure we were meeting project objectives of keeping future support costs to a minimum.
    """)

    st.image('./images/queue.png',caption='Queue was our patient registration system used across 500 health systems', use_column_width=True)

    st.write('<img src="https://www.google-analytics.com/collect?v=1&tid=UA-18433914-1&cid=555&aip=1&t=event&ec=portfolio&ea=projects">',unsafe_allow_html=True)


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
