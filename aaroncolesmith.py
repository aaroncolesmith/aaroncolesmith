import pandas as pd
import numpy as np
import plotly_express as px
import streamlit as st
from pandas.io.json import json_normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from pandas import json_normalize
#from bs4 import BeautifulSoup
import requests
#import datetime
#import html
#from urllib.request import Request, urlopen
#from PIL import Image


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("", ('About Me','Work Experience','Projects','Data - Stocks','Data - Coronavirus','Data - NBA Clusters'))

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

    # st.sidebar.title("Data Products")
    # selection = st.sidebar.radio("", ('Stocks','TBD','TBD'))
    #
    # if selection == 'Stocks':
    #     stocks()
    # if selection == 'Work Experience':
    #     experience()
    # if selection == 'Projects':
    #     projects()


def about():
    st.write("""
    # Aaron Cole Smith
    I am a data-driven problem solver who believes that any problem can be solved with hard work, creativity and technology. I have been deployed in a wide range of roles, but have recently excelled as a Product Manager focused on a high-tech product.


    I am currently a Product Manager at [SafeRide Health](https://saferidehealth.com) where we are building the technology platform to enable anyone to get the care they need.

    I'm also very active in building side projects, mainly centered around data-related apps. You can find a few of these on this portfolio, but if these interest you, I'd love to talk further.

    If you have any questions / thoughts, feel free to reach out to me via [email](mailto:aaroncolesmith@gmail.com), [LinkedIn](https://linkedin.com/in/aaroncolesmith) or [Twitter](https://www.twitter.com/aaroncolesmith).

    """)

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

def projects():
    st.write("""
    # Projects
    ### Olive - Process Mining / Process Mapping Product | September 2018 - July 2019
    **Problem:** Process discovery and process documentation were very difficult and time consuming for our team. We would go into meetings with customers and struggle to identify what were some good areas for automation and we would try to document existing processes, but interviews would only uncover a small set of the entire scope of the process.

    **Solution:** We built Pupil which was a Process Discover & Process Mapping product. Pupil would ingest process related data like system logs and using machine learning and process mining algorithms, it would cluster processes together. We could then display those processes in a 3d scatter visualization that would allow users to group similar sets of processes together. Finally, they could output those groups to a process map which could be handed off to the automation team to build an Olive.""")
    st.image('pupil.gif',caption='Pupil was used for Process Mining & Process Discovery',use_column_width=True)

    st.write("""

    ---

    ### Olive - Process Automation Platform | July 2017 - May 2018
    **Problem:** As a company, we realized that we had product market fit, but we needed to build a platform to enable our team to efficiently scale operations of building and deploying Olive.

    **Solution:** We built a process automation platform, speficially for our internal team to build Olives. Built the platform specifically for our team and for our industry so that we could build healthcare automations as quickly, resiliently & easily as possible.""")
    st.image('mimic.gif',caption='The OliveBuilder platform was used to build & deploy thousands of automated workers',use_column_width=True)

    st.write("""
    ---

    ### Olive - Data Analytics Project | July 2016 - February 2017
    **Problem:** Our team at Olive was not harnessing data to maximize our team's operations. We were making decisions based on gut feel or instinct solely as relying on data as a valuable input into the decision-making process.

    **Solution:** As I had a strong data background as well as a strong understanding of our internal data schema, our COO tasked me with harnessing our data to provide intelligence to our staff to operate more efficiently. This was done by connecting multiple datasets together and working with leaders from across the company to ensure everyone was using our data to most efficiently align their team's operations.
    """)
    st.image('grow.png',caption='An example of a data dashboard used to visualize company performance metrics & goals',use_column_width=True)

    st.write("""
    ---

    ### Olive - Connect for Hospital Registration System | January 2016 - July 2016
    **Problem:** We had a legacy product that was built on old technology that was very buggy. It was also built on a single-tenant architecture which meant that costs for supporting the product were growing at an unsupportable rate.

    **Solution:** Went all-in on rewriting and redesigning our existing product so that we had a much more stable future. I led the product requirements and product design while working very closely with engineering leadership to ensure we were meeting project objectives of keeping future support costs to a minimum.
    """)

    st.image('queue.png',caption='Queue was our patient registration system used across 500 health systems', use_column_width=True)

def stocks():
    #d=pd.read_csv('./stocks/stocks_month_chg.csv')
    df=pd.read_csv('./stocks/stocks.csv')
    group=df.groupby(['symbol','name']).agg({'date':'last',
                                      'close':'last',
                                      'close_last_month':'last',
                                      'volume':['last','mean'],
                                      'pct_chg':'last'}).reset_index(drop=False)
    group.columns=['symbol','name','date','close','close_last_month','volume_last','volume_average','pct_chg']
    a=px.scatter(group,
    x='close', y='pct_chg', hover_data = ['symbol','name'],
    title='3 Month Pct Change vs. Most Recent Price',
    width=800, height=600,
    range_y=[-1,2])
    a.update_xaxes(title='Close')
    a.update_yaxes(title='% Change Since Last Month')
    a.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    st.plotly_chart(a)

    l=(df['symbol'] + ' - ' + df['name']).unique()
    l=np.insert(l,0,'')
    selection=st.multiselect('Select stocks -',l)
    #option=st.selectbox('Select a bet -', state.a)
    if len(selection) > 0:
        # st.write(selection)
        f = df.loc[(df['symbol'] + ' - ' + df['name']).isin(selection)]
        # st.write(f)
        line_g=px.line(f,x='date',y='close',color='symbol')
        line_g.update_traces(mode='lines+markers')
        st.plotly_chart(line_g)

def price_tracker():
    df = get_price_data()
    st.write(df)
    fig=px.scatter(df, x='date',y='price',color='item')
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    st.plotly_chart(fig)

def get_price_data():
    agent = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36'}
    df=pd.read_csv('./price_tracker/price_tracker.csv')
    #UNIQLO
    url = 'https://www.uniqlo.com/us/en/men-airism-micro-mesh-tank-top-423528.html'
    company = 'uniqlo'
    req = requests.get(url,headers=agent)
    soup = BeautifulSoup(req.text)
    item = soup.find_all(itemprop="name")[0].text
    price = soup.find_all(itemprop="price")[0].text
    sale_price = ''
    used_price = ''
    date = datetime.datetime.now()

    row = pd.Series([date,url,company,item,price,sale_price,used_price])
    row_df = pd.DataFrame([row])
    row_df.columns = ['date','url','company','item','price','sale_price','used_price']
    df = pd.concat([row_df, df], ignore_index=True, sort=False)

    #CB2
    try:
        url='https://www.cb2.com/justice-oak-coffee-table/s393709'
        company = 'cb2'
        req = requests.get(url,headers=agent)
        soup = BeautifulSoup(req.text)
        item = soup.find_all(class_="shop-bar-product-title")[0].text
        price = soup.find_all(class_="regPrice")[0].text
        try:
            sale_price = soup.find_all(class_="salePrice")[0].text
        except:
            sale_price = ''
        used_price = ''
        date = datetime.datetime.now()

        row = pd.Series([date,url,company,item,price,sale_price,used_price])
        row_df = pd.DataFrame([row])
        row_df.columns = ['date','url','company','item','price','sale_price','used_price']
        df = pd.concat([row_df, df], ignore_index=True, sort=False)
    except:
        pass

    #AMAZON - INSPIRED
    try:
        url = 'https://www.amazon.com/INSPIRED-Create-Tech-Products-Customers/dp/1119387507/ref=tmm_hrd_swatch_0?_encoding=UTF8&qid=&sr='
        company = 'Amazon'
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage)
        item = soup.find_all(id='productTitle')[0].text.replace('\n','').lstrip().rstrip()
        try:
            price = soup.find_all(class_="a-size-medium a-color-price offer-price a-text-normal")[0].text.replace('\n','').lstrip().rstrip()
        except:
            price = soup.find_all(class_="a-size-medium a-color-price")[0].text.replace('\n','').lstrip().rstrip()
        try:
            used_price = soup.find_all(class_="a-color-base offer-price a-text-normal")[0].text
        except:
            used_price = ''
        date = datetime.datetime.now()

        row = pd.Series([date,url,company,item,price,sale_price,used_price])
        row_df = pd.DataFrame([row])
        row_df.columns = ['date','url','company','item','price','sale_price','used_price']

        df = pd.concat([row_df, df], ignore_index=True, sort=False)
    except:
        pass

    #AMAZON - BIKE Tool
    try:
        url = 'https://www.amazon.com/CRANKBROTHERs-Crank-Brothers-Bicycle-19-Function/dp/B002VYB4QC/ref=sr_1_2?dchild=1&keywords=crankbrothers%2Bm19&qid=1588034039&sr=8-2&th=1&psc=1'
        company = 'Amazon'
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage)
        item = soup.find_all(id='productTitle')[0].text.replace('\n','').lstrip().rstrip()
        try:
            price = soup.find_all(class_="a-size-medium a-color-price offer-price a-text-normal")[0].text.replace('\n','').lstrip().rstrip()
        except:
            price = soup.find_all(class_="a-size-medium a-color-price")[0].text.replace('\n','').lstrip().rstrip()
        try:
            used_price = soup.find_all(class_="a-color-base offer-price a-text-normal")[0].text
        except:
            used_price = ''
        date = datetime.datetime.now()

        row = pd.Series([date,url,company,item,price,sale_price,used_price])
        row_df = pd.DataFrame([row])
        row_df.columns = ['date','url','company','item','price','sale_price','used_price']

        df = pd.concat([row_df, df], ignore_index=True, sort=False)
        df.to_csv('./price_tracker.csv',index=False)
    except:
        pass

    return df

def coronavirus():

    url = 'https://covidtracking.com/api/v1/us/daily.json'
    req = requests.get(url)
    df=json_normalize(req.json())
    df['dateChecked'] = pd.to_datetime(df['dateChecked'])
    df['date'] = df.dateChecked.dt.date
    df = df.loc[df.date > df.date.max() - pd.to_timedelta(60, unit='d')]
    #df['day_of_week']=df.dateChecked.dt.weekday_name

    df = df.sort_values('date',ascending=True)
    df['rolling_avg'] = df['positiveIncrease'].rolling(window=7).mean()
    d1 = df[['date','positiveIncrease','rolling_avg']]
    d1 = d1.melt(id_vars=['date']+list(d1.keys()[5:]), var_name='val')
    fig=px.line(d1, x='date', y='value', color='val', title='United States Daily COVID Growth vs 7 Day Rolling Avg')
    fig.update_traces(showlegend=False,
                     mode='lines+markers',
                     marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='# of COVID Cases')
    st.plotly_chart(fig)


    df['rolling_avg_deaths'] = df['deathIncrease'].rolling(window=7).mean()
    d1 = df[['date','deathIncrease','rolling_avg_deaths']]
    d1 = d1.melt(id_vars=['date']+list(d1.keys()[5:]), var_name='val')
    fig=px.line(d1, x='date', y='value', color='val', title='United States Daily COVID Deaths vs 7 Day Rolling Avg')
    fig.update_traces(showlegend=False,
                     mode='lines+markers',
                     marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')))
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='# of COVID Deaths')
    st.plotly_chart(fig)


    url = 'https://covidtracking.com/api/v1/states/daily.json'
    req = requests.get(url)
    df=json_normalize(req.json())
    df['dateChecked'] = pd.to_datetime(df['dateChecked'])
    df['date'] = df.dateChecked.dt.date
    df = df.loc[df.date > df.date.max() - pd.to_timedelta(60, unit='d')]

    a=df['state'].unique()
    a=np.insert(a,0,'')
    option=st.selectbox('Select a State to view data', a)
    if len(option) > 0:
        state=option
        d = df.loc[(df.state == state) & (df.date > df.date.max() - pd.to_timedelta(60, unit='d'))].sort_values('date',ascending=True)
        d['rolling_avg'] = d['positiveIncrease'].rolling(window=7).mean()
        d1 = d[['date','positiveIncrease','rolling_avg']]
        d1.columns = ['date','Daily COVID Cases','7 Day Rolling Avg.']
        d1 = d1.melt(id_vars=['date']+list(d1.keys()[5:]), var_name='val')
        fig=px.line(d1, x='date', y='value', color='val', title='Daily COVID Growth vs 7 Day Rolling Avg for '+state)
        fig.update_traces(showlegend=False,
                         mode='lines+markers',
                         marker=dict(size=6,
                                      line=dict(width=1,
                                                color='DarkSlateGrey')))
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='# of COVID Cases')
        st.plotly_chart(fig)

        d = df.loc[(df.state == state) & (df.date > df.date.max() - pd.to_timedelta(60, unit='d'))].sort_values('date',ascending=True)
        d['rolling_avg'] = d['deathIncrease'].rolling(window=7).mean()
        d1 = d[['date','deathIncrease','rolling_avg']]
        d1.columns = ['date','Daily COVID Deaths','7 Day Rolling Avg.']
        d1 = d1.melt(id_vars=['date']+list(d1.keys()[5:]), var_name='val')
        fig=px.line(d1, x='date', y='value', color='val', title='Daily COVID Deaths vs 7 Day Rolling Avg for '+state)
        fig.update_traces(showlegend=False,
                         mode='lines+markers',
                         marker=dict(size=6,
                                      line=dict(width=1,
                                                color='DarkSlateGrey')))
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='# of COVID Deaths')
        st.plotly_chart(fig)

@st.cache
def load_nba():
    df = pd.read_csv('./nba_year_stats.csv')

    return df

def nba_clusters():
    df = load_nba()
    df=df.loc[df.Year >= 1952]
    df=df.loc[df.MP > 100]
    df=df.loc[df.All_Stat_PM.notnull()]
    year = df.Year.unique()
    #year = np.insert(year,0,np.nan)
    year_min = st.selectbox('Select beginning year - ',year,0)
    #year2 = [row for row in year if row >= year_min]
    year_max = st.selectbox('Select ending year - ',[row for row in year if row >= year_min],len([row for row in year if row >= year_min])-1)

    clusters = st.selectbox('Number of clusters',[2,3,4,5,6,7,8,9,10,11],5)

    df=df.reset_index(drop=True)


    # l=d.columns
    # l=np.insert(l,0,'')
    # selection=st.multiselect('Select fields -',l)
    #st.write(d[l])
    if st.button('Go!'):
        df = df.loc[(df.Year >= year_min) & (df.Year <= year_max)]
        d=df.drop(['Rk','Player','Pos','Age','Tm','Year','Key_Player'],axis=1)
        X=d
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=8)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        test=X_pca

        #Predict K-Means cluster membership
        km_neat = KMeans(n_clusters=clusters, random_state=2).fit_predict(test)
        df['Cluster'] = km_neat
        df['Cluster_x'] = test[:,0]
        df['Cluster_y'] = test[:,1]
        fig = px.scatter(df, x='Cluster_x',y='Cluster_y',color='Cluster',hover_data=['Player','Year','Age','Tm','PPG'])
        fig.update_traces(mode='markers',
                  marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')))
        st.plotly_chart(fig)

        st.write(df[['Year','Player','Pos','Age','Tm','Cluster','PPG','RPG','APG','BPG','SPG','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','All_Stat_PM']].sort_values('All_Stat_PM',ascending=False))

        # fig=px.scatter(df.groupby('Cluster').agg({'PPG':'median','RPG':'median','APG':'median','Player':'size'}).reset_index(),x='PPG',y='RPG',size='Player',color='Cluster',title='PPG vs. RPG by Cluster')
        # fig.update_traces(mode='markers',
        #                   marker=dict(line=dict(width=1,
        #                                         color='DarkSlateGrey')))
        # st.plotly_chart(fig)
        #
        #
        #
        # fig=px.scatter(df.groupby('Cluster').agg({'PPG':'median','RPG':'median','APG':'median','SPG':'median','Player':'size'}).reset_index(),x='SPG',y='APG',size='Player',color='Cluster',title='SPG vs. APG by Cluster')
        # fig.update_traces(mode='markers',
        #                   marker=dict(line=dict(width=1,
        #                                         color='DarkSlateGrey')))
        # st.plotly_chart(fig)

        fig = px.scatter_3d(df.groupby('Cluster').agg({'PPG':'median','RPG':'median','APG':'median','SPG':'median','Player':'size'}).reset_index(),
                            x='PPG',y='RPG',z='APG',color='Cluster')
        fig.update_traces(marker=dict(line=dict(width=1,color='DarkSlateGrey')))
        st.plotly_chart(fig)

        fig = px.scatter_3d(df, x='PPG', y='RPG', z='APG',color='Cluster')
        st.plotly_chart(fig)

        # fields = df.columns
        # fields = np.insert(fields,0,'')
        # x = st.selectbox('Select a field for x-axis',fields)
        # y = st.selectbox('Select a field for y-axis',fields)
        #
        # if len(x) > 0:
        #     if len(y) > 0:
        #         fig = px.scatter(df, x=x, y=y)
        #         fig.update_traces(mode='markers',
        #           marker=dict(size=8,
        #                       line=dict(width=1,
        #                                 color='DarkSlateGrey')))
        #         st.plotly_chart(fig)

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
