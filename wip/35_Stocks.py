import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from utils.utils import quick_clstr_util


st.set_page_config(
        page_title='aaroncolesmith.com',
        page_icon='dog',
        layout='wide'
    )


# --- Load S&P 500 tickers from Wikipedia ---
@st.cache_data
def load_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table['Symbol'].tolist()

# --- Load historical data ---
@st.cache_data
def load_price_data(tickers, period="6mo"):
    df = yf.download(tickers, period=period, progress=False)['Close']
    return df

# --- Streamlit UI ---
st.title("📈 Stock Clustering Based on % Price Change")

with st.sidebar:
    st.header("Configuration")
    
    # Ticker list
    tickers = load_sp500_tickers()

    tickers_to_add = ['TECL','VOO','SPXL','TQQQ','UPRO','SOXL','FNGU','CWEB','ARKK','ARKG','ARKQ','ARKW','SPY','SCHX','QQQ']
    tickers += tickers_to_add
    
    selected_tickers = st.multiselect("Select Stocks", tickers, default=tickers[:25])
    
    # Custom percent change windows
    change_windows = st.text_input("Percent Change Windows (days, comma-separated)", "1,7,30,90")
    window_list = [int(w.strip()) for w in change_windows.split(",") if w.strip().isdigit()]
    
    # Clustering
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)

# --- Load and preprocess data ---
with st.spinner("Fetching stock data..."):
    price_data = load_price_data(selected_tickers)
    pct_change = price_data.pct_change().dropna()

# --- Create features ---
summary = pd.DataFrame(index=pct_change.columns)

window_field_list = []
for window in window_list:
    label = f"{window}D Change"
    summary[label] = price_data.pct_change(window).iloc[-1]
    window_field_list.append(label)

st.write(summary)


# --- Clustering ---
X = summary.dropna().values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)
summary = summary.loc[summary.dropna().index]  # align with scaler input
summary["Cluster"] = clusters

# --- PCA for visualization ---
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
summary["PC1"] = components[:, 0]
summary["PC2"] = components[:, 1]

# --- Plot ---
fig = px.scatter(
    summary,
    x="PC1", y="PC2",
    color=summary["Cluster"].astype(str),
    hover_name=summary.index,
    title="Clusters of Stocks Based on Price Changes",
)







st.plotly_chart(fig, use_container_width=True)






fig=px.scatter(summary,
                x='PC1',
                y='PC2',
                width=800,
                height=800,
                color=summary["Cluster"].astype(str),
                hover_name=summary.index,
                title="Clusters of Stocks Based on Price Changes",
                template='simple_white' # Added template for consistency with previous snippets
                )


fig.update_layout(
    # legend_title_text=color,
        font_family='Futura',
        height=800,
        font_color='black',
                    )

fig.update_traces(mode='markers',
                    opacity=.75,
                    marker=dict(size=16,line=dict(width=2,color='DarkSlateGrey'))
                    )

st.plotly_chart(fig, use_container_width=True)




summary = summary.reset_index(drop=False)

quick_clstr_util(summary,window_field_list, ['Ticker'], 'Ticker', player=None, player_list=None)




# --- Show table ---
st.subheader("📊 Cluster Table")
st.dataframe(summary.sort_values("Cluster"))