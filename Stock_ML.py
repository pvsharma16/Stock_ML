import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import plotly.express as px
import io

# Page config
st.set_page_config(page_title="NSE Stock Clustering", layout="wide")
st.title("📊 NSE Stock Clustering Dashboard")

# Load stock metadata
metadata_df = pd.read_csv("nse_fo_sample_template.csv")

# Sidebar controls
st.sidebar.header("Settings")
stock_set = st.sidebar.selectbox("Stock Universe", ["Nifty 50", "F&O Stocks"])
clustering_algo = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])
n_clusters = st.sidebar.slider("Number of Clusters (KMeans/Agglomerative)", min_value=2, max_value=8, value=4)
eps_val = st.sidebar.slider("DBSCAN: Epsilon (Neighborhood Size)", 0.1, 5.0, 1.0, step=0.1)
min_samples_val = st.sidebar.slider("DBSCAN: Min Samples", 2, 10, 3)
date_range = st.sidebar.date_input("Select Date Range", ["2024-01-01", "2025-06-01"])
mode = st.sidebar.selectbox("Cluster by", ["Overall returns", "Day-of-week patterns"])

# Optional filters
selected_sector = st.sidebar.multiselect(
    "Filter by Sector", options=metadata_df['Sector'].unique(), default=metadata_df['Sector'].unique()
)
selected_cap = st.sidebar.multiselect(
    "Filter by Market Cap", options=metadata_df['MarketCap'].unique(), default=metadata_df['MarketCap'].unique()
)

# Apply filters
filtered_df = metadata_df[
    metadata_df['Sector'].isin(selected_sector) & metadata_df['MarketCap'].isin(selected_cap)
]

# Select tickers based on stock set
if stock_set == "Nifty 50":
    tickers = filtered_df['Ticker'].tolist()[:50]
else:
    tickers = filtered_df['Ticker'].tolist()

@st.cache_data
def fetch_data(tickers, start, end):
    raw_data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by='ticker',
        auto_adjust=True,
        threads=True
    )
    adj_close = pd.DataFrame({
        ticker: raw_data[ticker]['Close']
        for ticker in tickers
        if ticker in raw_data and 'Close' in raw_data[ticker]
    })
    return adj_close

# Download data
with st.spinner("Fetching data from Yahoo Finance..."):
    try:
        adj_close = fetch_data(tickers, date_range[0], date_range[1])
    except Exception as e:
        st.error(f"❌ Error while downloading data: {e}")
        st.stop()

if adj_close.empty:
    st.error("No data could be retrieved for the selected date range. Try a broader range or different tickers.")
    st.stop()

returns = adj_close.pct_change()
returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

if mode == "Day-of-week patterns":
    returns_clean['Day'] = returns_clean.index.day_name()
    weekday_avg = returns_clean.groupby('Day').mean().T
    weekday_avg = weekday_avg[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]
    X = weekday_avg
else:
    X = returns_clean.T

st.info(f"✅ {X.shape[0]} out of {len(tickers)} stocks have usable return data.")

if X.shape[0] < 2:
    st.warning("Not enough data for clustering.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Select clustering algorithm
if clustering_algo == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif clustering_algo == "DBSCAN":
    model = DBSCAN(eps=eps_val, min_samples=min_samples_val)
elif clustering_algo == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=n_clusters)
else:
    st.error("Invalid clustering algorithm")
    st.stop()

labels = model.fit_predict(X_scaled)

# Interactive cluster scatter plot using Plotly
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'], index=X.index)
pca_df['Cluster'] = labels.astype(str)
pca_df['Ticker'] = pca_df.index.str.replace('.NS', '', regex=False)

fig = px.scatter(
    pca_df,
    x='PCA1', y='PCA2',
    color='Cluster',
    hover_data=['Ticker'],
    title=f'📊 {clustering_algo} NSE Stock Clusters',
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig, use_container_width=True)

# Cluster-level summary
st.subheader("📈 Cluster Averages: Return, Volatility & Count")
X_df = pd.DataFrame(X_scaled, index=X.index)
X_df['Cluster'] = labels
X_df['Mean Return'] = X_df.index.map(lambda t: returns_clean[t].mean())
X_df['Volatility'] = X_df.index.map(lambda t: returns_clean[t].std())

summary = X_df.groupby('Cluster').agg({
    'Mean Return': 'mean',
    'Volatility': 'mean',
    'Cluster': 'count'
}).rename(columns={'Cluster': 'Count'}).round(4)

# Highlight best cluster (only for numeric cluster labels)
try:
    best_cluster = summary['Mean Return'].idxmax()
    def highlight_best(s):
        is_best = s.name == best_cluster
        return ['background-color: #c6f6d5; font-weight: bold' if is_best else '' for _ in s]
    summary_style = summary.style.apply(highlight_best, axis=1)
    st.dataframe(summary_style)
except:
    st.dataframe(summary)

# Scatter plot of cluster means using Plotly
fig2 = px.scatter(
    summary.reset_index(),
    x='Volatility', y='Mean Return',
    size='Count', color='Cluster',
    text='Cluster',
    title=f'📊 {clustering_algo} Cluster Centers: Mean Return vs. Volatility',
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig2.update_traces(textposition='top center')
fig2.update_layout(xaxis_title='Volatility', yaxis_title='Mean Return')
st.plotly_chart(fig2, use_container_width=True)

# CSV download
csv_buffer = io.StringIO()
X_df.reset_index().rename(columns={'index': 'Ticker'}).to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Clustered Stock Data as CSV",
    data=csv_buffer.getvalue(),
    file_name="nse_clusters.csv",
    mime="text/csv"
)

# Show raw data option
if st.sidebar.checkbox("Show Raw Returns Data"):
    st.subheader("Daily Returns")
    st.dataframe(returns_clean.T.round(4))
