import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from scipy import stats
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Sales Anomaly Detection", layout="wide", initial_sidebar_state="expanded")

# Authentication (basic password protection)
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets.get("password", "admin123"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.info("ℹ️ Default password is: admin123 (change in .streamlit/secrets.toml)")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()

# Title
st.title("📊 Sales Anomaly Detection Dashboard")
st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader("📁 Upload your sales data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])

# Load data function
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Convert date column to datetime
        if 'DataReal' in df.columns:
            df['DataReal'] = pd.to_datetime(df['DataReal'])
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Generate sample data for demo
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    products = ['Skincare', 'Haircare', 'Makeup', 'Fragrance', 'Supplements']
    actions = ['Purchase', 'Return', 'Exchange']
    
    data = []
    for date in dates:
        for _ in range(np.random.randint(50, 150)):
            data.append({
                'Zedd': np.random.randint(1000, 9999),
                'AdjTanxa': round(np.random.uniform(10, 500), 2),
                'UN': np.random.randint(1, 10),
                'DataReal': date,
                'ProdG': np.random.choice(products, p=[0.3, 0.2, 0.25, 0.15, 0.1]),
                'IdProd': np.random.randint(100, 999),
                'Actions': np.random.choice(actions, p=[0.85, 0.1, 0.05]),
                'Weekday': date.day_name(),
                'Month': date.month,
                'Year': date.year
            })
    
    df = pd.DataFrame(data)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[anomaly_indices, 'AdjTanxa'] = df.loc[anomaly_indices, 'AdjTanxa'] * np.random.uniform(3, 8, len(anomaly_indices))
    
    return df

# Load data based on upload or use sample
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    st.success(f"✅ File uploaded successfully! Loaded {len(df)} rows.")
else:
    st.info("ℹ️ No file uploaded. Using sample data for demonstration. Upload your CSV/Excel file to analyze your data.")
    if st.button("📥 Download Sample Data Template"):
        sample_df = generate_sample_data()
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=csv,
            file_name="sample_sales_data.csv",
            mime="text/csv"
        )
    df = generate_sample_data()

# Validate required columns
required_columns = ['DataReal', 'AdjTanxa']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
    st.info("Required columns: DataReal (date), AdjTanxa (sales amount)")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range filter
min_date = df['DataReal'].min().date()
max_date = df['DataReal'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Product Group filter (if exists)
if 'ProdG' in df.columns:
    prodg_options = ['All'] + sorted(df['ProdG'].unique().tolist())
    selected_prodg = st.sidebar.multiselect(
        "Product Group (ProdG)",
        options=prodg_options,
        default=['All']
    )
else:
    selected_prodg = ['All']

# Product ID filter (if exists)
if 'IdProd' in df.columns:
    idprod_options = ['All'] + sorted(df['IdProd'].unique().tolist())
    selected_idprod = st.sidebar.multiselect(
        "Product ID (IdProd)",
        options=idprod_options,
        default=['All']
    )
else:
    selected_idprod = ['All']

# Actions filter (if exists)
if 'Actions' in df.columns:
    actions_options = ['All'] + sorted(df['Actions'].unique().tolist())
    selected_actions = st.sidebar.multiselect(
        "Actions",
        options=actions_options,
        default=['All']
    )
else:
    selected_actions = ['All']

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Algorithm Settings")

# Algorithm parameters
iso_contamination = st.sidebar.slider("Isolation Forest - Contamination", 0.01, 0.2, 0.05, 0.01)
zscore_threshold = st.sidebar.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.5)
rolling_window = st.sidebar.slider("Rolling Statistics Window (days)", 3, 30, 7, 1)
rolling_std_threshold = st.sidebar.slider("Rolling Std Threshold", 2.0, 4.0, 2.5, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("📈 Visualization Options")

show_markers = st.sidebar.checkbox("Show all data points", value=True)
show_anomaly_markers = st.sidebar.checkbox("Highlight anomalies", value=True)
chart_height = st.sidebar.slider("Chart Height (px)", 300, 800, 450, 50)

# Filter data
filtered_df = df.copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['DataReal'].dt.date >= start_date) & 
                               (filtered_df['DataReal'].dt.date <= end_date)]

if 'ProdG' in df.columns and 'All' not in selected_prodg:
    filtered_df = filtered_df[filtered_df['ProdG'].isin(selected_prodg)]

if 'IdProd' in df.columns and 'All' not in selected_idprod:
    filtered_df = filtered_df[filtered_df['IdProd'].isin(selected_idprod)]

if 'Actions' in df.columns and 'All' not in selected_actions:
    filtered_df = filtered_df[filtered_df['Actions'].isin(selected_actions)]

# Aggregate daily sales
agg_dict = {'AdjTanxa': 'sum'}
if 'UN' in filtered_df.columns:
    agg_dict['UN'] = 'sum'
if 'Zedd' in filtered_df.columns:
    agg_dict['Zedd'] = 'count'

daily_sales = filtered_df.groupby('DataReal').agg(agg_dict).reset_index()

# Rename columns
col_names = ['Date', 'TotalSales']
if 'UN' in filtered_df.columns:
    col_names.append('TotalUnits')
if 'Zedd' in filtered_df.columns:
    col_names.append('InvoiceCount')
daily_sales.columns = col_names

# Check if we have enough data
if len(daily_sales) < 3:
    st.error("⚠️ Not enough data points after filtering. Please adjust your filters.")
    st.stop()

# Display summary metrics
cols = st.columns(4)
with cols[0]:
    st.metric("Total Sales", f"${daily_sales['TotalSales'].sum():,.2f}")
with cols[1]:
    if 'TotalUnits' in daily_sales.columns:
        st.metric("Total Units", f"{daily_sales['TotalUnits'].sum():,.0f}")
    else:
        st.metric("Data Points", f"{len(filtered_df):,}")
with cols[2]:
    st.metric("Avg Daily Sales", f"${daily_sales['TotalSales'].mean():,.2f}")
with cols[3]:
    st.metric("Date Range", f"{len(daily_sales)} days")

st.markdown("---")

# Anomaly Detection Functions
def detect_isolation_forest(data, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    data['anomaly_iso'] = iso_forest.fit_predict(data[['TotalSales']].values)
    data['anomaly_iso'] = data['anomaly_iso'].map({1: 0, -1: 1})
    data['anomaly_score_iso'] = iso_forest.score_samples(data[['TotalSales']].values)
    return data

def detect_zscore(data, threshold=3.0):
    data['zscore'] = np.abs(stats.zscore(data['TotalSales']))
    data['anomaly_zscore'] = (data['zscore'] > threshold).astype(int)
    return data

def detect_rolling_stats(data, window=7, threshold=2.5):
    data['rolling_mean'] = data['TotalSales'].rolling(window=window, center=True).mean()
    data['rolling_std'] = data['TotalSales'].rolling(window=window, center=True).std()
    data['rolling_lower'] = data['rolling_mean'] - threshold * data['rolling_std']
    data['rolling_upper'] = data['rolling_mean'] + threshold * data['rolling_std']
    data['anomaly_rolling'] = ((data['TotalSales'] < data['rolling_lower']) | 
                                (data['TotalSales'] > data['rolling_upper'])).astype(int)
    return data

# Apply anomaly detection
daily_sales = detect_isolation_forest(daily_sales.copy(), contamination=iso_contamination)
daily_sales = detect_zscore(daily_sales, threshold=zscore_threshold)
daily_sales = detect_rolling_stats(daily_sales, window=rolling_window, threshold=rolling_std_threshold)

# Create visualizations
def create_anomaly_plot(data, anomaly_col, title, show_bands=False):
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['TotalSales'],
        mode='lines+markers' if show_markers else 'lines',
        name='Daily Sales',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Rolling bands for rolling stats method
    if show_bands and 'rolling_mean' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['rolling_mean'],
            mode='lines',
            name='Rolling Mean',
            line=dict(color='green', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['rolling_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['rolling_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=False
        ))
    
    # Anomalies
    if show_anomaly_markers:
        anomalies = data[data[anomaly_col] == 1]
        fig.add_trace(go.Scatter(
            x=anomalies['Date'],
            y=anomalies['TotalSales'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=12, symbol='x', line=dict(width=2))
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Total Sales ($)",
        hovermode='x unified',
        height=chart_height,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

# Display charts
st.subheader("🔴 Isolation Forest Detection")
anomaly_count_iso = daily_sales['anomaly_iso'].sum()
st.info(f"Detected {anomaly_count_iso} anomalies ({anomaly_count_iso/len(daily_sales)*100:.1f}% of days)")
fig1 = create_anomaly_plot(daily_sales, 'anomaly_iso', 
                           f"Isolation Forest (Contamination: {iso_contamination})")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

st.subheader("🔵 Z-Score Detection")
anomaly_count_z = daily_sales['anomaly_zscore'].sum()
st.info(f"Detected {anomaly_count_z} anomalies ({anomaly_count_z/len(daily_sales)*100:.1f}% of days)")
fig2 = create_anomaly_plot(daily_sales, 'anomaly_zscore', 
                           f"Z-Score Method (Threshold: {zscore_threshold}σ)")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.subheader("🟢 Rolling Statistics Detection")
anomaly_count_roll = daily_sales['anomaly_rolling'].sum()
st.info(f"Detected {anomaly_count_roll} anomalies ({anomaly_count_roll/len(daily_sales)*100:.1f}% of days)")
fig3 = create_anomaly_plot(daily_sales, 'anomaly_rolling', 
                           f"Rolling Statistics (Window: {rolling_window} days, Threshold: {rolling_std_threshold}σ)",
                           show_bands=True)
st.plotly_chart(fig3, use_container_width=True)

# Anomaly details table
st.markdown("---")
st.subheader("📋 Anomaly Details")

# Combine all anomalies
daily_sales['total_anomalies'] = (daily_sales['anomaly_iso'] + 
                                   daily_sales['anomaly_zscore'] + 
                                   daily_sales['anomaly_rolling'])

anomaly_details = daily_sales[daily_sales['total_anomalies'] > 0].copy()
anomaly_details['Algorithms'] = anomaly_details.apply(
    lambda row: ', '.join([
        'Isolation Forest' if row['anomaly_iso'] else '',
        'Z-Score' if row['anomaly_zscore'] else '',
        'Rolling Stats' if row['anomaly_rolling'] else ''
    ]).strip(', '), axis=1
)

if len(anomaly_details) > 0:
    display_df = anomaly_details[['Date', 'TotalSales', 'total_anomalies', 'Algorithms']].copy()
    display_df.columns = ['Date', 'Total Sales ($)', '# Algorithms', 'Detected By']
    display_df = display_df.sort_values('Date', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No anomalies detected with current settings.")

# Download option
st.markdown("---")
if st.button("📥 Download Anomaly Report"):
    csv = anomaly_details.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"

    )

