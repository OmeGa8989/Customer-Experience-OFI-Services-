# app.py
"""
Streamlit Customer Experience Dashboard (interactive visualizations)

Improved: Orders trend block now checks multiple possible date columns (order_date, date,
created_at, order_timestamp) and only renders the chart if a valid date column with
parsable values is found. If none are found, a friendly message is shown.

Place this file in the project root (same level as `data/`).
Run: streamlit run app.py
"""
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Plotting
import plotly.express as px
import plotly.graph_objects as go

# ML / embedding
from sklearn.pipeline import Pipeline

# Sentence embeddings (optional)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

st.set_page_config(layout="wide", page_title="Customer Experience Dashboard", initial_sidebar_state="expanded")
st.title("📊 Customer Experience Dashboard — Transformer-enhanced")

# --------------------- Config / Paths ---------------------
DATA_DIR = "data"
ORDERS_PATH = os.path.join(DATA_DIR, "orders.csv")
FEEDBACK_PATH = os.path.join(DATA_DIR, "customer_feedback.csv")
DELIVERY_PATH = os.path.join(DATA_DIR, "delivery_performance.csv")
MODEL_PATH = os.path.join(DATA_DIR, "customer_risk_model.joblib")  # saved by model_training.py

EMBEDDER_NAME = None  # will try to read from model artifact

# --------------------- Utilities ---------------------
@st.cache_data(ttl=600)
def load_csv_safe(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

@st.cache_data(ttl=600)
def load_model(path):
    if os.path.exists(path):
        try:
            artifact = joblib.load(path)
            return artifact
        except Exception:
            return None
    return None

@st.cache_data(ttl=600)
def embed_feedback(feedback_df, embedder_name):
    if feedback_df is None or 'customer_id' not in feedback_df.columns:
        return None
    if 'feedback_text' not in feedback_df.columns:
        if 'text' in feedback_df.columns:
            feedback_df = feedback_df.rename(columns={'text': 'feedback_text'})
        else:
            return None
    grouped = feedback_df.groupby('customer_id').agg({'feedback_text': lambda x: ' '.join(x.dropna().astype(str))}).reset_index()
    if grouped.shape[0] == 0:
        return None
    if SentenceTransformer is None:
        return None
    model = SentenceTransformer(embedder_name)
    embs = model.encode(grouped['feedback_text'].tolist(), show_progress_bar=False)
    emb_df = pd.DataFrame(embs)
    emb_df['customer_id'] = grouped['customer_id'].values
    emb_df = emb_df[['customer_id'] + [c for c in emb_df.columns if c != 'customer_id']]
    emb_df.columns = ['customer_id'] + [f"emb_{i}" for i in range(emb_df.shape[1] - 1)]
    return emb_df

def prepare_features_preview(orders, delivery, feedback_emb=None):
    if orders is None:
        n = 500
        rng = np.random.RandomState(42)
        tab = pd.DataFrame({
            'customer_id': np.arange(1, n + 1),
            'num_orders_90d': rng.poisson(3, n),
            'avg_order_value': rng.uniform(50, 500, n),
            'days_since_last_order': rng.randint(1, 200, n),
        })
    else:
        tab = orders.copy()
        if 'customer_id' in tab.columns:
            if 'order_date' in tab.columns:
                tab['order_date'] = pd.to_datetime(tab['order_date'], errors='coerce')
            agg = tab.groupby('customer_id').agg(
                num_orders_90d=('order_id', 'count') if 'order_id' in tab.columns else ('order_date', 'count'),
                avg_order_value=('order_value', 'mean') if 'order_value' in tab.columns else ('order_id', 'count'),
                days_since_last_order=('order_date', lambda x: (pd.Timestamp.now() - pd.to_datetime(x).max()).days if pd.notna(x).any() else np.nan)
            ).reset_index()
            tab = agg
        else:
            tab = tab.head(500)
            tab['customer_id'] = np.arange(1, len(tab) + 1)
            tab['num_orders_90d'] = 1
            tab['avg_order_value'] = tab.get('order_value', 100)
            tab['days_since_last_order'] = 30

    # delivery merge
    if delivery is not None and 'customer_id' in delivery.columns:
        dp = delivery.groupby('customer_id').agg(
            on_time_rate=('on_time', 'mean') if 'on_time' in delivery.columns else ('delay_hours', lambda x: (x == 0).mean()),
            avg_delay_hours=('delay_hours', 'mean') if 'delay_hours' in delivery.columns else ('on_time', lambda x: 0)
        ).reset_index()
        tab = tab.merge(dp, on='customer_id', how='left')
    else:
        tab['on_time_rate'] = np.nan
        tab['avg_delay_hours'] = np.nan

    tab['recency'] = tab['days_since_last_order'].fillna(999)
    tab['frequency'] = tab['num_orders_90d'].fillna(0)
    tab['monetary'] = tab['avg_order_value'].fillna(tab['avg_order_value'].median() if 'avg_order_value' in tab else 100)

    # merge embeddings
    if feedback_emb is not None:
        tab['customer_id'] = tab['customer_id'].astype(str)
        feedback_emb['customer_id'] = feedback_emb['customer_id'].astype(str)
        df = tab.merge(feedback_emb, on='customer_id', how='left')
    else:
        df = tab.copy()

    return df

def suggest_intervention(row):
    label = row.get('risk_label', None)
    avg_rating = row.get('avg_rating', None)
    recency = row.get('recency', 999)
    negative_feedback = row.get('negative_feedback_count', 0)
    if label == 'High':
        if avg_rating is not None and avg_rating <= 2:
            return "Personal outreach: call from CS + refund/priority support"
        if recency > 90:
            return "Win-back: targeted discount + reminder email"
        if negative_feedback and negative_feedback > 0:
            return "Escalate to retention team + offer credit"
        return "Loyalty offer + curated recommendations"
    elif label == 'Medium':
        if avg_rating is not None and avg_rating <= 3:
            return "Automated survey + coupon for next order"
        return "Email nudge with recommendations"
    else:
        return "Monitor"

# --------------------- Sidebar: controls ---------------------
with st.sidebar:
    st.header("Data & Model")
    st.markdown("The app loads data from the `data/` folder and the trained model artifact saved there.")
    show_raw = st.checkbox("Show raw data preview", value=False)
    refresh = st.button("Refresh data (clear cache)")

# refresh handling
if refresh:
    st.cache_data.clear()
    st.experimental_rerun()

# --------------------- Load data & model ---------------------
orders = load_csv_safe(ORDERS_PATH)
feedback = load_csv_safe(FEEDBACK_PATH)
delivery = load_csv_safe(DELIVERY_PATH)
artifact = load_model(MODEL_PATH)
if artifact is not None:
    pipeline = artifact.get('pipeline', None)
    feature_columns = artifact.get('feature_columns', None)
    EMBEDDER_NAME = artifact.get('embedder_name', EMBEDDER_NAME)
else:
    pipeline = None
    feature_columns = None

# Precompute embeddings if possible
if EMBEDDER_NAME and SentenceTransformer is not None:
    try:
        feedback_emb = embed_feedback(feedback, EMBEDDER_NAME)
    except Exception as e:
        st.warning(f"Embedding failed: {e}")
        feedback_emb = None
else:
    feedback_emb = None

# Prepare features frame used for predictions
features_df = prepare_features_preview(orders, delivery, feedback_emb)
features_df['customer_id'] = features_df['customer_id'].astype(str)

# If pipeline exists, ensure all feature_columns exist in features_df
if pipeline is not None and feature_columns is not None:
    for c in feature_columns:
        if c not in features_df.columns:
            features_df[c] = 0.0

# Predictions
if pipeline is not None and feature_columns is not None:
    X = features_df[feature_columns].fillna(0)
    try:
        probs = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X)
    except Exception:
        probs = np.zeros(X.shape[0])
    features_df['risk_score'] = np.clip(probs, 0.0, 1.0)
    features_df['risk_label'] = features_df['risk_score'].apply(lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low'))
else:
    features_df['risk_score'] = 0.0
    features_df['risk_label'] = 'Unknown'

# Optional: merge ratings if feedback has ratings
if feedback is not None and 'customer_id' in feedback.columns:
    if 'rating' in feedback.columns:
        rating_agg = feedback.groupby('customer_id').agg(
            avg_rating=('rating', 'mean'),
            negative_feedback_count=('rating', lambda x: (x <= 2).sum())
        ).reset_index()
        rating_agg['customer_id'] = rating_agg['customer_id'].astype(str)
        features_df = features_df.merge(rating_agg, on='customer_id', how='left')
    else:
        features_df['avg_rating'] = np.nan
        features_df['negative_feedback_count'] = 0
else:
    features_df['avg_rating'] = np.nan
    features_df['negative_feedback_count'] = 0

# --------------------- Raw data preview (safe) ---------------------
if show_raw:
    st.markdown("---")
    st.subheader("Raw data preview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Orders")
        if orders is None:
            st.info("No orders.csv found in ./data/")
        elif isinstance(orders, pd.DataFrame):
            st.dataframe(orders.head(200))
        else:
            st.warning("orders.csv could not be read as a table")
    with col2:
        st.write("Customer feedback")
        if feedback is None:
            st.info("No customer_feedback.csv found in ./data/")
        elif isinstance(feedback, pd.DataFrame):
            st.dataframe(feedback.head(200))
        else:
            st.warning("customer_feedback.csv could not be read as a table")
    with col3:
        st.write("Delivery performance")
        if delivery is None:
            st.info("No delivery_performance.csv found in ./data/")
        elif isinstance(delivery, pd.DataFrame):
            st.dataframe(delivery.head(200))
        else:
            st.warning("delivery_performance.csv could not be read as a table")

st.markdown("---")

# --------------------- Dashboard layout ---------------------
left_col, center_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.header("Filters")
    risk_filter = st.multiselect("Risk label", options=features_df['risk_label'].unique().tolist(), default=features_df['risk_label'].unique().tolist())
    recency_slider = st.slider("Recency (days since last order)", int(features_df['recency'].min()), int(features_df['recency'].max()), (0, int(features_df['recency'].max())))
    min_monetary = float(features_df['monetary'].min())
    max_monetary = float(features_df['monetary'].max())
    monetary_range = st.slider("Monetary (avg order value)", min_monetary, max_monetary, (min_monetary, max_monetary))
    st.markdown("### Select a customer")
    selected_customer = st.selectbox("Customer ID", options=["-- none --"] + features_df['customer_id'].tolist())

# apply filters
df_viz = features_df[
    (features_df['risk_label'].isin(risk_filter)) &
    (features_df['recency'] >= recency_slider[0]) & (features_df['recency'] <= recency_slider[1]) &
    (features_df['monetary'] >= monetary_range[0]) & (features_df['monetary'] <= monetary_range[1])
].copy()

with center_col:
    st.header("Risk Overview")

    # 1) Risk distribution: interactive bar chart
    risk_counts = df_viz['risk_label'].value_counts().reset_index()
    risk_counts.columns = ['risk_label', 'count']
    fig_bar = px.bar(risk_counts, x='risk_label', y='count', text='count', title="Risk Label Distribution", labels={'count': 'Number of customers', 'risk_label': 'Risk'})
    fig_bar.update_traces(marker_line_width=0.5)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2) Scatter: recency vs frequency colored by risk, sized by monetary
    st.subheader('Recency vs Frequency')
    if df_viz.shape[0] > 0:
        fig_scat = px.scatter(df_viz.sample(min(len(df_viz), 2000)), x='recency', y='frequency', color='risk_label',
                             size='monetary', hover_data=['customer_id', 'monetary', 'recency', 'frequency'],
                             title='Recency vs Frequency (size=monetary)')
        fig_scat.update_layout(xaxis_title='Recency (days)', yaxis_title='Frequency (# orders in 90d)')
        st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.info('No customers match the current filters.')

    # 3) Line chart: orders trend over time (aggregated)
    st.subheader("Orders trend")
    # Try multiple possible date columns for flexibility
    possible_date_cols = ['order_date', 'date', 'created_at', 'order_timestamp']
    date_col_found = None
    orders_df_for_plot = None
    if orders is not None and isinstance(orders, pd.DataFrame):
        for col in possible_date_cols:
            if col in orders.columns:
                temp = orders.copy()
                temp[col] = pd.to_datetime(temp[col], errors='coerce')
                if temp[col].notna().sum() > 0:
                    date_col_found = col
                    orders_df_for_plot = temp
                    break

    if date_col_found is not None and orders_df_for_plot is not None:
        # aggregate monthly (or daily if few months)
        orders_df_for_plot = orders_df_for_plot.dropna(subset=[date_col_found])
        # choose granularity: if >18 months span, aggregate monthly, else daily
        span_days = (orders_df_for_plot[date_col_found].max() - orders_df_for_plot[date_col_found].min()).days if orders_df_for_plot.shape[0] > 0 else 0
        if span_days > 540:  # > ~18 months
            orders_df_for_plot['order_period'] = orders_df_for_plot[date_col_found].dt.to_period('M').dt.to_timestamp()
        else:
            orders_df_for_plot['order_period'] = orders_df_for_plot[date_col_found].dt.to_period('D').dt.to_timestamp()

        if 'order_id' in orders_df_for_plot.columns:
            trend = orders_df_for_plot.groupby('order_period').agg(num_orders=('order_id', 'count')).reset_index()
        else:
            trend = orders_df_for_plot.groupby('order_period').agg(num_orders=(date_col_found, 'count')).reset_index()

        if trend.shape[0] > 0:
            fig_line = px.line(trend, x='order_period', y='num_orders', title='Orders Trend', markers=True)
            fig_line.update_xaxes(title='Date')
            fig_line.update_yaxes(title='Number of orders')
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Found a date column but could not aggregate any orders to plot.")
    else:
        st.info("No parsable order date column found (checked: order_date, date, created_at, order_timestamp). Orders trend chart is hidden.")

    # 4) Box plot: monetary distribution by risk
    st.subheader("Monetary (avg order value) by risk label")
    fig_box = px.box(df_viz, x='risk_label', y='monetary', points='all', title='Monetary distribution by risk label')
    st.plotly_chart(fig_box, use_container_width=True)

with right_col:
    st.header("Top at-risk customers")
    top_n = st.number_input("How many top at-risk to show?", min_value=5, max_value=500, value=20, step=5)
    top_df = features_df.sort_values('risk_score', ascending=False).head(top_n)
    display_cols = ['customer_id', 'risk_score', 'risk_label', 'recency', 'frequency', 'monetary']
    display_cols = [c for c in display_cols if c in top_df.columns]
    st.dataframe(top_df[display_cols].reset_index(drop=True))

    # Suggested interventions for top customers
    st.subheader("Suggested interventions (top list)")
    top_df = top_df.copy()
    top_df['suggestion'] = top_df.apply(suggest_intervention, axis=1)
    st.dataframe(top_df[['customer_id', 'risk_label', 'risk_score', 'suggestion']].head(top_n).reset_index(drop=True))

    # Download predictions
    csv = features_df.to_csv(index=False)
    st.download_button("Download full predictions CSV", csv, file_name="customer_predictions.csv")

# selected customer details
if selected_customer and selected_customer != "-- none --":
    st.markdown("---")
    st.header(f"Customer details — {selected_customer}")
    cust_row = features_df[features_df['customer_id'] == str(selected_customer)]
    if cust_row.shape[0] == 0:
        st.write("Customer ID not found in prepared features.")
    else:
        st.write(cust_row.T)
        s = cust_row.iloc[0]
        st.subheader("Suggested action")
        st.info(suggest_intervention(s))

        # show customer's raw feedback (if any)
        if feedback is not None and isinstance(feedback, pd.DataFrame) and 'customer_id' in feedback.columns:
            fb = feedback[feedback['customer_id'].astype(str) == str(selected_customer)]
            if fb.shape[0] > 0:
                st.subheader("Customer feedback (raw)")
                cols = [c for c in ['feedback_text', 'rating', 'created_at'] if c in fb.columns]
                st.dataframe(fb[cols].head(50) if cols else fb.head(50))
            else:
                st.write("No feedback entries for this customer.")

st.markdown("---")
st.write("Notes: Visualizations are interactive (hover, zoom). For best results, train your model with `model_training.py` with good labels and re-run this dashboard. Transformer embeddings require `sentence-transformers` installed in the environment where the dashboard runs.")
