import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Multichannel Volume Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📞 Multichannel Contact Volume Forecasting")
st.markdown("VAR-based forecasting across voice, chat, email and ticket channels")

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("volume_forecasting.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def load_forecast():
    fc = pd.read_csv("var_forecast.csv")
    fc['timestamp'] = pd.to_datetime(fc['timestamp'])
    return fc

df = load_data()

try:
    forecast_df = load_forecast()
    has_forecast = True
except FileNotFoundError:
    has_forecast = False
    st.warning("⚠️ var_forecast.csv not found. Run the notebook to generate forecasts. Showing raw data only.")

CHANNELS = ['voice_calls', 'chats', 'emails', 'tickets']
CHANNEL_LABELS = {
    'voice_calls': 'Voice Calls',
    'chats': 'Chats',
    'emails': 'Emails',
    'tickets': 'Tickets'
}
COLORS = {
    'voice_calls': '#2c7bb6',
    'chats':       '#1a9641',
    'emails':      '#d7191c',
    'tickets':     '#fdae61'
}

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_channel = st.sidebar.selectbox(
    "Channel",
    options=CHANNELS,
    format_func=lambda x: CHANNEL_LABELS[x]
)

if has_forecast:
    view_mode = st.sidebar.radio(
        "View",
        ["Forecast vs Actual", "All Channels", "Outage Impact", "SLA Breaches"]
    )
else:
    view_mode = st.sidebar.radio(
        "View",
        ["All Channels", "Outage Impact", "SLA Breaches"]
    )

# ── KPIs ──────────────────────────────────────────────────────────
st.markdown("### Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Calls",  f"{df['voice_calls'].sum():,.0f}")
col2.metric("Total Chats",  f"{df['chats'].sum():,.0f}")
col3.metric("Total Emails", f"{df['emails'].sum():,.0f}")
col4.metric("Total Tickets",f"{df['tickets'].sum():,.0f}")
col5.metric("Avg Handle Time", f"{df['avg_handle_time'].mean():.2f} min")

st.markdown("---")

# ── FORECAST VS ACTUAL ────────────────────────────────────────────
if view_mode == "Forecast vs Actual" and has_forecast:
    st.subheader(f"VAR Forecast vs Actual — {CHANNEL_LABELS[selected_channel]}")

    # Merge actual test period with forecast
    test_actual = df.set_index('timestamp')[selected_channel]

    # Only show the test window (rows that exist in forecast_df)
    fc_indexed = forecast_df.set_index('timestamp')

    fig = go.Figure()

    # Actual line
    fig.add_trace(go.Scatter(
        x=test_actual.index,
        y=test_actual.values,
        name="Actual",
        line=dict(color=COLORS[selected_channel], width=1.5)
    ))

    # Forecast line (test window only)
    fig.add_trace(go.Scatter(
        x=fc_indexed.index,
        y=fc_indexed[selected_channel + '_forecast'],
        name="VAR Forecast",
        line=dict(color='#FF6B00', width=2, dash='dash')
    ))

    # Confidence band (±1 std of residuals as a simple proxy)
    residuals = test_actual.loc[fc_indexed.index] - fc_indexed[selected_channel + '_forecast']
    std = residuals.std()
    fig.add_trace(go.Scatter(
        x=fc_indexed.index.tolist() + fc_indexed.index.tolist()[::-1],
        y=(fc_indexed[selected_channel + '_forecast'] + std).tolist() +
          (fc_indexed[selected_channel + '_forecast'] - std).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,0,0,0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        name="±1 std band",
        showlegend=True
    ))

    fig.update_layout(
        height=420,
        xaxis_title="Time",
        yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics row ───────────────────────────────────────────────
    actual_vals   = test_actual.loc[fc_indexed.index]
    forecast_vals = fc_indexed[selected_channel + '_forecast']
    mae   = np.mean(np.abs(actual_vals - forecast_vals))
    rmse  = np.sqrt(np.mean((actual_vals - forecast_vals) ** 2))
    # sMAPE — avoids divide-by-zero issues
    smape = np.mean(2 * np.abs(actual_vals - forecast_vals) /
                    (np.abs(actual_vals) + np.abs(forecast_vals) + 1e-8)) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",   f"{mae:.2f}")
    m2.metric("RMSE",  f"{rmse:.2f}")
    m3.metric("sMAPE", f"{smape:.2f}%")

    st.caption("sMAPE (Symmetric MAPE) used instead of MAPE to avoid division-by-zero on near-zero values.")

    # ── Per-channel summary table ─────────────────────────────────
    st.markdown("#### Model Performance — All Channels")
    rows = []
    for ch in CHANNELS:
        a = test_actual_all = df.set_index('timestamp')[ch].loc[fc_indexed.index]
        f = fc_indexed[ch + '_forecast']
        rows.append({
            "Channel": CHANNEL_LABELS[ch],
            "MAE":     round(np.mean(np.abs(a - f)), 2),
            "RMSE":    round(np.sqrt(np.mean((a - f) ** 2)), 2),
            "sMAPE":   round(np.mean(2 * np.abs(a - f) / (np.abs(a) + np.abs(f) + 1e-8)) * 100, 2)
        })
    st.dataframe(pd.DataFrame(rows).set_index("Channel"), use_container_width=True)


# ── ALL CHANNELS ──────────────────────────────────────────────────
elif view_mode == "All Channels":
    st.subheader("Channel Volume Trends Over Time")

    fig = go.Figure()
    for ch in CHANNELS:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df[ch],
            name=CHANNEL_LABELS[ch],
            line=dict(color=COLORS[ch], width=1.2)
        ))

    fig.update_layout(
        height=420,
        xaxis_title="Time", yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Channel share pie
    st.subheader("Channel Volume Share")
    totals = {CHANNEL_LABELS[ch]: df[ch].sum() for ch in CHANNELS}
    fig2 = px.pie(
        names=list(totals.keys()),
        values=list(totals.values()),
        color_discrete_sequence=list(COLORS.values())
    )
    fig2.update_layout(height=360)
    st.plotly_chart(fig2, use_container_width=True)


# ── OUTAGE IMPACT ─────────────────────────────────────────────────
elif view_mode == "Outage Impact":
    st.subheader("Outage Impact on Call Volume")

    fig = px.scatter(
        df, x='timestamp', y='voice_calls',
        color='outage_flag',
        color_discrete_map={0: COLORS['voice_calls'], 1: '#d7191c'},
        labels={'outage_flag': 'Outage', 'voice_calls': 'Voice Calls'},
        title="Voice Calls — Outage vs Normal Periods"
    )
    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.markdown("#### Average Volume During vs Outside Outages")
    outage_summary = df.groupby('outage_flag')[CHANNELS].mean().round(1)
    outage_summary.index = ['Normal', 'Outage']
    st.dataframe(outage_summary, use_container_width=True)


# ── SLA BREACHES ──────────────────────────────────────────────────
elif view_mode == "SLA Breaches":
    st.subheader("SLA Breach Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        fig3 = px.histogram(
            df, x='sla_breach',
            title="SLA Breach Distribution",
            color_discrete_sequence=['#d7191c']
        )
        fig3.update_layout(height=360)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        # SLA breach rate over time (rolling)
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['sla_breach_rate'] = df_sorted['sla_breach'].rolling(50, min_periods=1).mean() * 100
        fig4 = px.line(
            df_sorted, x='timestamp', y='sla_breach_rate',
            title="Rolling SLA Breach Rate (%)",
            color_discrete_sequence=['#d7191c']
        )
        fig4.update_layout(height=360, yaxis_title="Breach Rate (%)")
        st.plotly_chart(fig4, use_container_width=True)

    # SLA breach by channel volume bucket
    st.markdown("#### SLA Breach Rate by Volume Quartile")
    df['volume_quartile'] = pd.qcut(df[selected_channel], q=4,
                                     labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    breach_by_vol = df.groupby('volume_quartile', observed=True)['sla_breach'].mean().reset_index()
    breach_by_vol['sla_breach'] = (breach_by_vol['sla_breach'] * 100).round(2)
    fig5 = px.bar(
        breach_by_vol,
        x='volume_quartile', y='sla_breach',
        labels={'sla_breach': 'Breach Rate (%)', 'volume_quartile': f'{CHANNEL_LABELS[selected_channel]} Volume Quartile'},
        color_discrete_sequence=['#d7191c']
    )
    fig5.update_layout(height=340)
    st.plotly_chart(fig5, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Vector Autoregression (VAR) | Evaluation: MAE, RMSE, sMAPE | Built with Streamlit + Plotly")