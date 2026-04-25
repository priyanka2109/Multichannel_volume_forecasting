import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="Support Volume Dashboard", layout="wide")

# Load data
df = pd.read_csv("volume_forecasting.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Title
st.title("📞 Multichannel Support Volume Dashboard")

# KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Calls", f"{df['voice_calls'].sum():,}")
col2.metric("Total Chats", f"{df['chats'].sum():,}")
col3.metric("Total Emails", f"{df['emails'].sum():,}")
col4.metric("Avg Handle Time", f"{df['avg_handle_time'].mean():.2f} min")

# Line chart
st.subheader("Channel Trends Over Time")
fig = px.line(
    df,
    x='timestamp',
    y=['voice_calls', 'chats', 'emails', 'tickets'],
    title="Hourly Volume Trends"
)
st.plotly_chart(fig, use_container_width=True)

# Outage impact
st.subheader("Outage Impact")
fig2 = px.scatter(
    df,
    x='timestamp',
    y='voice_calls',
    color='outage_flag',
    title="Calls During Outages"
)
st.plotly_chart(fig2, use_container_width=True)

# SLA breaches
st.subheader("SLA Breach Distribution")
fig3 = px.histogram(
    df,
    x='sla_breach',
    title="SLA Breach Count"
)
st.plotly_chart(fig3, use_container_width=True)