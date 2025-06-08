import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load your API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Business Report", layout="centered")
st.title("📊 AI-Powered Business Report Generator")
st.write("Upload a CSV file and let AI analyze it for you!")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Auto-Generated Charts Based on Your Data")

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Limit to top 2 categorical × 2 numeric combinations to keep it readable
    max_charts = 2

    # Bar Charts: Categorical vs Numeric
    shown = 0
    for cat in categorical_cols:
        for num in numeric_cols:
            if shown >= max_charts:
                break
            st.write(f"📈 **{num} by {cat}**")
            grouped = df.groupby(cat)[num].sum().sort_values(ascending=False)

            fig, ax = plt.subplots()
            grouped.plot(kind="bar", ax=ax)
            ax.set_ylabel(num)
            ax.set_title(f"{num} by {cat}")
            st.pyplot(fig)
            shown += 1

    # Time Series Chart (if any column looks like a date)
    date_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
    if not date_cols:
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    break
                except:
                    pass

    if date_cols:
        st.subheader("📉 Time Series Plot (based on detected date column)")
        date_col = date_cols[0]
        for num in numeric_cols[:2]:
            ts = df[[date_col, num]].dropna()
            ts = ts.groupby(date_col)[num].sum()

            fig, ax = plt.subplots()
            ts.plot(ax=ax, marker="o")
            ax.set_title(f"{num} Over Time ({date_col})")
            ax.set_ylabel(num)
            st.pyplot(fig)
