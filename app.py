import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

# Load API Key from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

st.set_page_config(page_title="AI Business Report", layout="centered")
st.title("📊 AI-Powered Business Report Generator")
st.write("Upload a CSV file and let AI analyze it for you!")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("⚠️ The uploaded file is empty. Please upload a valid CSV.")
            st.stop()

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        # Auto Charts
        st.subheader("📊 Auto-Generated Charts Based on Your Data")
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        max_charts = 2
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

        # Time Series
        date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()
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

        # AI Insights
        st.subheader("💡 AI-Generated Business Insights")
        prompt = f"""
You are a business analyst. Read the following dataset (first 100 rows) and provide 3 key business insights, such as trends, anomalies, or recommendations.

Data:
{df.head(100).to_csv(index=False)}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        insights = response.choices[0].message.content
        st.markdown(insights)

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is empty or unreadable. Please choose another CSV.")
    except Exception as e:
        st.error(f"❌ Error generating insights: {e}")
