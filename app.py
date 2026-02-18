import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

load_dotenv()

st.set_page_config(page_title="LLM DataViz Assistant", layout="wide")
st.title("LLM-powered Data Visualization Assistant")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("🔎 Data Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("Columns:", list(df.columns))

    st.sidebar.header('Select Columns & Filters')

    selected_columns = st.sidebar.multiselect(
        "Select columns to include in analysis",
        options=df.columns,
        default = list(df.columns)
    )

    if selected_columns:
        filtered_df = df[selected_columns].copy()


        numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.sidebar.subheader("Filter numeric columns (optional)")
            for col in numeric_cols:
                min_val, max_val = float(filtered_df[col].min()), float(filtered_df[col].max())
                selected_range = st.sidebar.slider(
                    f"{col} range", min_val, max_val, (min_val, max_val)
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])
                ]

        
        st.subheader("Filtered Data Preview")
        st.dataframe(filtered_df.head())

    st.subheader("Ask a Question about Your Data")
    user_query = st.text_input("Example: 'Show me correlation between revenue and ad spend'")

    if user_query:
        st.write(f"You asked: **{user_query}**")

        schema = ", ".join(df.columns)

        prompt = PromptTemplate.from_template(
            """
            You are an expert Data Visualization Assistant.
            The dataset has these columns: {schema}.
            The user asked: "{user_query}".

            Your task:
            1. First, provide a **clear explanation in Markdown** (no code).
            - Summarize what you understood from the question.
            - Describe what insights the chart(s) will provide.
            2. Then, provide **only Python code** in a ```python ... ``` block.
            - Use `df` for the dataframe and `px` (plotly.express) for charts.
            - Always store each chart in a variable named `fig1`, `fig2`, ... etc.
            - Do NOT show or display inside the code (no fig.show(), no st.plotly_chart).
            - Just define the figures.

            Format strictly like this:

            Explanation:
            <your explanation here>

            Code:
            ```python
            # your python code
            fig1 = px....
            fig2 = px....
            ```
            """
        )

        try:
            llm = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-2.5-flash")

            formatted_prompt = prompt.format(schema=schema, user_query=user_query)
            response = llm.invoke(formatted_prompt)
            content = response.content

            # Strip markdown formatting
            code_match = re.search(r"```python(.*?)```", content, re.DOTALL)

            if code_match:
                code = code_match.group(1).strip()
                explanation = re.sub(r"```python.*?```", "", content, flags=re.DOTALL).strip()
            else:
                code = content
                explanation = ""

            # Show explanation
            if explanation:
                st.subheader("📝 Explanation")
                st.write(explanation)

            # Show code
            st.subheader("💻 Generated Code")
            st.code(code, language="python")

            # Execute safely
            local_vars = {"df": df, "px": px}
            try:
                exec(code, {}, local_vars)
                figures = [v for k, v in local_vars.items() if k.startswith("fig")]
                if figures:
                    st.subheader("📊 Visualization")
                    for i, fig in enumerate(figures, 1):
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
                else:
                    st.error("No figures found in the generated code.")

            except Exception as e:
                st.error(f"Execution failed: {e}")
        except Exception as e:
            st.error(f"Execution failed: {e}")
else:
    st.info("👆 Upload a CSV file to get started")
