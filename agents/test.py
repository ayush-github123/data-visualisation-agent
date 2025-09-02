import streamlit as st
import pandas as pd
import plotly.express as px
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ------------------ Config ------------------
ALLOWED_TASKS = {"plot_distribution", "summarize_data", "correlation_analysis", "filter_rows", "group_by_aggregate"}
ALLOWED_CHARTS = {"bar", "line", "pie", "scatter", "histogram", "box", "none"}

# ------------------ Enhanced Planner Agent ------------------
llm_planner = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

intent_prompt = PromptTemplate.from_template(template="""
You are an expert data visualization specialist. Convert the user query into the optimal visualization strategy.

User Query: {user_query}
Available Columns: {df_columns}
Data Types: {df_dtypes}
Numeric Columns: {numeric_columns}
Categorical Columns: {categorical_columns}

CHART SELECTION RULES:
- **Categorical distribution**: Use "bar" chart (NOT pie unless specifically requested)
- **Numeric distribution**: Use "histogram" for single variable, "box" for comparing groups
- **Time series**: Use "line" chart
- **Correlation**: Use "scatter" for 2 variables
- **Comparison across categories**: Use "bar" chart
- **Part-to-whole relationships**: Use "pie" chart (only when explicitly asked)
- **Performance metrics/benchmarks**: Use "bar" or "line" based on data type

Return a JSON object with this exact format:
{{
  "task": "<one of: plot_distribution, summarize_data, correlation_analysis, filter_rows, group_by_aggregate>",
  "target_column": "<single column name or list for multiple columns>",
  "chart_type": "<one of: bar, line, pie, scatter, histogram, box, none>",
  "output_title": "<descriptive title for what will be shown>",
  "additional_params": {{
      "groupby": "<column name or null>",
      "filter_condition": "<pandas query string or null>",
      "aggregate_function": "<mean, sum, count, etc. or null>",
      "sort_values": "<asc, desc, or null>",
      "top_n": "<number or null>"
  }}
}}

IMPORTANT:
- For benchmark/performance data: Use bar charts to compare values across categories
- For token speed analysis: Use bar chart to compare speeds across different models/conditions
- For count/frequency data: Use bar chart (vertical bars work best)
- Only use pie charts when user explicitly asks for "pie chart" or "proportion"
- Always sort data appropriately (desc for performance metrics, asc for rankings)
""")

planner_parser = JsonOutputParser()
planner_chain = intent_prompt | llm_planner | planner_parser

def get_task_plan(user_query, df_columns, df_dtypes, df_sample):
    # Convert dtypes to string to avoid JSON serialization issues
    dtype_strings = {col: str(dtype) for col, dtype in df_dtypes.items()}
    
    # Identify numeric and categorical columns
    numeric_columns = [col for col, dtype in dtype_strings.items() if 'int' in dtype.lower() or 'float' in dtype.lower()]
    categorical_columns = [col for col, dtype in dtype_strings.items() if 'object' in dtype.lower() or 'string' in dtype.lower()]
    
    return planner_chain.invoke({
        "user_query": user_query,
        "df_columns": ", ".join(df_columns),
        "df_dtypes": ", ".join([f"{col}: {dtype}" for col, dtype in dtype_strings.items()]),
        "numeric_columns": ", ".join(numeric_columns),
        "categorical_columns": ", ".join(categorical_columns)
    })

# ------------------ Enhanced Task Plan Validator ------------------
def validate_task_plan(task_plan, df_columns):
    errors = []
    
    # Validate required keys
    required_keys = ["task", "target_column", "chart_type", "output_title", "additional_params"]
    for key in required_keys:
        if key not in task_plan:
            errors.append(f"Missing required key: {key}")
            return errors

    # Validate task
    if task_plan["task"] not in ALLOWED_TASKS:
        errors.append(f"Invalid task: {task_plan['task']}")

    # Validate chart type
    if task_plan["chart_type"] not in ALLOWED_CHARTS:
        errors.append(f"Invalid chart type: {task_plan['chart_type']}")

    # Validate target_column(s)
    target_col = task_plan["target_column"]
    if isinstance(target_col, str):
        target_col = [target_col]
    
    if isinstance(target_col, list):
        for col in target_col:
            if col not in df_columns:
                errors.append(f"Column '{col}' not found in dataset")

    # Validate additional_params
    additional_params = task_plan.get("additional_params", {})
    
    # Validate groupby
    groupby = additional_params.get("groupby")
    if groupby and groupby not in df_columns:
        errors.append(f"Groupby column '{groupby}' not found in dataset")
    
    # Validate sort_values
    sort_values = additional_params.get("sort_values")
    if sort_values and sort_values not in ["asc", "desc", "ascending", "descending", None]:
        errors.append(f"Invalid sort_values: {sort_values}")
    
    # Validate top_n
    top_n = additional_params.get("top_n")
    if top_n is not None:
        try:
            int(top_n)
        except (ValueError, TypeError):
            errors.append(f"Invalid top_n value: {top_n}")

    return errors

# ------------------ Enhanced Coder Agent ------------------
def write_code_agent(task_plan: dict, df_info: dict):
    code_prompt = PromptTemplate.from_template("""
Generate optimal Python code for data visualization and analysis.

TASK PLAN: {task_plan}
DATAFRAME INFO: {df_info}

VISUALIZATION BEST PRACTICES:
1. **Bar Charts**: 
   - Always sort values (descending for performance/metrics, ascending for categories)
   - Use horizontal bars for long labels
   - Limit to top 10-15 items for readability

2. **Histograms**: 
   - Use appropriate bin size (usually 20-30 bins)
   - Add proper labels and titles

3. **Scatter Plots**: 
   - Add trendline if correlation analysis
   - Use appropriate point size and opacity

4. **Line Charts**: 
   - Ensure x-axis is properly ordered (especially for time series)
   - Use markers for small datasets

CODE REQUIREMENTS:
- Create ONLY ONE primary output: `primary_fig`, `primary_table`, or `primary_summary`
- Use descriptive variable names
- Handle missing values appropriately
- Sort data logically (performance metrics: descending, categories: by value)
- Limit displayed items to top/bottom N for readability
- Add proper axis labels and titles
- Use reset_index() after value_counts() and rename columns clearly

EXAMPLE STRUCTURE:
```python
# Data preparation
analysis_df = df.copy()
# ... filtering/cleaning ...

# Analysis with proper sorting
if task involves distribution:
    result_df = analysis_df['column'].value_counts().head(10).reset_index()
    result_df.columns = ['Category', 'Count']
    result_df = result_df.sort_values('Count', ascending=False)  # or ascending=True
    primary_fig = px.bar(result_df, x='Category', y='Count', title='...')

# OR for summary
primary_table = result_df
```

Generate ONLY executable Python code, no markdown formatting.
""")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)  # Lower temperature for more consistent output
    parser = StrOutputParser()
    chain = code_prompt | llm | parser
    
    return chain.invoke({
        "task_plan": task_plan,
        "df_info": df_info
    })

# ------------------ Enhanced Code Execution ------------------
def clean_code(code_str: str) -> str:
    # Remove markdown code blocks
    code_str = re.sub(r"^```(python)?", "", code_str.strip(), flags=re.IGNORECASE | re.MULTILINE)
    code_str = re.sub(r"```$", "", code_str.strip(), flags=re.MULTILINE)
    return code_str.strip()

def execute_generated_code(code_str, df: pd.DataFrame):
    local_vars = {"df": df, "px": px, "pd": pd}
    code_str = clean_code(code_str)

    try:
        # Compile first to catch syntax errors
        compile(code_str, "<string>", "exec")
        exec(code_str, {"px": px, "pd": pd}, local_vars)

        # Extract primary outputs only
        primary_fig = local_vars.get("primary_fig")
        primary_table = local_vars.get("primary_table") 
        primary_summary = local_vars.get("primary_summary")

        return primary_fig, primary_table, primary_summary, code_str
        
    except Exception as e:
        raise RuntimeError(f"Code execution failed: {str(e)}")

# ------------------ Enhanced Explanation Agent ------------------
def explain_results(user_query, task_plan, primary_output_type, df_sample):
    explanation_prompt = PromptTemplate.from_template("""
Answer the user's question based on the analysis performed.

User Query: {user_query}
Analysis Type: {analysis_type}
Output Type: {output_type}
Data Sample: {data_sample}

Provide a clear, direct answer to the user's question. Focus on:
1. Direct answer to what they asked
2. Key insights from the analysis
3. Any important patterns or findings

Keep it concise and actionable. Avoid technical jargon.
""")

    llm_explainer = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    parser = StrOutputParser()
    chain = explanation_prompt | llm_explainer | parser

    return chain.invoke({
        "user_query": user_query,
        "analysis_type": task_plan["task"],
        "output_type": primary_output_type,
        "data_sample": df_sample.head(3).to_string() if isinstance(df_sample, pd.DataFrame) else str(df_sample)[:200]
    })

# ------------------ Enhanced Streamlit UI ------------------
def main():
    st.set_page_config(
        page_title="AI Data Analyst", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Analyst</h1>
        <p style="color: white; text-align: center; margin: 0;">
            Upload your data and ask questions in natural language
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_rows = st.number_input("Max rows for analysis", 1000, 50000, 5000, step=1000)
        show_code = st.checkbox("Show generated code", value=False)
        show_plan = st.checkbox("Show task plan", value=False)

    # File upload with better UX
    uploaded_file = st.file_uploader(
        "📁 Upload your dataset", 
        type=["csv"],
        help="Upload a CSV file to start analyzing your data"
    )
    
    if uploaded_file:
        try:
            with st.spinner("Loading dataset..."):
                df = pd.read_csv(uploaded_file)
            
            # Dataset info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))
            with col4:
                st.metric("Categorical Columns", len(df.select_dtypes(include='object').columns))

            # Data sampling for large datasets
            if len(df) > max_rows:
                st.info(f"📊 Large dataset detected ({len(df):,} rows). Using a sample of {max_rows:,} rows for analysis.")
                df_for_analysis = df.sample(max_rows, random_state=42)
            else:
                df_for_analysis = df

            # Data preview in expandable section
            with st.expander("📄 Data Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data info
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Column Types")
                    dtype_df = pd.DataFrame({
                        'Column': df.dtypes.index,
                        'Type': df.dtypes.values.astype(str)
                    })
                    st.dataframe(dtype_df, hide_index=True)
                
                with col2:
                    st.subheader("Missing Values")
                    missing_df = pd.DataFrame({
                        'Column': df.columns,
                        'Missing': df.isnull().sum().values,
                        'Percentage': (df.isnull().sum() / len(df) * 100).round(2).values
                    })
                    missing_df = missing_df[missing_df['Missing'] > 0]
                    if len(missing_df) > 0:
                        st.dataframe(missing_df, hide_index=True)
                    else:
                        st.success("No missing values found!")

            # Main query interface
            st.markdown("### 💭 Ask Your Question")
            
            # Example queries
            with st.expander("💡 Example Questions", expanded=False):
                st.markdown("""
                **Distribution Analysis:**
                - "Show the distribution of age"
                - "Create a pie chart for gender"
                
                **Correlation Analysis:**
                - "Show correlation between price and rating"
                - "How does salary relate to experience?"
                
                **Summary Statistics:**
                - "Summarize the sales data"
                - "Show average salary by department"
                
                **Filtering & Grouping:**
                - "Group sales by region and show totals"
                - "Filter data where age > 25"
                """)

            query = st.text_input(
                "Enter your question about the data:",
                placeholder="e.g., 'Show the distribution of sales by region'",
                help="Ask any question about your data in natural language"
            )

            if query:
                try:
                    # Create columns for layout
                    main_col, side_col = st.columns([3, 1])
                    
                    with main_col:
                        # Step 1: Planning
                        with st.spinner("🧠 Understanding your question..."):
                            # Convert dtypes to strings to avoid JSON serialization issues
                            dtypes_str = {col: str(dtype) for col, dtype in df.dtypes.items()}
                            df_info = {
                                "columns": df.columns.tolist(),
                                "dtypes": dtypes_str,
                                "shape": df.shape,
                                "numeric_columns": df.select_dtypes(include='number').columns.tolist(),
                                "categorical_columns": df.select_dtypes(include='object').columns.tolist()
                            }
                            task_plan = get_task_plan(query, df.columns.tolist(), dtypes_str, df_for_analysis)

                        # Validation
                        errors = validate_task_plan(task_plan, df.columns.tolist())
                        if errors:
                            st.error("❌ Could not understand your question:")
                            for error in errors:
                                st.error(f"• {error}")
                            st.info("💡 Try rephrasing your question or check column names in the data preview.")
                            return

                        # Step 2: Code Generation
                        with st.spinner("⚡ Generating analysis..."):
                            code_str = write_code_agent(task_plan, df_info)

                        # Step 3: Execution
                        with st.spinner("📊 Creating visualization..."):
                            primary_fig, primary_table, primary_summary, executed_code = execute_generated_code(code_str, df_for_analysis)

                        # Display results with better organization
                        st.markdown("### 📈 Analysis Results")
                        
                        # Display primary output with proper title
                        output_title = task_plan.get("output_title", "Analysis Result")
                        
                        if primary_fig:
                            st.subheader(f"📊 {output_title}")
                            # Enhance chart display with better configuration
                            st.plotly_chart(
                                primary_fig, 
                                use_container_width=True,
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                                }
                            )
                            
                        if primary_table is not None:
                            st.subheader(f"📋 {output_title}")
                            
                            # Format table display based on size
                            if isinstance(primary_table, pd.DataFrame):
                                if len(primary_table) > 20:
                                    st.dataframe(primary_table, use_container_width=True)
                                    st.info(f"Showing top {len(primary_table)} results. Full dataset has more entries.")
                                else:
                                    st.dataframe(primary_table, hide_index=True, use_container_width=True)
                            else:
                                st.dataframe(primary_table, use_container_width=True)

                        # Step 4: AI Explanation
                        st.markdown("### 💬 AI Insights")
                        with st.spinner("🔍 Analyzing results..."):
                            output_type = "visualization" if primary_fig else "table" if primary_table is not None else "summary"
                            explanation = explain_results(query, task_plan, output_type, df_for_analysis)
                        
                        # Display explanation in a nice box
                        st.markdown(f"""
                        <div class="info-box">
                            {explanation}
                        </div>
                        """, unsafe_allow_html=True)

                    # Sidebar information
                    with side_col:
                        st.markdown("#### 📋 Analysis Details")
                        
                        # Task info
                        st.info(f"**Task:** {task_plan['task'].replace('_', ' ').title()}")
                        st.info(f"**Chart:** {task_plan['chart_type'].title()}")
                        
                        if isinstance(task_plan['target_column'], list):
                            st.info(f"**Columns:** {len(task_plan['target_column'])} columns")
                        else:
                            st.info(f"**Column:** {task_plan['target_column']}")

                        # Optional detailed views
                        if show_plan:
                            with st.expander("🔍 Task Plan"):
                                st.json(task_plan)
                        
                        if show_code:
                            with st.expander("💻 Generated Code"):
                                st.code(executed_code, language="python")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Try simplifying your question or check if the column names are correct.")
        
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            st.info("💡 Please ensure your CSV file is properly formatted.")

    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Getting Started</h3>
            <p>Upload a CSV file to begin analyzing your data with AI!</p>
            <p><strong>What you can do:</strong></p>
            <ul>
                <li>📈 Create visualizations from natural language questions</li>
                <li>📊 Generate summary statistics and insights</li>
                <li>🔍 Filter and group your data</li>
                <li>📋 Get correlation analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()