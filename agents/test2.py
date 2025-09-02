# multi_agent_data_analysis_prod_ready.py
import os
import re
import json
import uuid
import time
import shlex
import queue
import tempfile
import logging
import subprocess
from typing import Tuple, List, Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

# LLM libs (user must have these)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# Optional: pydantic for schema validation
try:
    from pydantic import BaseModel, validator
    PydanticAvailable = True
except Exception:
    PydanticAvailable = False

# ----------------------------- Basic config & logging -----------------------------
load_env_key = None
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configure logging for observability
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # In production, add FileHandler or a proper logger/ship to central logging
    ],
)
logger = logging.getLogger("multi-agent-data-analysis")

# Check for expected environment secrets (example)
LLM_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    logger.warning("No LLM API key found in environment variables (GOOGLE_API_KEY or LLM_API_KEY). LLM calls may fail.")

# ----------------------------- Constants & Allowed sets -----------------------------
ALLOWED_TASKS = {"plot_distribution", "summarize_data", "correlation_analysis", "filter_rows", "group_by_aggregate"}
ALLOWED_CHARTS = {"bar", "line", "pie", "scatter", "none"}
# Allowed imports inside generated code
ALLOWED_IMPORTS = {"pandas", "pd", "plotly", "plotly.express", "px", "numpy", "np", "math"}
# Blacklist tokens - if any appear in code -> reject
BLACKLIST_PATTERNS = [
    r"\bsubprocess\b",
    r"\bos\.system\b",
    r"\bopen\(",
    r"\beval\(",
    r"\bexec\(",
    r"__import__",
    r"import\s+socket",
    r"import\s+requests",
    r"import\s+urllib",
    r"import\s+ftplib",
    r"import\s+paramiko",
    r"\bsocket\b",
    r"\bshutil\b",
    r"\bsys\.",
    r"\bos\.",
    r"\bpasswd\b",
    r"\b/etc/passwd\b",
    r"\brm\s+-rf\b",
]

# ----------------------------- LLM wrappers with caching & retry -----------------------------
# Create LLM clients lazily to avoid failing at import-time if keys missing
def create_llm_client(model: str = "gemini-2.0-flash", temp: float = 0.7):
    return ChatGoogleGenerativeAI(model=model, temperature=temp)

# Use streamlit caching for planner and code generator to avoid repeated LLM calls for same inputs
@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_get_task_plan(user_query: str, df_columns: List[str]) -> dict:
    """Cache planner results keyed by (query + columns)."""
    logger.info("Invoking planner LLM for query=%s", user_query)
    llm_planner = create_llm_client()
    intent_prompt = PromptTemplate.from_template(template="""
You are a data analysis assistant. Your job is to convert natural language queries into structured tasks for a data visualization tool.

User Query: {user_query}
Available Columns: {df_columns}

Your output MUST be a JSON object in the following format:
{{
  "task": "<one of: plot_distribution, summarize_data, correlation_analysis, filter_rows, group_by_aggregate>",
  "target_column": "<a single column name, or a list of column names (if user says 'all columns' or 'everything')>",
  "chart_type": "<one of: bar, line, pie, scatter, none>",
  "additional_params": {{
      "groupby": "<valid column or null>",
      "percentile": "<number or null>"
  }}
}}

Important Rules:
- If the user asks to visualize "all columns", "each column", or "everything", return all valid columns in a list.
- Only include columns from df_columns.
- If no clear column is mentioned, return the first valid column.
""")
    planner_parser = JsonOutputParser()
    planner_chain = intent_prompt | llm_planner | planner_parser

    # simple retry loop
    for attempt in range(2):
        try:
            raw = planner_chain.invoke({
                "user_query": user_query,
                "df_columns": ", ".join(df_columns)
            })
            logger.debug("Planner output: %s", raw)
            return raw
        except Exception as e:
            logger.warning("Planner LLM call failed (attempt %d): %s", attempt + 1, e)
            time.sleep(0.5)
    raise RuntimeError("Planner agent failed after retries.")

@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_write_code_agent(task_plan: dict) -> str:
    """Cache code generation for identical task plans."""
    logger.info("Invoking coder LLM for task_plan=%s", task_plan)
    code_prompt = PromptTemplate.from_template("""
You are an expert Python coding agent working with pandas and plotly for data analysis and visualization.

You are given a `task_plan` in JSON format.  

Your job is to generate detailed Python code that:
- Assumes a DataFrame named `df` is already loaded (loaded from a CSV file path passed in variable INPUT_CSV_PATH).
- Produces outputs depending on the task type:
    - For visualizations → use plotly (px) and assign to variables `fig1`, `fig2`, etc.
    - For tabular summaries → assign the result to variables `table1`, `table2`, etc.
    - For descriptive text/numeric summaries → assign to variables `summary1`, `summary2`, etc.
- Always use multiple clear steps (create intermediate DataFrames before plotting/aggregating).
- Always create a separate DataFrame for value_counts(), groupby(), or aggregation before plotting.
- Do NOT pass .value_counts() results directly to px.bar — instead reset_index and rename columns.
- If `target_column` is a list of length more than 1 and of distinct columns, generate a chart or table for each column in that list using a loop, storing results in `locals()[f"fig{{i}}"]` or `locals()[f"table{{i}}"]`.
- Code must be explicit and detailed, not minimal one-liners. Show filtering, renaming, and grouping steps clearly.
- Only output valid executable Python code — no markdown, no comments, no natural language explanations.

Here is the task plan:
{task_plan}
""")
    llm = create_llm_client()
    parser = StrOutputParser()
    chain = code_prompt | llm | parser

    # simple retry
    for attempt in range(2):
        try:
            code_str = chain.invoke({"task_plan": json.dumps(task_plan)})
            logger.debug("Coder output length=%d", len(code_str or ""))
            return code_str
        except Exception as e:
            logger.warning("Coder LLM call failed (attempt %d): %s", attempt + 1, e)
            time.sleep(0.5)
    raise RuntimeError("Coder agent failed after retries.")

# ----------------------------- Validation with Pydantic (if available) -----------------------------
if PydanticAvailable:
    class TaskPlanModel(BaseModel):
        task: str
        target_column: Any
        chart_type: str
        additional_params: dict

        @validator("task")
        def task_allowed(cls, v):
            if v not in ALLOWED_TASKS:
                raise ValueError(f"Invalid task: {v}")
            return v

        @validator("chart_type")
        def chart_allowed(cls, v):
            if v not in ALLOWED_CHARTS:
                raise ValueError(f"Invalid chart type: {v}")
            return v

        @validator("additional_params")
        def ensure_additional_params(cls, v):
            if not isinstance(v, dict):
                raise ValueError("additional_params must be a dict")
            # ensure keys exist
            if "groupby" not in v:
                v["groupby"] = None
            if "percentile" not in v:
                v["percentile"] = None
            return v

    def validate_task_plan(task_plan: dict, df_columns: List[str]) -> List[str]:
        errors = []
        try:
            model = TaskPlanModel(**task_plan)
        except Exception as e:
            errors.append(f"Task plan schema error: {e}")
            return errors

        # Validate target columns exist
        target_col = model.target_column
        if isinstance(target_col, str):
            target_col = [target_col]
        if target_col is None:
            errors.append("target_column is null")
        else:
            for c in target_col:
                if c not in df_columns:
                    errors.append(f"Invalid column: {c}")

        groupby = model.additional_params.get("groupby")
        if groupby and groupby not in df_columns:
            errors.append(f"Invalid groupby column: {groupby}")

        return errors

else:
    # Lightweight validation fallback
    def validate_task_plan(task_plan: dict, df_columns: List[str]) -> List[str]:
        errors = []
        if "task" not in task_plan or task_plan["task"] not in ALLOWED_TASKS:
            errors.append(f"Invalid task: {task_plan.get('task')}")
        if "chart_type" not in task_plan or task_plan["chart_type"] not in ALLOWED_CHARTS:
            errors.append(f"Invalid chart type: {task_plan.get('chart_type')}")
        target_col = task_plan.get("target_column")
        if isinstance(target_col, str):
            target_col = [target_col]
        if target_col is None:
            errors.append("target_column is null")
        else:
            for col in target_col:
                if col not in df_columns:
                    errors.append(f"Invalid column: {col}")
        groupby = task_plan.get("additional_params", {}).get("groupby")
        if groupby and groupby not in df_columns:
            errors.append(f"Invalid groupby column: {groupby}")
        return errors

# ----------------------------- Code sanitization & subprocess execution -----------------------------
def contains_blacklist(code_str: str) -> Tuple[bool, str]:
    """Return (True, match) if blacklisted pattern found."""
    for pat in BLACKLIST_PATTERNS:
        if re.search(pat, code_str, flags=re.IGNORECASE):
            return True, pat
    return False, ""

def check_for_disallowed_imports(code_str: str) -> Tuple[bool, List[str]]:
    """Return (has_disallowed, list_of_disallowed). We'll allow a small set of imports only."""
    # find import lines
    disallowed = []
    for line in code_str.splitlines():
        m = re.match(r"\s*import\s+([\w\.]+)", line)
        if m:
            token = m.group(1)
            if token not in ALLOWED_IMPORTS:
                disallowed.append(line.strip())
        m2 = re.match(r"\s*from\s+([\w\.]+)\s+import\s+(.+)", line)
        if m2:
            token = m2.group(1)
            if token not in ALLOWED_IMPORTS:
                disallowed.append(line.strip())
    return (len(disallowed) > 0, disallowed)

def prepare_runner_script(user_code: str, input_csv_path: str, output_json_path: str) -> str:
    """
    Wrap the user_code into a script that:
    - loads df from input_csv_path
    - executes user_code (which should produce figX, tableX, summaryX variables)
    - serializes outputs to output_json_path
    """
    wrapper = f"""
import json
import pandas as pd
import plotly.io as pio
from plotly.graph_objs import Figure
import traceback

# Load DataFrame from CSV (input provided by the controller)
INPUT_CSV = r\"\"\"{input_csv_path}\"\"\"

try:
    df = pd.read_csv(INPUT_CSV)
except Exception as e:
    out = {{
        "error": "Failed loading input CSV: " + str(e),
        "stdout": "",
        "stderr": ""
    }}
    with open(r\"\"\"{output_json_path}\"\"\", "w", encoding="utf-8") as f:
        json.dump(out, f)
    raise SystemExit(1)

# Begin user code (generated). The user code should assume df is available.
{user_code}

# Prepare outputs
result = {{
    "figures": [],
    "tables": {{}},
    "texts": {{}},
    "error": None
}}

try:
    # Figures: gather any variables starting with fig that are plotly figures
    for var_name in list(globals().keys()):
        if var_name.startswith("fig"):
            try:
                val = globals()[var_name]
                # convert Plotly figure to JSON string
                if isinstance(val, Figure):
                    result["figures"].append({{"name": var_name, "json": pio.to_json(val)}})
                else:
                    # attempt to convert; if not possible, skip
                    try:
                        result["figures"].append({{"name": var_name, "json": pio.to_json(val)}})
                    except Exception:
                        pass
            except Exception:
                pass

    # Tables: pick up pandas DataFrame or Series objects
    for var_name, val in list(globals().items()):
        if var_name.startswith("table") or var_name.startswith("df_") or var_name.endswith("_table"):
            try:
                if isinstance(val, (pd.DataFrame, pd.Series)):
                    # We'll serialize as JSON (orient=split) to preserve index/columns
                    result["tables"][var_name] = val.to_json(orient="split", date_format="iso")
            except Exception:
                pass

    # Text outputs
    for var_name, val in list(globals().items()):
        if var_name.startswith("summary") or var_name.startswith("text") or var_name.startswith("insight"):
            try:
                if isinstance(val, str):
                    result["texts"][var_name] = val
            except Exception:
                pass

except Exception as e:
    result["error"] = "Postprocessing error: " + str(e) + "\\n" + traceback.format_exc()

with open(r\"\"\"{output_json_path}\"\"\", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
"""
    return wrapper

def execute_generated_code_subprocess(code_str: str, df: pd.DataFrame, timeout: int = 12) -> Tuple[List[Any], Dict[str, Any], Dict[str, str]]:
    """
    Execute the user-provided code (string) in a separate subprocess.
    Input DataFrame is saved to a temporary CSV and given to the child script.
    The child script writes back results to a temporary JSON file.
    Returns: (figures, tables, texts)
    Figures are returned as plotly Figure objects (reconstructed from JSON).
    Tables are returned as pandas DataFrame objects.
    Texts are returned as dict of strings.
    """
    # sanitize code first
    has_black, pat = contains_blacklist(code_str)
    if has_black:
        raise RuntimeError(f"Rejected generated code due to blacklisted pattern: {pat}")

    has_disallowed_imports, disallowed = check_for_disallowed_imports(code_str)
    if has_disallowed_imports:
        raise RuntimeError(f"Rejected generated code due to disallowed imports: {disallowed}")

    # Write input CSV
    tmp_dir = tempfile.mkdtemp(prefix="code-run-")
    input_csv = os.path.join(tmp_dir, "input.csv")
    output_json = os.path.join(tmp_dir, "output.json")
    script_path = os.path.join(tmp_dir, "runner.py")

    df.to_csv(input_csv, index=False)

    # Prepare runner script (wraps user's code)
    runner_script = prepare_runner_script(code_str, input_csv, output_json)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(runner_script)

    # Execute runner script with a subprocess for isolation
    try:
        proc = subprocess.run(
            ["python3", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Execution timed out after {timeout}s. Stdout: {e.stdout if hasattr(e, 'stdout') else ''} Stderr: {e.stderr if hasattr(e, 'stderr') else ''}")
    except Exception as e:
        raise RuntimeError(f"Execution error launching subprocess: {e}")

    # Read output JSON if present
    if not os.path.exists(output_json):
        # Try to return some diagnostics
        raise RuntimeError(f"Runner did not produce output file. stdout: {proc.stdout}\nstderr: {proc.stderr}")

    with open(output_json, "r", encoding="utf-8") as f:
        result = json.load(f)

    if result.get("error"):
        # include child's stdout/stderr for debugging
        raise RuntimeError(f"Child script reported error: {result['error']}\nChild stdout:\n{proc.stdout}\nChild stderr:\n{proc.stderr}")

    figures = []
    for fig_entry in result.get("figures", []):
        try:
            fig_json = fig_entry.get("json")
            if fig_json:
                fig_obj = pio.from_json(fig_json)
                figures.append(fig_obj)
        except Exception:
            logger.exception("Failed to reconstruct figure %s", fig_entry.get("name"))

    tables = {}
    for k, v in result.get("tables", {}).items():
        try:
            # v is JSON produced earlier by pandas.to_json(orient='split')
            df_table = pd.read_json(v, orient="split")
            tables[k] = df_table
        except Exception:
            logger.exception("Failed to reconstruct table %s", k)

    texts = result.get("texts", {})

    return figures, tables, texts

# ----------------------------- Explanation Agent -----------------------------
explanation_prompt = PromptTemplate.from_template("""
You are a data analysis assistant, and your task is to explain data visualizations and summary statistics.

Inputs:
- Visualization type: {chart_type}
- Summary statistics (JSON or table): {summary_stats}
- Data insights (text or context): {data_insights}

Your output should:
1. Describe what is shown in the chart/table.
2. Highlight trends, correlations, or anomalies.
3. Give clear, concise insights or recommendations.

Keep the explanation user-friendly and actionable.
""")

def explain_data_visualization(chart_type, summary_stats, data_insights):
    logger.info("Invoking explanation agent for chart_type=%s", chart_type)
    llm_explainer = create_llm_client()
    parser = StrOutputParser()
    chain = explanation_prompt | llm_explainer | parser
    for attempt in range(2):
        try:
            return chain.invoke({
                "chart_type": chart_type,
                "summary_stats": json.dumps(summary_stats),
                "data_insights": data_insights
            })
        except Exception as e:
            logger.warning("Explainer LLM failed (attempt %d): %s", attempt + 1, e)
            time.sleep(0.5)
    return "Explanation agent failed to produce output."

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Multi-Agent Data Analysis (Hardened)", layout="wide")
st.title("📊 Multi-Agent Data Analysis & Visualization (Hardened Prototype)")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# Sidebar controls
MAX_ROWS = st.sidebar.number_input("Max rows for visualization", 1000, 100000, 5000, step=1000)
if len(df) > MAX_ROWS:
    st.warning(f"Large dataset detected ({len(df):,} rows). Using a sample of {MAX_ROWS:,} rows.")
    df_for_plot = df.sample(MAX_ROWS, random_state=42)
else:
    df_for_plot = df.copy()

st.subheader("📄 Data Preview (first 50 rows)")
st.dataframe(df.head(50))

st.markdown("**Columns:** " + ", ".join(list(df.columns)))

query = st.text_input("Ask a question (e.g., 'Show pie chart of gender distribution')")

if query:
    # Step 1: Planner Agent
    with st.spinner("Planning task..."):
        try:
            task_plan = cached_get_task_plan(query, df.columns.tolist())
        except Exception as e:
            logger.exception("Planner failed")
            st.error(f"Planner agent failed: {e}")
            st.stop()

    st.subheader("🧠 Task Plan (generated)")
    st.json(task_plan)

    # Allow user to edit the task plan before validation/execution
    st.caption("You may edit the task plan below before validation (must remain valid JSON).")
    task_plan_edit_text = st.text_area("Editable task plan JSON", value=json.dumps(task_plan, indent=2), height=180)
    try:
        task_plan_user = json.loads(task_plan_edit_text)
    except Exception as e:
        st.error(f"Edited task plan is not valid JSON: {e}")
        st.stop()

    # Step 2: Validation
    errors = validate_task_plan(task_plan_user, df.columns.tolist())
    if errors:
        st.error("❌ Task plan validation failed:\n" + "\n".join(errors))
        st.stop()

    # Step 3: Coder Agent (generate code)
    with st.spinner("Generating Python code..."):
        try:
            code_str_raw = cached_write_code_agent(task_plan_user)
        except Exception as e:
            logger.exception("Coder failed")
            st.error(f"Coder agent failed: {e}")
            st.stop()

    # Clean code fences if LLM returned them
    def clean_code(code_str: str) -> str:
        code_str = re.sub(r"^```(python)?", "", code_str.strip(), flags=re.IGNORECASE | re.MULTILINE)
        code_str = re.sub(r"```$", "", code_str.strip(), flags=re.MULTILINE)
        return code_str.strip()

    cleaned_code = clean_code(code_str_raw)
    st.subheader("🧾 Generated Code (from coder agent)")
    st.code(cleaned_code, language="python")

    # Step 4: Preview sanitization results
    try:
        blacklisted, bad_pat = contains_blacklist(cleaned_code)
        has_disallowed_imports, disallowed = check_for_disallowed_imports(cleaned_code)
    except Exception as e:
        st.error(f"Error sanitizing generated code: {e}")
        st.stop()

    if blacklisted:
        st.error(f"Generated code contains disallowed pattern: {bad_pat}. Execution blocked.")
        st.stop()
    if has_disallowed_imports:
        st.error(f"Generated code has disallowed imports: {disallowed}. Execution blocked.")
        st.stop()

    # Give the user option to run the code
    run_button = st.button("Run generated code (isolated subprocess)")
    if not run_button:
        st.info("Click 'Run generated code' to execute the generated script in a secure subprocess.")
        st.stop()

    # Step 5: Execute in subprocess
    with st.spinner("Executing generated code (isolated)..."):
        try:
            figures, tables, texts = execute_generated_code_subprocess(cleaned_code, df_for_plot, timeout=25)

            if figures:
                st.subheader("📈 Figures")
                for i, fig in enumerate(figures):
                    st.plotly_chart(fig, use_container_width=True, key=f"fig_{i}")

            if tables:
                st.subheader("📊 Tables")
                for name, tbl in tables.items():
                    st.write(f"**{name}**")
                    st.dataframe(tbl)

            if texts:
                st.subheader("📝 Text Outputs")
                for name, txt in texts.items():
                    st.write(f"**{name}**")
                    st.write(txt)

            if not (figures or tables or texts):
                st.info("Execution completed but no figures/tables/texts were returned by the generated code.")

            # Step 6: Explanation
            st.subheader("💬 Explanation")
            # Choose representative summary
            summary_stats = {}
            if tables:
                # reuse first table for explanation
                _, first_tbl = next(iter(tables.items()))
                try:
                    summary_stats = first_tbl.head(10).to_dict()
                except Exception:
                    summary_stats = {}

            data_insights = ""
            if texts:
                _, first_txt = next(iter(texts.items()))
                data_insights = first_txt

            explanation_text = explain_data_visualization(
                chart_type=task_plan_user.get("chart_type", "none"),
                summary_stats=summary_stats,
                data_insights=data_insights,
            )
            st.write(explanation_text)

        except Exception as e:
            logger.exception("Execution failed")
            st.error(str(e))
            st.stop()
