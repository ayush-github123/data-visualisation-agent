from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import plotly.express as px
import pandas as pd


# ------------------ LLM Code Generation ------------------

def write_code_agent(task_plan: dict):
    """
    Generate Python code from task plan using Gemini.
    """

    code_prompt = PromptTemplate.from_template(
        template="""
You are a Python coding agent working with pandas and plotly for data analysis and visualization.

You are given a `task_plan` in JSON format that specifies:
- The user's task (e.g., plot, summarize, correlate, group)
- The column(s) involved
- The chart type (if any)
- Any additional parameters (e.g., groupby, percentile)

Your job is to generate Python code that:
- Assumes a DataFrame named `df` is already loaded
- Uses plotly (px) for charts
- Assigns each chart to a variable like fig1, fig2, etc.
- Includes all necessary imports
- Only returns valid Python code — no explanations, markdown, or comments

Here is the task plan:
{task_plan}

Return Python code only:
"""
    )

    llm = ChatGoogleGenerativeAI(temperature=0.7, model='gemini-2.0-flash')
    parser = StrOutputParser()

    llm_chain = code_prompt | llm | parser

    result = llm_chain.invoke({"task_plan": json.dumps(task_plan)})
    return result


# ------------------ Code Execution ------------------

def execute_generated_code(code_str: str, df: pd.DataFrame):
    """
    Executes LLM-generated code safely using a restricted local scope.
    """
    local_vars = {
        "df": df,
        "px": px
    }

    try:
        exec(code_str, {}, local_vars)

        # Show generated figures (if any)
        figures = [v for k, v in local_vars.items() if k.startswith("fig")]
        if figures:
            print("✅ Code executed successfully. Plot(s) generated.")
            for i, fig in enumerate(figures, 1):
                fig.show()
        else:
            print("⚠️ No figures found. Check if the code assigned plots to fig1, fig2, etc.")

    except Exception as e:
        print(f"❌ Code execution failed: {e}")
