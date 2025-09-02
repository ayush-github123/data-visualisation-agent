"""Enhanced Planning Agent with memory capabilities."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import PLANNER_MODEL, PLANNER_TEMPERATURE, ALLOWED_TASKS, ALLOWED_CHARTS
import json


class PlannerAgent:
    def __init__(self, memory_manager=None):
        self.llm = ChatGoogleGenerativeAI(
            model=PLANNER_MODEL, 
            temperature=PLANNER_TEMPERATURE
        )
        self.parser = JsonOutputParser()
        self.memory_manager = memory_manager
        
        # Enhanced prompt with memory context
        self.prompt = PromptTemplate.from_template(template="""
You are an expert data visualization specialist with memory of previous interactions.

CURRENT REQUEST:
User Query: {user_query}
Available Columns: {df_columns}
Data Types: {df_dtypes}
Numeric Columns: {numeric_columns}
Categorical Columns: {categorical_columns}

MEMORY CONTEXT:
Previous Chat Context: {chat_context}
Dataset Context: {dataset_context}
Similar Past Queries: {similar_queries}
Previous Successful Analyses: {previous_analyses}

CHART SELECTION RULES:
- **Categorical distribution**: Use "bar" chart (NOT pie unless specifically requested)
- **Numeric distribution**: Use "histogram" for single variable, "box" for comparing groups
- **Time series**: Use "line" chart
- **Correlation**: Use "scatter" for 2 variables
- **Comparison across categories**: Use "bar" chart
- **Part-to-whole relationships**: Use "pie" chart (only when explicitly asked)
- **Performance metrics/benchmarks**: Use "bar" or "line" based on data type

MEMORY-ENHANCED ANALYSIS:
- Consider previous successful analyses for similar queries
- Build upon previous context when the user refers to "that chart" or "the previous analysis"
- Use memory to understand follow-up questions and refinements
- If user asks for variations of previous analyses, adapt accordingly

Return a JSON object with this exact format:
{{
  "task": "<one of: plot_distribution, summarize_data, correlation_analysis, filter_rows, group_by_aggregate>",
  "target_column": "<single column name or list for multiple columns>",
  "chart_type": "<one of: bar, line, pie, scatter, histogram, box, none>",
  "output_title": "<descriptive title for what will be shown>",
  "memory_influenced": "<true/false - whether this plan was influenced by memory context>",
  "additional_params": {{
      "groupby": "<column name or null>",
      "filter_condition": "<pandas query string or null>",
      "aggregate_function": "<mean, sum, count, etc. or null>",
      "sort_values": "<asc, desc, or null>",
      "top_n": "<number or null>"
  }}
}}

IMPORTANT:
- Use memory context to provide better, more contextual responses
- For benchmark/performance data: Use bar charts to compare values across categories
- Only use pie charts when user explicitly asks for "pie chart" or "proportion"
- Always sort data appropriately (desc for performance metrics, asc for rankings)
""")

    def get_task_plan(self, user_query, df_columns, df_dtypes, df_sample):
        """Generate a task plan with memory context."""
        # Convert dtypes to string to avoid JSON serialization issues
        dtype_strings = {col: str(dtype) for col, dtype in df_dtypes.items()}
        
        # Identify numeric and categorical columns
        numeric_columns = [
            col for col, dtype in dtype_strings.items() 
            if 'int' in dtype.lower() or 'float' in dtype.lower()
        ]
        categorical_columns = [
            col for col, dtype in dtype_strings.items() 
            if 'object' in dtype.lower() or 'string' in dtype.lower()
        ]
        
        # Get memory context if available
        memory_context = {}
        if self.memory_manager:
            context = self.memory_manager.get_context_for_llm(user_query)
            memory_context = {
                "chat_context": context.get("recent_chat_context", "No previous context"),
                "dataset_context": json.dumps(context.get("dataset_context", {}), indent=2),
                "similar_queries": json.dumps(context.get("similar_past_queries", []), indent=2),
                "previous_analyses": json.dumps(context.get("previous_successful_analyses", []), indent=2)
            }
        else:
            memory_context = {
                "chat_context": "No memory available",
                "dataset_context": "{}",
                "similar_queries": "[]",
                "previous_analyses": "[]"
            }

        # Create the chain
        chain = (
            RunnablePassthrough() 
            | self.prompt 
            | self.llm 
            | self.parser
        )
        
        return chain.invoke({
            "user_query": user_query,
            "df_columns": ", ".join(df_columns),
            "df_dtypes": ", ".join([f"{col}: {dtype}" for col, dtype in dtype_strings.items()]),
            "numeric_columns": ", ".join(numeric_columns),
            "categorical_columns": ", ".join(categorical_columns),
            **memory_context
        })


def validate_task_plan(task_plan, df_columns):
    """Validate the generated task plan against available data and allowed operations."""
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