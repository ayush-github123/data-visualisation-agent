# agents/coder_agent.py
"""Enhanced Code Generation Agent with memory capabilities."""

import re
import json
import pandas as pd
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import CODER_MODEL, CODER_TEMPERATURE


class CoderAgent:
    def __init__(self, memory_manager=None):
        self.llm = ChatGoogleGenerativeAI(
            model=CODER_MODEL, 
            temperature=CODER_TEMPERATURE
        )
        self.parser = StrOutputParser()
        self.memory_manager = memory_manager
        
        # Enhanced prompt with memory context
        self.prompt = PromptTemplate.from_template("""
Generate optimal Python code for data visualization and analysis with memory awareness.

CURRENT TASK:
Task Plan: {task_plan}
Dataframe Info: {df_info}

MEMORY CONTEXT:
Previous Successful Code Patterns: {previous_patterns}
Similar Past Analyses: {similar_analyses}
Dataset Context: {dataset_context}

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

MEMORY-ENHANCED CODING:
- Learn from previous successful code patterns for similar tasks
- Reuse effective data processing techniques from memory
- Adapt successful visualization configurations from past analyses
- Improve upon previous approaches based on memory insights

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

    def write_code(self, task_plan: dict, df_info: dict):
        """Generate Python code with memory context."""
        # Get memory context if available
        memory_context = {}
        if self.memory_manager:
            # Extract successful code patterns from memory
            successful_analyses = self.memory_manager.get_successful_analyses()
            
            memory_context = {
                "previous_patterns": self._extract_code_patterns(successful_analyses),
                "similar_analyses": json.dumps([
                    {
                        "task": session.task_plan.get("task"),
                        "chart_type": session.task_plan.get("chart_type"),
                        "target_column": session.task_plan.get("target_column")
                    }
                    for session in successful_analyses[-3:]  # Last 3 successful
                ], indent=2),
                "dataset_context": json.dumps(self.memory_manager.get_dataset_context(), indent=2)
            }
        else:
            memory_context = {
                "previous_patterns": "No previous patterns available",
                "similar_analyses": "[]",
                "dataset_context": "{}"
            }

        # Create the chain
        chain = (
            RunnablePassthrough() 
            | self.prompt 
            | self.llm 
            | self.parser
        )
        
        return chain.invoke({
            "task_plan": task_plan,
            "df_info": df_info,
            **memory_context
        })
    
    def _extract_code_patterns(self, successful_analyses):
        """Extract common code patterns from successful analyses."""
        if not successful_analyses:
            return "No previous successful patterns available"
        
        patterns = []
        task_counts = {}
        chart_counts = {}
        
        for session in successful_analyses:
            task = session.task_plan.get("task", "unknown")
            chart_type = session.task_plan.get("chart_type", "unknown")
            
            task_counts[task] = task_counts.get(task, 0) + 1
            chart_counts[chart_type] = chart_counts.get(chart_type, 0) + 1
        
        patterns.append(f"Most used tasks: {dict(sorted(task_counts.items(), key=lambda x: x[1], reverse=True))}")
        patterns.append(f"Most used charts: {dict(sorted(chart_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        return "\n".join(patterns)


def clean_code(code_str: str) -> str:
    """Clean the generated code by removing markdown formatting."""
    # Remove markdown code blocks
    code_str = re.sub(r"^```(python)?", "", code_str.strip(), flags=re.IGNORECASE | re.MULTILINE)
    code_str = re.sub(r"```$", "", code_str.strip(), flags=re.MULTILINE)
    return code_str.strip()


def execute_generated_code(code_str, df: pd.DataFrame):
    """Execute the generated code and return the primary outputs."""
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