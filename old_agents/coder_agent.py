


class CoderAgent:
    def __init__(self):
        pass

    def generate_code(self, task_plan: dict):
        task = task_plan.get("task")
        target = task_plan.get("target_column")
        chart = task_plan.get("chart_type", "none")
        params = task_plan.get("additional_params", {})
        groupby = params.get("groupby")
        percentile = params.get("percentile")

        code = "# Generated Python code for the requested analysis\n"
        
        if task == "plot_distribution":
            if chart == "bar":
                code += f"""
import matplotlib.pyplot as plt

df['{target}'].value_counts().plot(kind="bar")
plt.title("Distribution of {target}")
plt.xlabel("{target}")
plt.ylabel("Count")
plt.show()
"""
            elif chart == "hist":
                code += f"""
import matplotlib.pyplot as plt

df['{target}'].plot(kind="hist", bins=20)
plt.title("Histogram of {target}")
plt.xlabel("{target}")
plt.ylabel("Frequency")
plt.show()
"""

        elif task == "summarize_data":
            code += f"""
summary_stats = df['{target}'].describe()
print("Summary statistics for {target}:")
print(summary_stats)
"""

        elif task == "correlation_analysis":
            code += """
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
"""

        elif task == "filter_rows":
            # Example: filter above percentile
            if percentile:
                code += f"""
threshold = df['{target}'].quantile({percentile}/100)
filtered_df = df[df['{target}'] > threshold]
print("Filtered rows where {target} > {percentile}th percentile:")
print(filtered_df.head())
"""
            else:
                code += f"""
# Example filter: values greater than mean
filtered_df = df[df['{target}'] > df['{target}'].mean()]
print(filtered_df.head())
"""

        elif task == "group_by_aggregate":
            if groupby:
                code += f"""
grouped = df.groupby('{groupby}')['{target}'].mean().reset_index()
print("Mean {target} grouped by {groupby}:")
print(grouped)

import matplotlib.pyplot as plt
grouped.plot(x='{groupby}', y='{target}', kind='bar')
plt.title("Mean {target} by {groupby}")
plt.show()
"""
            else:
                code += "# No groupby column specified."

        else:
            code += "# Task not recognized. No code generated."

        return code
