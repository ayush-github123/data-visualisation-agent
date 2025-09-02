import re


class Planner:
    """
    Agent to Plan things + refine the user prompt or query
    """

    def __init__(self):
        self.chart_keywords = {
            "bar": ["bar", "histogram", "count"],
            "line": ["line", "trend", "time series"],
            "scatter": ["scatter", "correlation", "relationship"],
            "pie": ["pie", "distribution", "share"]
        }

        # EDA / statistical keywords
        self.analysis_keywords = {
            "quantiles": ["quantile", "quartile"],
            "percentile": ["percentile"],
            "summary": ["mean", "median", "std", "describe", "summary"],
            "correlation": ["correlation", "corr"],
            "groupby": ["group by", "average by", "mean by"]
        }

    def detect_chart_type(self, query):
        chart=[]
        query_lower = query.lower()
        if query_lower:
            for chart, keywords in self.chart_keywords.items():
                for kw in keywords:
                    if kw in query_lower:
                        return chart
            return "bar"
        else:
            return "Query is needed..."
        

    def detect_analysis_types(self, query):
        query_lower = query.lower()
        detected = []
        for analysis, keywords in self.analysis_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    detected.append(analysis)
                    break
        return detected
    

    def detect_columns(self, query, df_columns):
        query_lower = query.lower()
        matched = [col for col in df_columns if col.lower() in query_lower]
        return matched if matched else df_columns[:1]  # default: first column
        

    def detect_groupby_column(self, query, df_columns):
        # Simple regex to find "by <column>"
        pattern = r"by (\w+)"
        match = re.search(pattern, query.lower())
        if match:
            col = match.group(1)
            # Check if matched word is a column
            for c in df_columns:
                if c.lower() == col:
                    return c
        return None

    def detect_percentile(self, query):
        # Find numbers before "percentile" in query
        pattern = r"(\d+)[a-z]* percentile"
        match = re.search(pattern, query.lower())
        if match:
            return int(match.group(1))
        return None

    def plan(self, user_query, df_columns):
        chart_type = self.detect_chart_type(user_query)
        target_columns = self.detect_columns(user_query, df_columns)
        analysis_types = self.detect_analysis_types(user_query)
        groupby_column = self.detect_groupby_column(user_query, df_columns)
        percentile = self.detect_percentile(user_query)

        task_plan = {
            "user_query": user_query,
            "target_columns": target_columns,
            "chart_type": chart_type,
            "analysis_type": analysis_types,
            "groupby_column": groupby_column,
            "percentile": percentile
        }

