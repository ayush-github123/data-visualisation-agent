import re
import json
import difflib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Utilities: schema + fuzzy matching
# -----------------------------

@dataclass
class Schema:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]


def infer_schema(df: pd.DataFrame) -> Schema:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # try to coerce object columns to datetime if they parse well
    dt_candidates = []
    for col in df.columns:
        if col in numeric_cols:
            continue
        s = df[col]
        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        parsed_ratio = parsed.notna().mean()
        if parsed_ratio > 0.8:  # mostly datetimes
            df[col] = parsed  # mutate df in place to datetime
            dt_candidates.append(col)

    datetime_cols = list({*dt_candidates, *df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()})

    # categoricals = non-numeric, non-datetime or numeric with low cardinality
    categorical_cols = []
    for col in df.columns:
        if col in numeric_cols or col in datetime_cols:
            continue
        # treat low-cardinality ints/floats as categorical too
        nunique = df[col].nunique(dropna=True)
        if nunique <= max(50, int(0.05 * len(df))):
            categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    return Schema(numeric=numeric_cols, categorical=categorical_cols, datetime=datetime_cols)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def fuzzy_pick(name: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    canon = {normalize(c): c for c in candidates}
    key = normalize(name)
    if key in canon:
        return canon[key]
    matches = difflib.get_close_matches(key, list(canon.keys()), n=1, cutoff=0.6)
    return canon[matches[0]] if matches else None

# -----------------------------
# Spec + parsing
# -----------------------------

@dataclass
class ChartSpec:
    task: str                     # scatter|bar|line|hist|heatmap
    x: Optional[str] = None
    y: Optional[str] = None
    agg: Optional[str] = None     # mean|sum|count|median|max|min
    color: Optional[str] = None
    bins: Optional[int] = None
    topk: Optional[int] = None
    trendline: bool = False
    groupby: Optional[str] = None
    title: Optional[str] = None


ALLOWED_TASKS = {"scatter", "bar", "line", "hist", "heatmap"}
ALLOWED_AGGS = {"mean", "sum", "count", "median", "max", "min"}


class RegexParser:
    def __init__(self, schema: Schema):
        self.schema = schema

    def _col(self, raw: str) -> Optional[str]:
        return fuzzy_pick(raw, list(self.schema.numeric + self.schema.categorical + self.schema.datetime))

    def parse(self, q: str) -> ChartSpec:
        q0 = q.strip().lower()

        # 1) correlation between X and Y
        m = re.search(r"correlation between ([\w ._-]+) and ([\w ._-]+)", q0)
        if m:
            x = self._col(m.group(1))
            y = self._col(m.group(2))
            if x and y:
                return ChartSpec(task="scatter", x=x, y=y, trendline=True, title=f"Correlation between {x} and {y}")

        # 2) avg/sum/count of metric by group
        m = re.search(r"(average|avg|mean|sum|count|median|max|min)\s+(?:of\s+)?([\w ._-]+)\s+(?:by|per)\s+([\w ._-]+)", q0)
        if m:
            agg_raw, metric_raw, group_raw = m.groups()
            agg = {"average":"mean","avg":"mean"}.get(agg_raw, agg_raw)
            agg = agg if agg in ALLOWED_AGGS else "mean"
            metric = self._col(metric_raw)
            group = self._col(group_raw)
            if metric and group:
                return ChartSpec(task="bar", x=group, y=metric, agg=agg, title=f"{agg.title()} of {metric} by {group}")

        # 3) top K <group> by <metric>
        m = re.search(r"top\s+(\d+)\s+([\w ._-]+)\s+by\s+([\w ._-]+)", q0)
        if m:
            k_raw, group_raw, metric_raw = m.groups()
            group = self._col(group_raw)
            metric = self._col(metric_raw)
            if group and metric:
                return ChartSpec(task="bar", x=group, y=metric, agg="sum", topk=int(k_raw), title=f"Top {k_raw} {group} by {metric}")

        # 4) trend/time series of metric (auto-pick datetime)
        m = re.search(r"(trend|time\s*series|over time)\s+(?:of\s+)?([\w ._-]+)", q0)
        if m:
            metric_raw = m.groups()[-1]
            metric = self._col(metric_raw)
            if metric:
                dt = self.schema.datetime[0] if self.schema.datetime else None
                if dt:
                    return ChartSpec(task="line", x=dt, y=metric, agg="sum", title=f"Trend of {metric} over time")

        # 5) distribution/histogram of metric
        m = re.search(r"(distribution|histogram)\s+(?:of\s+)?([\w ._-]+)", q0)
        if m:
            metric = self._col(m.groups()[-1])
            if metric:
                return ChartSpec(task="hist", x=metric, bins=30, title=f"Distribution of {metric}")

        # 6) correlation matrix / heatmap
        if re.search(r"(correlation matrix|heatmap)", q0):
            return ChartSpec(task="heatmap", title="Correlation Matrix")

        # fallback: if we have datetime + a numeric, produce a line
        if self.schema.datetime and self.schema.numeric:
            return ChartSpec(task="line", x=self.schema.datetime[0], y=self.schema.numeric[0], agg="sum", title=f"Trend of {self.schema.numeric[0]} over time")
        # else histogram of first numeric
        if self.schema.numeric:
            return ChartSpec(task="hist", x=self.schema.numeric[0], bins=30, title=f"Distribution of {self.schema.numeric[0]}")

        # last resort
        return ChartSpec(task="bar")

# -----------------------------
# Execution: figure + explanations
# -----------------------------

def build_figure(spec: ChartSpec, df: pd.DataFrame):
    if spec.task not in ALLOWED_TASKS:
        raise ValueError("Unsupported chart type")

    if spec.task == "scatter":
        fig = px.scatter(df, x=spec.x, y=spec.y, trendline="ols" if spec.trendline else None, title=spec.title)
        return fig

    if spec.task == "bar":
        if spec.x and spec.y:
            agg = spec.agg or "sum"
            g = df.groupby(spec.x, dropna=False)[spec.y].agg(agg).reset_index(name=f"{agg}_{spec.y}")
            # top-k if requested
            if spec.topk:
                g = g.sort_values(by=g.columns[-1], ascending=False).head(spec.topk)
            fig = px.bar(g, x=spec.x, y=g.columns[-1], title=spec.title)
            return fig
        # basic bar fallback: count by first categorical
        cat = spec.x or (spec.groupby or None)
        if not cat:
            # pick some categorical
            cat = df.select_dtypes(exclude=[np.number, "datetime", "datetimetz"]).columns.tolist()
            cat = cat[0] if cat else None
        if cat:
            g = df[cat].value_counts(dropna=False).reset_index()
            g.columns = [cat, "count"]
            return px.bar(g, x=cat, y="count", title=spec.title or f"Count by {cat}")

    if spec.task == "line":
        agg = spec.agg or "sum"
        x = spec.x
        y = spec.y
        if x is None:
            # find datetime
            dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
            x = dt_cols[0] if len(dt_cols) else None
        if x is None or y is None:
            raise ValueError("Line chart needs a datetime x and numeric y")
        g = df.groupby(pd.Grouper(key=x, freq="D"))[y].agg(agg).reset_index(name=f"{agg}_{y}")
        fig = px.line(g, x=x, y=g.columns[-1], title=spec.title)
        return fig

    if spec.task == "hist":
        if not spec.x:
            x = df.select_dtypes(include=[np.number]).columns
            spec.x = x[0] if len(x) else None
        fig = px.histogram(df, x=spec.x, nbins=spec.bins or 30, title=spec.title)
        return fig

    if spec.task == "heatmap":
        nums = df.select_dtypes(include=[np.number])
        if nums.empty:
            raise ValueError("No numeric columns for correlation matrix")
        corr = nums.corr(numeric_only=True)
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1, colorbar=dict(title="corr")))
        fig.update_layout(title=spec.title or "Correlation Matrix")
        return fig

    raise ValueError("Unhandled chart type")


def explain(spec: ChartSpec, df: pd.DataFrame) -> str:
    try:
        if spec.task == "scatter" and spec.x and spec.y:
            x = pd.to_numeric(df[spec.x], errors="coerce")
            y = pd.to_numeric(df[spec.y], errors="coerce")
            mask = x.notna() & y.notna()
            x, y = x[mask], y[mask]
            if len(x) < 3:
                return "Not enough data to compute correlation."
            r = np.corrcoef(x, y)[0, 1]
            # simple slope via least squares
            slope = np.polyfit(x, y, 1)[0]
            direction = "positive" if r >= 0 else "negative"
            return f"Pearson r = {r:.2f} ({direction} relationship). Slope ≈ {slope:.3f} per unit of {spec.x}."

        if spec.task == "bar" and spec.x and spec.y:
            agg = spec.agg or "sum"
            g = df.groupby(spec.x, dropna=False)[spec.y].agg(agg).sort_values(ascending=False)
            top = g.head(3)
            parts = ", ".join([f"{idx}: {val:.2f}" for idx, val in top.items()])
            return f"Top {min(3, len(top))} by {agg} of {spec.y} → {parts}."

        if spec.task == "line" and spec.x and spec.y:
            # daily aggregation same as build_figure
            agg = spec.agg or "sum"
            g = df.groupby(pd.Grouper(key=spec.x, freq="D"))[spec.y].agg(agg)
            g = g.dropna()
            if len(g) < 3:
                return "Not enough points to assess trend."
            x = np.arange(len(g))
            y = g.values
            slope = np.polyfit(x, y, 1)[0]
            trend = "upward" if slope > 0 else "downward"
            return f"Time trend appears {trend}; slope ≈ {slope:.3f} per day (aggregated by {agg})."

        if spec.task == "hist" and spec.x:
            s = pd.to_numeric(df[spec.x], errors="coerce").dropna()
            if s.empty:
                return "No numeric data to describe distribution."
            return f"Mean={s.mean():.2f}, Median={s.median():.2f}, Std={s.std():.2f}, Skew={s.skew():.2f}."

        if spec.task == "heatmap":
            nums = df.select_dtypes(include=[np.number])
            corr = nums.corr(numeric_only=True)
            # pick strongest pair excluding self correlations
            best = None
            for i, c1 in enumerate(corr.columns):
                for j, c2 in enumerate(corr.columns):
                    if i >= j:
                        continue
                    val = abs(corr.loc[c1, c2])
                    if best is None or val > best[0]:
                        best = (val, c1, c2)
            if best:
                return f"Strongest absolute correlation: {best[1]} vs {best[2]} → |r|={best[0]:.2f}."
    except Exception as e:
        return f"Explanation unavailable: {e}"

    return "Generated chart and summary."

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="LLM Viz Assistant — MVP", layout="wide")

st.title("📊 LLM‑powered Data Viz Assistant — MVP (No LLM)")
st.caption("Milestone 1: Spec‑driven agent loop using regex + fuzzy matching. Next: swap parser with an LLM.")

with st.sidebar:
    st.header("1) Load Data")
    up = st.file_uploader("Upload CSV", type=["csv"])

    if st.button("Generate sample dataset"):
        np.random.seed(7)
        n = 1000
        df_sample = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "ad_spend": np.random.gamma(5, 50, n).round(2),
            "revenue": 2000 + np.random.normal(0, 500, n) + np.linspace(0, 10000, n) + np.random.gamma(5, 30, n),
            "region": np.random.choice(["North", "South", "East", "West"], size=n),
            "orders": np.random.poisson(200, n),
            "price": np.clip(np.random.normal(50, 15, n), 5, None).round(2),
        })
        st.session_state["df"] = df_sample
        st.success("Sample dataset generated. Scroll down to query!")

    if up is not None:
        df = pd.read_csv(up)
        st.session_state["df"] = df
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    st.divider()
    st.header("2) Help")
    st.markdown(
        "Examples:\n\n- correlation between **revenue** and **ad spend**\n- average **revenue** by **region**\n- top 5 **region** by **revenue**\n- trend of **orders** over time\n- distribution of **price**\n- correlation matrix"
    )

# Data present?
if "df" not in st.session_state:
    st.info("Upload a CSV or generate a sample dataset from the sidebar.")
    st.stop()

df = st.session_state["df"].copy()
schema = infer_schema(df)

# Schema panel
with st.expander("🔎 Detected schema", expanded=False):
    st.json({"numeric": schema.numeric, "categorical": schema.categorical, "datetime": schema.datetime})

# Query box
q = st.text_input("Ask in natural language", placeholder="e.g., correlation between revenue and ad spend")

col_a, col_b = st.columns([1, 3])
with col_a:
    go_btn = st.button("Generate", type="primary")
with col_b:
    add_btn_disabled = "last_fig" not in st.session_state
    add_btn = st.button("➕ Add to dashboard", disabled=add_btn_disabled)

if "panels" not in st.session_state:
    st.session_state["panels"] = []

if go_btn and q.strip():
    parser = RegexParser(schema)
    spec = parser.parse(q)
    try:
        fig = build_figure(spec, df)
        text = explain(spec, df)
        st.session_state["last_spec"] = asdict(spec)
        st.session_state["last_fig"] = fig
        st.session_state["last_text"] = text
    except Exception as e:
        st.error(f"Failed to build chart: {e}")

# Show result
if "last_fig" in st.session_state:
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(st.session_state["last_fig"], use_container_width=True)
    with right:
        st.subheader("Explanation")
        st.write(st.session_state["last_text"])
        with st.expander("Spec JSON"):
            st.code(json.dumps(st.session_state["last_spec"], indent=2))

if add_btn and "last_fig" in st.session_state:
    st.session_state["panels"].append({
        "spec": st.session_state["last_spec"],
        "text": st.session_state["last_text"],
        "fig": st.session_state["last_fig"],
    })
    st.success("Added to dashboard.")

# Dashboard grid
if st.session_state["panels"]:
    st.subheader("📋 Dashboard")
    cols = st.columns(2)
    for i, panel in enumerate(st.session_state["panels"]):
        with cols[i % 2]:
            st.plotly_chart(panel["fig"], use_container_width=True)
            st.caption(panel["text"])

    # Export as a self-contained HTML
    import plotly.io as pio
    from pathlib import Path

    if st.button("💾 Export Dashboard (HTML)"):
        html_parts = [
            "<html><head><meta charset='utf-8'><title>Viz Dashboard</title></head><body>",
            "<h1>Viz Dashboard</h1>"
        ]
        for panel in st.session_state["panels"]:
            html_parts.append(f"<h3>{panel['spec'].get('title','Chart')}</h3>")
            html_parts.append(pio.to_html(panel["fig"], full_html=False, include_plotlyjs='cdn'))
            html_parts.append(f"<p><em>{panel['text']}</em></p>")
        html_parts.append("</body></html>")
        out = "dashboard.html"
        Path(out).write_text("\n".join(html_parts), encoding="utf-8")
        with open(out, "rb") as f:
            st.download_button("Download dashboard.html", f, file_name="dashboard.html", mime="text/html")

