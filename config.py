# config.py
"""Enhanced configuration settings for the AI Data Analyst application with memory."""

# Task and chart type configurations
ALLOWED_TASKS = {
    "plot_distribution", 
    "summarize_data", 
    "correlation_analysis", 
    "filter_rows", 
    "group_by_aggregate"
}

ALLOWED_CHARTS = {
    "bar", 
    "line", 
    "pie", 
    "scatter", 
    "histogram", 
    "box", 
    "none"
}

# Model configurations
PLANNER_MODEL = "gemini-2.0-flash"
PLANNER_TEMPERATURE = 0.3

CODER_MODEL = "gemini-2.0-flash"
CODER_TEMPERATURE = 0.1

EXPLAINER_MODEL = "gemini-2.0-flash"
EXPLAINER_TEMPERATURE = 0.3

# UI configurations
DEFAULT_MAX_ROWS = 5000
MAX_DISPLAY_ITEMS = 15
SAMPLE_RANDOM_STATE = 42

# Memory configurations
MEMORY_SETTINGS = {
    "max_chat_messages": 50,          # Maximum chat messages to keep
    "max_analysis_sessions": 20,      # Maximum analysis sessions to remember
    "context_window_exchanges": 3,    # Number of recent exchanges to include in context
    "similarity_threshold": 0.3,      # Threshold for similar query detection
    "enable_session_persistence": True, # Enable session persistence across browser refreshes
    "auto_clear_old_sessions": True,  # Automatically clear old sessions
    "session_timeout_hours": 24       # Hours after which session expires
}