# ui_components.py
"""Enhanced UI components with memory features for the Streamlit application."""

import streamlit as st
import pandas as pd
import json
from datetime import datetime


class UIComponents:
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

    def setup_page_config(self):
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="AI Data Analyst", 
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    def apply_custom_css(self):
        """Apply custom CSS styling."""
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
        .memory-box {
            background-color: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .chat-history {
            max-height: 300px;
            overflow-y: auto;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        .memory-stats {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 0.75rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        """Render the main application header with memory status."""
        memory_status = ""
        if self.memory_manager:
            stats = self.memory_manager.get_memory_stats()
            memory_status = f" | 🧠 {stats['total_analyses']} analyses remembered"
        
        st.markdown(f"""
        <div class="main-header">
            <h1>🤖 AI Data Analyst</h1>
            <p style="color: white; text-align: center; margin: 0;">
                Upload your data and ask questions in natural language{memory_status}
            </p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with configuration options and memory controls."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            max_rows = st.number_input("Max rows for analysis", 1000, 50000, 5000, step=1000)
            show_code = st.checkbox("Show generated code", value=False)
            show_plan = st.checkbox("Show task plan", value=False)
            
            # Memory section
            if self.memory_manager:
                st.header("🧠 Memory")
                show_memory = st.checkbox("Show memory panel", value=False)
                
                if st.button("Clear Memory", type="secondary"):
                    self.memory_manager.clear_memory()
                    st.success("Memory cleared!")
                    st.rerun()
                
                # Memory stats
                stats = self.memory_manager.get_memory_stats()
                st.markdown(f"""
                <div class="memory-stats">
                    <strong>Memory Stats:</strong><br>
                    💬 Messages: {stats['total_messages']}<br>
                    📊 Analyses: {stats['total_analyses']}<br>
                    ✅ Successful: {stats['successful_analyses']}<br>
                    ⏱️ Session: {stats['session_duration']}
                </div>
                """, unsafe_allow_html=True)
                
                return max_rows, show_code, show_plan, show_memory
            
            return max_rows, show_code, show_plan, False

    def render_file_uploader(self):
        """Render the file upload component."""
        return st.file_uploader(
            "📁 Upload your dataset", 
            type=["csv"],
            help="Upload a CSV file to start analyzing your data"
        )

    def render_dataset_metrics(self, df: pd.DataFrame):
        """Render dataset overview metrics."""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))
        with col4:
            st.metric("Categorical Columns", len(df.select_dtypes(include='object').columns))

    def render_data_preview(self, df: pd.DataFrame, dtype_df: pd.DataFrame, missing_df: pd.DataFrame):
        """Render the data preview section."""
        with st.expander("📄 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Column Types")
                st.dataframe(dtype_df, hide_index=True)
            
            with col2:
                st.subheader("Missing Values")
                if len(missing_df) > 0:
                    st.dataframe(missing_df, hide_index=True)
                else:
                    st.success("No missing values found!")

    def render_memory_panel(self):
        """Render the memory panel showing chat history and analysis history."""
        if not self.memory_manager:
            return
            
        st.markdown("### 🧠 Memory Panel")
        
        tab1, tab2, tab3 = st.tabs(["💬 Chat History", "📊 Analysis History", "📋 Session Data"])
        
        with tab1:
            messages = self.memory_manager.get_chat_history()
            if messages:
                st.markdown('<div class="chat-history">', unsafe_allow_html=True)
                for i, msg in enumerate(messages[-10:]):  # Show last 10 messages
                    role = "👤 You" if msg.__class__.__name__ == "HumanMessage" else "🤖 AI"
                    with st.container():
                        st.markdown(f"**{role}:** {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No chat history yet. Start asking questions!")
        
        with tab2:
            analyses = self.memory_manager.get_analysis_history()
            if analyses:
                for i, session in enumerate(analyses[-5:]):  # Show last 5 analyses
                    status = "✅" if session.success else "❌"
                    timestamp = datetime.fromisoformat(session.timestamp).strftime("%H:%M")
                    
                    with st.expander(f"{status} {timestamp} - {session.user_query[:50]}...", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Task:** {session.task_plan.get('task', 'N/A')}")
                            st.write(f"**Chart:** {session.task_plan.get('chart_type', 'N/A')}")
                            st.write(f"**Column:** {session.task_plan.get('target_column', 'N/A')}")
                        with col2:
                            st.write(f"**Result:** {session.result_type}")
                            st.write(f"**Success:** {'Yes' if session.success else 'No'}")
                            if session.error_message:
                                st.error(f"Error: {session.error_message}")
            else:
                st.info("No analysis history yet. Run some analyses!")
        
        with tab3:
            if self.memory_manager:
                session_data = self.memory_manager.export_session()
                st.json(session_data)
                
                # Download button for session data
                session_json = json.dumps(session_data, indent=2)
                st.download_button(
                    label="📥 Download Session Data",
                    data=session_json,
                    file_name=f"analysis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    def render_example_queries(self):
        """Render example queries section with memory-aware suggestions."""
        with st.expander("💡 Example Questions", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Distribution Analysis:**
                - "Show the distribution of age"
                - "Create a pie chart for gender"
                
                **Correlation Analysis:**
                - "Show correlation between price and rating"
                - "How does salary relate to experience?"
                """)
            
            with col2:
                st.markdown("""
                **Summary Statistics:**
                - "Summarize the sales data"
                - "Show average salary by department"
                
                **Memory-Aware Queries:**
                - "Show me that chart again but for different column"
                - "Compare this with the previous analysis"
                """)
            
            # Show suggestions based on memory
            if self.memory_manager:
                similar_queries = self.memory_manager.get_successful_analyses()
                if similar_queries:
                    st.markdown("**🔄 Recent Successful Queries:**")
                    for session in similar_queries[-3:]:
                        if st.button(f"↻ {session.user_query[:60]}...", key=f"repeat_{session.timestamp}"):
                            return session.user_query
        return None

    def render_query_input(self):
        """Render the query input component with memory suggestions."""
        st.markdown("### 💭 Ask Your Question")
        
        # Show memory-based suggestions
        suggested_query = self.render_example_queries()
        
        query = st.text_input(
            "Enter your question about the data:",
            value=suggested_query if suggested_query else "",
            placeholder="e.g., 'Show the distribution of sales by region'",
            help="Ask any question about your data in natural language. The AI remembers previous conversations!"
        )
        
        return query

    def render_results(self, primary_fig, primary_table, primary_summary, output_title):
        """Render the analysis results."""
        st.markdown("### 📈 Analysis Results")
        
        if primary_fig:
            st.subheader(f"📊 {output_title}")
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

    def render_explanation(self, explanation):
        """Render the AI explanation with memory context indicator."""
        st.markdown("### 💬 AI Insights")
        
        # Check if explanation was influenced by memory
        memory_indicator = ""
        if self.memory_manager and self.memory_manager.get_analysis_history():
            memory_indicator = " 🧠"
        
        st.markdown(f"""
        <div class="info-box">
            <strong>AI Analysis{memory_indicator}:</strong><br>
            {explanation}
        </div>
        """, unsafe_allow_html=True)

    def render_analysis_details(self, task_plan, show_plan, show_code, executed_code, show_memory=False):
        """Render analysis details in sidebar with memory information."""
        st.markdown("#### 📋 Analysis Details")
        
        # Task info
        st.info(f"**Task:** {task_plan['task'].replace('_', ' ').title()}")
        st.info(f"**Chart:** {task_plan['chart_type'].title()}")
        
        # Memory influence indicator
        if task_plan.get("memory_influenced") == "true":
            st.success("🧠 **Memory Enhanced:** This analysis used previous context")
        
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
        
        # Memory panel
        if show_memory and self.memory_manager:
            with st.expander("🧠 Memory Context"):
                self.render_memory_panel()

    def render_welcome_message(self):
        """Render welcome message when no file is uploaded."""
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
                <li>🧠 AI remembers your conversation and builds context</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show memory stats if available
        if self.memory_manager:
            stats = self.memory_manager.get_memory_stats()
            if stats['total_analyses'] > 0:
                st.markdown(f"""
                <div class="memory-box">
                    <h4>🧠 Session Memory</h4>
                    <p>The AI remembers {stats['total_analyses']} previous analyses and {stats['total_messages']} messages from this session.</p>
                </div>
                """, unsafe_allow_html=True)