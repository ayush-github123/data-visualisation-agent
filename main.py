# main.py
"""Enhanced AI Data Analyst with memory capabilities."""

import streamlit as st
from dotenv import load_dotenv

# Import our custom modules
from agents.data_processor import DataProcessor
from agents.planner_agent import PlannerAgent, validate_task_plan
from agents.coder_agent import CoderAgent, execute_generated_code
from agents.explainer_agent import ExplainerAgent
from ui_components import UIComponents
from memory.memory_manager import MemoryManager

# Load environment variables
load_dotenv()


class AIDataAnalyst:
    def __init__(self):
        # Initialize memory manager first
        # self.memory_manager = MemoryManager()
        
        if "memory_manager" not in st.session_state:
            st.session_state.memory_manager = MemoryManager()
        
        self.memory_manager = st.session_state.memory_manager


        # Initialize all agents with memory
        self.data_processor = DataProcessor()
        self.planner_agent = PlannerAgent(memory_manager=self.memory_manager)
        self.coder_agent = CoderAgent(memory_manager=self.memory_manager)
        self.explainer_agent = ExplainerAgent(memory_manager=self.memory_manager)
        self.ui = UIComponents(memory_manager=self.memory_manager)

    def run(self):
        """Main application runner."""
        # Setup page
        self.ui.setup_page_config()
        self.ui.apply_custom_css()
        self.ui.render_header()

        # Sidebar configuration (now includes memory controls)
        sidebar_result = self.ui.render_sidebar()
        if len(sidebar_result) == 4:
            max_rows, show_code, show_plan, show_memory = sidebar_result
        else:
            max_rows, show_code, show_plan = sidebar_result
            show_memory = False

        # File upload
        uploaded_file = self.ui.render_file_uploader()
        
        if uploaded_file:
            self._handle_file_upload(uploaded_file, max_rows, show_code, show_plan, show_memory)
        else:
            self.ui.render_welcome_message()
            
            # Show memory panel even without data if there's history
            if show_memory and self.memory_manager.get_memory_stats()['total_analyses'] > 0:
                self.ui.render_memory_panel()

    def _handle_file_upload(self, uploaded_file, max_rows, show_code, show_plan, show_memory):
        """Handle the uploaded file and run analysis."""
        # Load dataset
        df, error = self.data_processor.load_csv(uploaded_file)
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Please ensure your CSV file is properly formatted.")
            return

        # Store dataset context in memory
        df_info = self.data_processor.get_dataframe_info(df)
        self.memory_manager.set_dataset_context(df_info, uploaded_file.name)

        # Display dataset metrics
        self.ui.render_dataset_metrics(df)

        # Handle large datasets
        df_for_analysis, was_sampled = self.data_processor.sample_large_dataset(df, max_rows)
        if was_sampled:
            st.info(f"📊 Large dataset detected ({len(df):,} rows). Using a sample of {max_rows:,} rows for analysis.")

        # Data preview
        dtype_df = self.data_processor.get_column_types_info(df)
        missing_df = self.data_processor.get_missing_values_info(df)
        self.ui.render_data_preview(df, dtype_df, missing_df)

        # Query interface
        query = self.ui.render_query_input()

        # Show memory panel if requested
        if show_memory:
            self.ui.render_memory_panel()

        if query:
            self._process_query(query, df_for_analysis, show_code, show_plan, show_memory)

    def _process_query(self, query, df_for_analysis, show_code, show_plan, show_memory):
        """Process the user query and generate results with memory integration."""
        try:
            # Create columns for layout
            main_col, side_col = st.columns([3, 1])
            
            with main_col:
                # Step 1: Planning (with memory)
                with st.spinner("🧠 Understanding your question..."):
                    df_info = self.data_processor.get_dataframe_info(df_for_analysis)
                    task_plan = self.planner_agent.get_task_plan(
                        query, 
                        df_for_analysis.columns.tolist(), 
                        df_info["dtypes"], 
                        df_for_analysis
                    )

                # Validation
                errors = validate_task_plan(task_plan, df_for_analysis.columns.tolist())
                if errors:
                    # Store failed analysis in memory
                    self.memory_manager.add_analysis_session(
                        query, task_plan, df_info, "error", False, "; ".join(errors)
                    )
                    
                    st.error("❌ Could not understand your question:")
                    for error in errors:
                        st.error(f"• {error}")
                    st.info("💡 Try rephrasing your question or check column names in the data preview.")
                    return

                # Step 2: Code Generation (with memory)
                with st.spinner("⚡ Generating analysis..."):
                    code_str = self.coder_agent.write_code(task_plan, df_info)

                # Step 3: Execution
                with st.spinner("📊 Creating visualization..."):
                    primary_fig, primary_table, primary_summary, executed_code = execute_generated_code(
                        code_str, df_for_analysis
                    )

                # Determine result type
                result_type = "visualization" if primary_fig else "table" if primary_table is not None else "summary"

                # Store successful analysis in memory
                self.memory_manager.add_analysis_session(
                    query, task_plan, df_info, result_type, True
                )

                # Display results
                output_title = task_plan.get("output_title", "Analysis Result")
                self.ui.render_results(primary_fig, primary_table, primary_summary, output_title)

                # Step 4: AI Explanation (with memory)
                with st.spinner("🔍 Analyzing results..."):
                    explanation = self.explainer_agent.explain_results(
                        query, task_plan, result_type, df_for_analysis
                    )
                
                self.ui.render_explanation(explanation)

            # Sidebar information
            with side_col:
                self.ui.render_analysis_details(task_plan, show_plan, show_code, executed_code, show_memory)

        except Exception as e:
            # Store failed analysis in memory
            if hasattr(self, 'memory_manager'):
                self.memory_manager.add_analysis_session(
                    query, {}, {}, "error", False, str(e)
                )
            
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Try simplifying your question or check if the column names are correct.")


def main():
    """Entry point for the application."""
    # Initialize session state for memory persistence
    if 'memory_initialized' not in st.session_state:
        st.session_state.memory_initialized = True
    
    app = AIDataAnalyst()
    app.run()


if __name__ == "__main__":
    main()