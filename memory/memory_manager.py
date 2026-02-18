import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st


@dataclass
class AnalysisSession:
    """Data class to store analysis session information."""
    timestamp: str
    user_query: str
    task_plan: Dict[str, Any]
    dataset_info: Dict[str, Any]
    result_type: str
    success: bool
    error_message: Optional[str] = None


class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """In-memory implementation of chat message history."""
    
    def __init__(self, session_id: str):
        print(f"[DEBUG] Initializing InMemoryChatMessageHistory for session: {session_id}")
        self.session_id = session_id
        self.messages: List[BaseMessage] = []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the store"""
        print(f"[DEBUG] Adding message to history: {message}")
        self.messages.append(message)
    
    def clear(self) -> None:
        """Clear all messages"""
        print("[DEBUG] Clearing all chat messages")
        self.messages = []
    
    def get_messages(self) -> List[BaseMessage]:
        """Retrieve all messages"""
        print(f"[DEBUG] Retrieving {len(self.messages)} messages")
        return self.messages


class MemoryManager:
    """Enhanced memory manager for the AI Data Analyst application."""
    
    def __init__(self):
        print("[DEBUG] Initializing MemoryManager")
        self.session_id = self._get_or_create_session_id()
        print(f"[DEBUG] Using session_id: {self.session_id}")
        self.chat_history = InMemoryChatMessageHistory(self.session_id)
        self.analysis_history: List[AnalysisSession] = []
        self.dataset_context: Dict[str, Any] = {}
        
    def _get_or_create_session_id(self) -> str:
        """Get or create a session ID using Streamlit session state."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[DEBUG] Created new session_id: {st.session_state.session_id}")
        else:
            print(f"[DEBUG] Found existing session_id: {st.session_state.session_id}")
        return st.session_state.session_id
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to chat history."""
        print(f"[DEBUG] Adding user message: {message}")
        self.chat_history.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to chat history."""
        print(f"[DEBUG] Adding AI message: {message}")
        self.chat_history.add_message(AIMessage(content=message))
    
    def get_chat_history(self) -> List[BaseMessage]:
        """Get the complete chat history."""
        print("[DEBUG] Fetching chat history")
        return self.chat_history.get_messages()
    
    def get_recent_chat_context(self, num_exchanges: int = 3) -> str:
        """Get recent chat context as formatted string."""
        print(f"[DEBUG] Getting recent chat context (last {num_exchanges} exchanges)")
        messages = self.chat_history.get_messages()
        if not messages:
            print("[DEBUG] No messages in chat history")
            return ""
        
        recent_messages = messages[-(num_exchanges * 2):]
        context = []
        for msg in recent_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            context.append(f"{role}: {msg.content}")
        
        result = "\n".join(context)
        print(f"[DEBUG] Recent chat context:\n{result}")
        return result
    
    def set_dataset_context(self, df_info: Dict[str, Any], filename: str = None) -> None:
        """Store current dataset context."""
        print(f"[DEBUG] Setting dataset context for file: {filename}")
        self.dataset_context = {
            "filename": filename,
            "columns": df_info.get("columns", []),
            "dtypes": df_info.get("dtypes", {}),
            "shape": df_info.get("shape", (0, 0)),
            "numeric_columns": df_info.get("numeric_columns", []),
            "categorical_columns": df_info.get("categorical_columns", []),
            "loaded_at": datetime.now().isoformat()
        }
        print(f"[DEBUG] Dataset context set: {self.dataset_context}")
    
    def get_dataset_context(self) -> Dict[str, Any]:
        """Get current dataset context."""
        print("[DEBUG] Fetching dataset context")
        return self.dataset_context
    
    def add_analysis_session(self, user_query: str, task_plan: Dict[str, Any], 
                           dataset_info: Dict[str, Any], result_type: str, 
                           success: bool, error_message: str = None) -> None:
        """Store an analysis session."""
        print(f"[DEBUG] Adding analysis session for query: {user_query}")
        session = AnalysisSession(
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            task_plan=task_plan,
            dataset_info=dataset_info,
            result_type=result_type,
            success=success,
            error_message=error_message
        )
        self.analysis_history.append(session)
        print(f"[DEBUG] Total analysis sessions: {len(self.analysis_history)}")
        
        if len(self.analysis_history) > 10:
            print("[DEBUG] Trimming analysis history to last 10 entries")
            self.analysis_history = self.analysis_history[-10:]
    
    def get_analysis_history(self) -> List[AnalysisSession]:
        """Get the analysis history."""
        print(f"[DEBUG] Fetching analysis history ({len(self.analysis_history)} sessions)")
        return self.analysis_history
    
    def get_successful_analyses(self) -> List[AnalysisSession]:
        """Get only successful analyses."""
        successes = [session for session in self.analysis_history if session.success]
        print(f"[DEBUG] Found {len(successes)} successful analyses")
        return successes
    
    def get_similar_queries(self, current_query: str, similarity_threshold: float = 0.3) -> List[AnalysisSession]:
        """Get queries similar to the current one (simple keyword matching)."""
        print(f"[DEBUG] Finding similar queries for: '{current_query}'")
        if not self.analysis_history:
            print("[DEBUG] No analysis history available")
            return []
        
        current_words = set(current_query.lower().split())
        similar_sessions = []
        
        for session in self.analysis_history:
            if not session.success:
                continue
            
            session_words = set(session.user_query.lower().split())
            intersection = len(current_words & session_words)
            union = len(current_words | session_words)
            similarity = intersection / union if union > 0 else 0
            print(f"[DEBUG] Query '{session.user_query}' similarity: {similarity:.2f}")
            
            if similarity >= similarity_threshold:
                similar_sessions.append(session)
        
        print(f"[DEBUG] Found {len(similar_sessions)} similar queries")
        return similar_sessions
    
    def get_context_for_llm(self, current_query: str) -> Dict[str, Any]:
        """Get comprehensive context for LLM including chat history and dataset info."""
        print(f"[DEBUG] Building context for LLM with query: {current_query}")
        context = {
            "session_id": self.session_id,
            "current_query": current_query,
            "recent_chat_context": self.get_recent_chat_context(),
            "dataset_context": self.get_dataset_context(),
            "previous_successful_analyses": [
                {
                    "query": session.user_query,
                    "task": session.task_plan.get("task"),
                    "chart_type": session.task_plan.get("chart_type"),
                    "target_column": session.task_plan.get("target_column")
                }
                for session in self.get_successful_analyses()[-3:]
            ],
            "similar_past_queries": [
                {
                    "query": session.user_query,
                    "task_plan": session.task_plan
                }
                for session in self.get_similar_queries(current_query)
            ]
        }
        print(f"[DEBUG] LLM context built: {json.dumps(context, indent=2)}")
        return context
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        print("[DEBUG] Clearing memory (chat, analysis, dataset)")
        self.chat_history.clear()
        self.analysis_history.clear()
        self.dataset_context.clear()
    
    def export_session(self) -> Dict[str, Any]:
        """Export current session data."""
        print("[DEBUG] Exporting session data")
        data = {
            "session_id": self.session_id,
            "chat_messages": [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content
                }
                for msg in self.chat_history.get_messages()
            ],
            "analysis_history": [asdict(session) for session in self.analysis_history],
            "dataset_context": self.dataset_context
        }
        print(f"[DEBUG] Exported session data: {json.dumps(data, indent=2)}")
        return data
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "total_messages": len(self.chat_history.get_messages()),
            "total_analyses": len(self.analysis_history),
            "successful_analyses": len(self.get_successful_analyses()),
            "dataset_loaded": bool(self.dataset_context),
            "session_duration": self._calculate_session_duration()
        }
        print(f"[DEBUG] Memory stats: {stats}")
        return stats
    
    def _calculate_session_duration(self) -> str:
        """Calculate session duration from first message."""
        messages = self.chat_history.get_messages()
        print(f"[DEBUG] Calculating session duration based on {len(messages)} messages")
        if not messages:
            return "0 minutes"
        return f"{len(messages) * 2} minutes (estimated)"
