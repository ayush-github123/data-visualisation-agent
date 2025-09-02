"""Enhanced Explanation Agent with memory capabilities."""

import pandas as pd
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import EXPLAINER_MODEL, EXPLAINER_TEMPERATURE


class ExplainerAgent:
    def __init__(self, memory_manager=None):
        self.llm = ChatGoogleGenerativeAI(
            model=EXPLAINER_MODEL, 
            temperature=EXPLAINER_TEMPERATURE
        )
        self.parser = StrOutputParser()
        self.memory_manager = memory_manager
        
        # Enhanced prompt with memory context
        self.prompt = PromptTemplate.from_template("""
You are an AI data analyst with memory of previous conversations and analyses.

CURRENT ANALYSIS:
User Query: {user_query}
Analysis Type: {analysis_type}
Output Type: {output_type}
Data Sample: {data_sample}

MEMORY CONTEXT:
Chat History: {chat_context}
Previous Analyses: {previous_analyses}
Dataset Context: {dataset_context}

EXPLANATION GUIDELINES:
1. **Build on Previous Context**: Reference previous analyses when relevant
2. **Answer Progression**: If this is a follow-up question, connect it to previous insights
3. **Comparative Insights**: Compare current results with previous findings when applicable
4. **Learning Enhancement**: Use memory to provide more personalized and contextual explanations

Provide a clear, direct answer to the user's question. Focus on:
1. Direct answer to what they asked
2. Key insights from the analysis
3. Any important patterns or findings
4. Connections to previous analyses (if relevant)
5. Suggested follow-up questions based on memory

Keep it conversational and build upon the ongoing analytical journey.
Avoid technical jargon unless the user has shown familiarity with it in previous interactions.
""")

    def explain_results(self, user_query, task_plan, primary_output_type, df_sample):
        """Generate explanations with memory context."""
        # Get memory context if available
        memory_context = {}
        if self.memory_manager:
            context = self.memory_manager.get_context_for_llm(user_query)
            memory_context = {
                "chat_context": context.get("recent_chat_context", "No previous context"),
                "previous_analyses": json.dumps(context.get("previous_successful_analyses", []), indent=2),
                "dataset_context": json.dumps(context.get("dataset_context", {}), indent=2)
            }
        else:
            memory_context = {
                "chat_context": "No memory available",
                "previous_analyses": "[]",
                "dataset_context": "{}"
            }

        # Create the chain
        chain = (
            RunnablePassthrough() 
            | self.prompt 
            | self.llm 
            | self.parser
        )
        
        explanation = chain.invoke({
            "user_query": user_query,
            "analysis_type": task_plan["task"],
            "output_type": primary_output_type,
            "data_sample": df_sample.head(3).to_string() if isinstance(df_sample, pd.DataFrame) else str(df_sample)[:200],
            **memory_context
        })
        
        # Store the interaction in memory
        if self.memory_manager:
            self.memory_manager.add_user_message(user_query)
            self.memory_manager.add_ai_message(explanation)
        
        return explanation