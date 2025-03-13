"""
MAC-SQL Streamlit App
=====================

This module provides a Streamlit web interface for the MAC-SQL framework.
"""

import os
import pandas as pd
import streamlit as st
from typing import Dict, Any

from mac_sql import MACSQL
from core.chat_manager import DB_CONFIG

# Set page config
st.set_page_config(
    page_title="MAC-SQL | Memory, Attention & Composition for Text-to-SQL",
    page_icon="🧠",
    layout="wide"
)

# Define available models
MODEL_OPTIONS = {
    "Meta Llama 3.3 70B Instruct": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Meta Llama 3.1 70B Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Meta Llama 3.1 8B Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    "Mistral 7B Instruct v0.3": "mistral-7B-Instruct-v0.3",
    "Qwen 1.5 72B Chat": "qwen-1.5-72B-Chat",
    "Qwen2.5 7B Instruct": "qwen2.5-7B-Instruct"
}

def format_sql(sql: str) -> str:
    """Format SQL query for display"""
    return sql.replace("SELECT", "\nSELECT").replace("FROM", "\nFROM").replace("WHERE", "\nWHERE").replace("GROUP BY", "\nGROUP BY").replace("ORDER BY", "\nORDER BY").replace("HAVING", "\nHAVING").replace("JOIN", "\nJOIN")

def initialize_session_state():
    """Initialize session state variables"""
    if "mac_sql" not in st.session_state:
        st.session_state.mac_sql = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_db" not in st.session_state:
        st.session_state.current_db = "postgres"

def create_sidebar() -> Dict[str, Any]:
    """Create sidebar with configuration options"""
    st.sidebar.title("MAC-SQL Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model:",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )
    
    # API key (optional)
    api_key = st.sidebar.text_input(
        "Together API Key (optional):",
        type="password",
        help="Leave blank to use default key"
    )
    
    # Database configuration
    st.sidebar.subheader("Database Configuration")
    
    db_config = DB_CONFIG.copy()
    
    db_config["dbname"] = st.sidebar.text_input("Database Name:", value=DB_CONFIG["dbname"])
    db_config["user"] = st.sidebar.text_input("Username:", value=DB_CONFIG["user"])
    db_config["password"] = st.sidebar.text_input("Password:", type="password", value=DB_CONFIG["password"])
    db_config["host"] = st.sidebar.text_input("Host:", value=DB_CONFIG["host"])
    db_config["port"] = st.sidebar.text_input("Port:", value=DB_CONFIG["port"])
    
    # Initialize or update MAC-SQL instance
    if (st.session_state.mac_sql is None or 
        st.session_state.current_db != db_config["dbname"] or
        st.sidebar.button("Reconnect to Database")):
        
        with st.sidebar.spinner("Connecting to database..."):
            st.session_state.mac_sql = MACSQL(
                model_name=MODEL_OPTIONS[model_name],
                api_key=api_key if api_key else None,
                db_config=db_config
            )
            st.session_state.current_db = db_config["dbname"]
        
        st.sidebar.success(f"Connected to {db_config['dbname']} database!")
        
    # Reset conversation
    if st.sidebar.button("Reset Conversation"):
        st.session_state.chat_history = []
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About MAC-SQL")
    st.sidebar.markdown(
        """
        MAC-SQL is a multi-agent collaborative framework for Text-to-SQL generation.
        
        **Features:**
        - **Memory**: Maintains conversation history
        - **Attention**: Focuses on relevant schema information
        - **Composition**: Multi-step query generation
        
        Built using LangChain and Together AI.
        """
    )
    
    return {
        "model_name": MODEL_OPTIONS[model_name],
        "api_key": api_key if api_key else None,
        "db_config": db_config
    }

def display_chat_history():
    """Display chat history"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sql" in message:
                st.code(format_sql(message["sql"]), language="sql")
                
            if message["role"] == "assistant" and "result" in message:
                if isinstance(message["result"], pd.DataFrame):
                    st.dataframe(message["result"])
                elif message["result"]:
                    st.info(message["result"])

def process_user_query(user_query: str, mac_sql: MACSQL):
    """Process user query and update chat history"""
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Process the query with MAC-SQL
    with st.spinner("Thinking..."):
        result = mac_sql.query(user_query)
    
    # Extract components from result
    sql_query = result["sql_query"]
    query_result = result["query_result"]
    understanding = result.get("understanding", "")
    plan = result.get("plan", "")
    
    # Format assistant response
    response = "Here's the SQL query for your question:"
    
    # Add assistant message to chat history
    assistant_message = {
        "role": "assistant",
        "content": response,
        "sql": sql_query
    }
    
    if isinstance(query_result, pd.DataFrame):
        assistant_message["result"] = query_result
    else:
        assistant_message["result"] = query_result
    
    st.session_state.chat_history.append(assistant_message)
    
    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
        st.code(format_sql(sql_query), language="sql")
        
        if isinstance(query_result, pd.DataFrame):
            st.dataframe(query_result)
        else:
            st.info(query_result)
        
        # Show reasoning process in an expander
        with st.expander("Show reasoning process"):
            st.subheader("Question Understanding")
            st.markdown(understanding)
            
            st.subheader("SQL Planning")
            st.markdown(plan)
            
            st.subheader("Relevant Tables")
            st.markdown(", ".join(result.get("relevant_schema", [])))

def main():
    """Main Streamlit app"""
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar and get configuration
    config = create_sidebar()
    
    # Main content
    st.title("🧠 MAC-SQL")
    st.subheader("Memory, Attention, and Composition for Text-to-SQL")
    
    # Ensure MAC-SQL is initialized
    if st.session_state.mac_sql is None:
        st.warning("Please configure and connect to a database using the sidebar.")
        return
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_query = st.chat_input("Ask a question about your database...")
    if user_query:
        process_user_query(user_query, st.session_state.mac_sql)

if __name__ == "__main__":
    main() 