"""
Configuration settings for MAC-SQL
"""

import os
import json
from typing import Dict, Any, Optional
import dotenv

# Load environment variables from .env file if present
dotenv.load_dotenv()

# PostgreSQL connection configuration 
DB_CONFIG = {
    "dbname": os.environ.get("POSTGRES_DB", "postgres"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}

# Model configuration
MODEL_CONFIG = {
    "default_model": os.environ.get("MACSQL_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "temperature": float(os.environ.get("MACSQL_TEMPERATURE", "0.0")),
    "max_tokens": int(os.environ.get("MACSQL_MAX_TOKENS", "1024")),
}

# Paths
BIRD_DATABASE = os.environ.get("BIRD_DATABASE", "minidev/BIRD/BIRD.db")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# LLM API configuration
def get_api_key() -> Optional[str]:
    """
    Get API key with proper priority order:
    1. Environment variable TOGETHER_API_KEY
    2. Config file (config.json)
    3. Return None
    """
    # Check environment variable first
    api_key = os.environ.get("TOGETHER_API_KEY")
    if api_key:
        return api_key
    
    # Try to load from config file
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get("together_api_key")
    except Exception:
        pass
    
    return None

# Define prompt templates
PROMPTS = {
    "selector": {
        "schema_selection": """You are a database expert. Your task is to analyze a user query and determine which database tables and their relationships are most relevant to answering the query.

USER QUERY: {query}

DATABASE SCHEMA:
{schema}

Select the most relevant tables for answering this query. Consider both direct tables mentioned and tables that would be needed for joins. For each table you select, explain why it's relevant.

IMPORTANT: Your analysis must be comprehensive but focused only on tables needed to answer the query. Rank tables by relevance.

FORMAT YOUR RESPONSE AS:
1. [TABLE_NAME]: [REASON FOR SELECTION]
2. [TABLE_NAME]: [REASON FOR SELECTION]
...

ANALYSIS:
""",

        "example_selection": """You are an expert SQL developer. Your task is to select which of the following previous query examples are most similar to the new user query.

NEW USER QUERY: {query}

PREVIOUS EXAMPLES:
{examples}

Select the indices of the 3 most relevant examples (0-indexed).
PROVIDE ONLY THE NUMBERS SEPARATED BY COMMAS, like: 0, 2, 3
"""
    },
    
    "decomposer": {
        "understand_and_plan": """You are an expert in understanding natural language questions and translating them into SQL queries. 
Given a user question and database schema, provide both an understanding of what is being asked and a plan for constructing the SQL query.

USER QUESTION: {query}

RELEVANT DATABASE SCHEMA:
{schema}

Please provide TWO things:
1. UNDERSTANDING: A clear explanation of what the question is asking for in database terms.
2. PLAN: A detailed step-by-step plan for constructing the SQL query, including tables to use, columns to select, conditions, joins, and any necessary aggregations.

Your response should be structured like this:
UNDERSTANDING:
[Your understanding here]

PLAN:
[Your step-by-step plan here]
"""
    },
    
    "refiner": {
        "generate_sql": """You are an expert SQL query generator. Generate a valid SQL query based on the user question, your understanding, and the query plan.

USER QUESTION: {query}

UNDERSTANDING: 
{understanding}

QUERY PLAN:
{plan}

RELEVANT DATABASE SCHEMA:
{schema}

{db_specific_guidance}

EXTREMELY IMPORTANT: CAREFULLY EXAMINE the schema above to identify the EXACT table and column names needed. Do NOT reference columns that don't exist in the schema. Double-check that every column and table name you use actually exists.

Generate a valid, executable SQL query that correctly answers the user's question. Follow these strict guidelines:
1. Use ONLY tables and columns that EXIST in the schema provided - verify each one
2. Use proper joins based on the foreign key relationships shown in the schema
3. Handle any necessary filtering, grouping, or aggregation
4. Double-check column names for spelling and case sensitivity
5. For tables containing financial data, look for columns named 'amount', 'cost', 'budget', 'spent', or similar
6. Return exactly ONE SQL query - do not repeat the same query multiple times

IMPORTANT: PROVIDE ONLY THE EXECUTABLE SQL WITHOUT ANY MARKDOWN FORMATTING, EXPLANATION, OR TEXT. DO NOT INCLUDE ANY ``` BACKTICKS OR COMMENTS IN YOUR RESPONSE, JUST THE RAW SQL QUERY. Your entire response should be valid SQL that can be executed directly.

YOUR SQL QUERY:
""",

        "refine_sql": """
You are an expert SQL debugger. The following SQL query was generated to answer a question but encountered an error during execution.

Question: {query}
Original SQL: {sql_query}
Error Message: {error_message}

Please fix the SQL query to address the error and ensure it correctly answers the original question.
Only return the fixed SQL query without any explanations.
"""
    }
}

# Database-specific guidance
DB_GUIDANCE = {
    "sqlite": """
For SQLite databases:
1. DO NOT use 'information_schema' tables - they don't exist in SQLite
2. Use PRAGMA statements to get schema information
3. Use single quotes for string literals, not double quotes
4. Date functions use strftime(), not EXTRACT
5. Remember that SQLite is case-sensitive for table and column names
""",

    "postgresql": """
For PostgreSQL databases:
1. You can use information_schema tables to get metadata
2. Use double quotes for identifiers and single quotes for string literals
3. Date functions use EXTRACT() and ::date casting
4. Remember that PostgreSQL is case-sensitive unless quoted
"""
} 