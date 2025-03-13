"""
MAC-SQL Agent Classes
====================

This module defines the three agents in the MAC-SQL framework:
1. Selector Agent: Handles schema and example selection
2. Decomposer Agent: Handles question understanding and SQL planning
3. Refiner Agent: Handles SQL generation and refinement

Based on the official MAC-SQL implementation.
"""

import os
import re
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain_together import Together
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import traceback

from core.config import DB_CONFIG

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all MAC-SQL agents"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """Initialize the agent."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set default API key if none provided
        if api_key is None:
            api_key = "6e4593b7c0e0279476b65f144273d1ee972a47e3eb543c9649b36aaf6c114a82"
        
        self.llm = Together(
            model=model_name,
            temperature=temperature,
            together_api_key=api_key,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.parser = StrOutputParser()
        self.name = "BaseAgent"
    
    def _clean_response(self, response: str) -> str:
        """Clean up LLM response (remove markdown, etc.)"""
        # Remove markdown code blocks
        response = re.sub(r"```sql|```", "", response)
        # Remove leading/trailing whitespace
        response = response.strip()
        return response


class SelectorAgent(BaseAgent):
    """Agent responsible for selecting relevant schema based on the user query."""
    
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        self.name = "Selector"

    def select_schema(self, query, schema_knowledge):
        """
        Select relevant schema based on user query.
        
        Args:
            query: User natural language query
            schema_knowledge: Database schema information
            
        Returns:
            List of relevant schema documents
        """
        print(f"\n[SelectorAgent] Inputs:")
        print(f"  - Query: {query}")
        print(f"  - Schema knowledge length: {len(str(schema_knowledge)) if schema_knowledge else 0} chars")
        
        if not schema_knowledge:
            logger.warning("No schema knowledge available")
            print("[SelectorAgent] Output: No schema knowledge available, returning empty list")
            return []
        
        # If schema_knowledge is a string, convert it to a list of Document objects
        if isinstance(schema_knowledge, str):
            if not schema_knowledge.strip():
                logger.warning("Empty schema knowledge string")
                print("[SelectorAgent] Output: Empty schema knowledge string, returning empty list")
                return []
            
            # Split the schema knowledge string into individual table descriptions
            table_sections = re.split(r'\n(?=Table: )', schema_knowledge.strip())
            
            # Convert each table section to a Document object
            schema_docs = [
                Document(
                    page_content=section,
                    metadata={"table": section.split('\n')[0].replace('Table: ', '')}
                )
                for section in table_sections if section.strip()
            ]
        elif isinstance(schema_knowledge, bool):
            logger.warning("Schema knowledge is a boolean value, expected string or list")
            print("[SelectorAgent] Output: Schema knowledge is a boolean value, returning empty list")
            return []
        else:
            # Assume it's already a list of Documents
            schema_docs = schema_knowledge
        
        if not schema_docs:
            logger.warning("No schema documents created")
            print("[SelectorAgent] Output: No schema documents created, returning empty list")
            return []
            
        prompt = f"""You are a database expert. Your task is to analyze a user query and determine which database tables and their relationships are most relevant to answering the query.

USER QUERY: {query}

DATABASE SCHEMA:
{schema_knowledge}

Select the most relevant tables for answering this query. Consider both direct tables mentioned and tables that would be needed for joins. For each table you select, explain why it's relevant.

IMPORTANT: Your analysis must be comprehensive but focused only on tables needed to answer the query. Rank tables by relevance.

FORMAT YOUR RESPONSE AS:
1. [TABLE_NAME]: [REASON FOR SELECTION]
2. [TABLE_NAME]: [REASON FOR SELECTION]
...

ANALYSIS:
"""
        try:
            print(f"[SelectorAgent] Sending prompt to LLM ({len(prompt)} chars)")
            result = self.llm.invoke(prompt)
            logger.info(f"Schema selection completed: {len(schema_docs)} tables available")
            
            # Parse the response to extract table names
            relevant_tables = []
            for line in result.split('\n'):
                if ':' in line and any(c.isdigit() for c in line.split(':')[0]):
                    table_info = line.split(':')[0].strip()
                    # Extract table name - assumes format like "1. tablename" or "1. [tablename]"
                    table_match = re.search(r'\d+\.\s+\[?([a-zA-Z0-9_]+)\]?', table_info)
                    if table_match:
                        table_name = table_match.group(1).lower()
                        # Find the corresponding schema document
                        for doc in schema_docs:
                            doc_table = doc.metadata['table'].lower()
                            if doc_table == table_name:
                                relevant_tables.append(doc)
                                break
            
            # If no tables were found, return a subset of tables that might be relevant
            if not relevant_tables and schema_docs:
                # Return up to 3 tables as a fallback
                print(f"[SelectorAgent] Output: No specific tables found in response, returning first {min(3, len(schema_docs))} tables as fallback")
                return schema_docs[:min(3, len(schema_docs))]
            
            table_names = [doc.metadata['table'] for doc in relevant_tables]
            print(f"[SelectorAgent] Output: Selected {len(relevant_tables)} tables - {', '.join(table_names)}")
            return relevant_tables
            
        except Exception as e:
            logger.error(f"Error in schema selection: {e}")
            # Return a subset of tables as fallback in case of error
            print(f"[SelectorAgent] Output: Error in schema selection: {e}, returning first {min(3, len(schema_docs))} tables as fallback")
            return schema_docs[:min(3, len(schema_docs))] if schema_docs else []
    
    def select_examples(self, query: str, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select relevant examples that might help with the current query
        
        Args:
            query: The user's natural language query
            examples: List of previous query/SQL pairs
            
        Returns:
            List of relevant examples
        """
        if not examples:
            return []
        
        # Create a prompt for example selection
        examples_prompt = f"""You are an expert SQL developer. Your task is to select which of the following previous query examples are most similar to the new user query.

NEW USER QUERY: {query}

PREVIOUS EXAMPLES:
"""
        # Format the examples for the prompt
        for i, example in enumerate(examples):
            examples_prompt += f"Example {i}:\n"
            examples_prompt += f"Question: {example.get('question', '')}\n"
            examples_prompt += f"SQL: {example.get('sql', '')}\n\n"
        
        examples_prompt += """Select the indices of the 3 most relevant examples (0-indexed).
PROVIDE ONLY THE NUMBERS SEPARATED BY COMMAS, like: 0, 2, 3
"""
        
        # Get the response from the LLM
        try:
            selection_result = self.llm.invoke(examples_prompt)
            
            # Parse the indices from the response
            indices = []
            for match in re.finditer(r"\b(\d+)\b", selection_result):
                idx = int(match.group(1))
                if 0 <= idx < len(examples):
                    indices.append(idx)
            
            # Get the selected examples
            selected_examples = [examples[i] for i in indices[:3]]  # Limit to top 3
            
            return selected_examples
            
        except Exception as e:
            logger.error(f"Error in example selection: {e}")
            # Return first examples as fallback
            return examples[:min(3, len(examples))]


class DecomposerAgent(BaseAgent):
    """
    Agent responsible for understanding the question and planning the query.
    
    The Decomposer agent focuses on:
    1. Understanding what the question is asking for
    2. Planning how to retrieve the requested information
    """
    
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
    
    def understand_and_plan(self, query: str, schema: str) -> Dict[str, str]:
        """
        Understand the question and plan the query in one step
        
        Args:
            query: The user's natural language query
            schema: Relevant schema information
            
        Returns:
            Dictionary containing understanding and query plan
        """
        print(f"\n[DecomposerAgent] Inputs:")
        print(f"  - Query: {query}")
        print(f"  - Schema length: {len(str(schema)) if schema else 0} chars")
        
        understanding_and_planning_prompt = f"""You are an expert in understanding natural language questions and translating them into SQL queries. 
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
        
        try:
            print(f"[DecomposerAgent] Sending prompt to LLM ({len(understanding_and_planning_prompt)} chars)")
            response = self.llm.invoke(understanding_and_planning_prompt)
            print(f"[DecomposerAgent] Output: {len(response)} chars")
            
            # Extract understanding and plan from response
            understanding = ""
            plan = ""
            
            # Simple parsing based on headers
            if "UNDERSTANDING:" in response and "PLAN:" in response:
                parts = response.split("PLAN:")
                understanding = parts[0].replace("UNDERSTANDING:", "").strip()
                plan = parts[1].strip()
            else:
                # Fallback if format is not as expected
                understanding = "Attempting to extract information about " + query
                plan = response
            
            return {
                "understanding": understanding,
                "plan": plan
            }
        except Exception as e:
            logger.error(f"Error in understanding and planning: {e}")
            print(f"[DecomposerAgent] Error: {e}, returning fallback")
            return {
                "understanding": f"Attempting to extract information about {query}",
                "plan": f"Search for data related to '{query}' in the most relevant tables."
            }
    
    def understand_question(self, query: str, schema: str) -> str:
        """
        Understand what the question is asking for
        
        Args:
            query: The user's natural language query
            schema: Relevant schema information
            
        Returns:
            Understanding of what the question is asking for
        """
        print(f"\n[DecomposerAgent] Understanding Inputs:")
        print(f"  - Query: {query}")
        print(f"  - Schema length: {len(str(schema)) if schema else 0} chars")
        
        understanding_prompt = f"""You are an expert in understanding natural language questions and translating them into database concepts. Based on the user question and database schema, provide a clear explanation of what the question is asking for in database terms.

USER QUESTION: {query}

RELEVANT DATABASE SCHEMA:
{schema}

Your task is to:
1. Identify the entities being asked about
2. Identify the attributes or metrics being requested
3. Identify any filters or conditions
4. Explain the query in database terms

YOUR UNDERSTANDING:
"""
        
        try:
            print(f"[DecomposerAgent] Sending understanding prompt to LLM ({len(understanding_prompt)} chars)")
            understanding = self.llm.invoke(understanding_prompt)
            print(f"[DecomposerAgent] Understanding Output: {len(understanding)} chars")
            return understanding
        except Exception as e:
            logger.error(f"Error in question understanding: {e}")
            fallback = f"This question is asking for information about {query}."
            print(f"[DecomposerAgent] Understanding Output Error: {e}, returning fallback")
            return fallback
    
    def plan_query(self, query: str, understanding: str, schema: str) -> str:
        """
        Plan how to construct the SQL query
        
        Args:
            query: The user's natural language query
            understanding: The understanding of what the question is asking for
            schema: Relevant schema information
            
        Returns:
            Step-by-step plan for constructing the SQL query
        """
        print(f"\n[DecomposerAgent] Planning Inputs:")
        print(f"  - Query: {query}")
        print(f"  - Understanding length: {len(understanding)} chars")
        print(f"  - Schema length: {len(str(schema)) if schema else 0} chars")
        
        planning_prompt = f"""You are an expert SQL query planner. Based on the user question and your understanding, create a step-by-step plan for constructing the SQL query.

USER QUESTION: {query}

YOUR UNDERSTANDING: 
{understanding}

RELEVANT DATABASE SCHEMA:
{schema}

Create a detailed step-by-step plan for constructing the SQL query, including:
1. Which tables to use and how to join them
2. What columns to select
3. What conditions to apply
4. Any necessary grouping, ordering, or aggregations

YOUR SQL QUERY PLAN:
"""
        
        try:
            print(f"[DecomposerAgent] Sending planning prompt to LLM ({len(planning_prompt)} chars)")
            plan = self.llm.invoke(planning_prompt)
            print(f"[DecomposerAgent] Planning Output: {len(plan)} chars")
            return plan
        except Exception as e:
            logger.error(f"Error in query planning: {e}")
            fallback = f"Plan to search for data related to {query} in the most relevant tables."
            print(f"[DecomposerAgent] Planning Output Error: {e}, returning fallback")
            return fallback


class RefinerAgent(BaseAgent):
    """
    Agent responsible for generating and refining SQL queries.
    
    The Refiner agent focuses on:
    1. Generating SQL based on the understanding and plan
    2. Refining SQL when errors occur
    3. Validating the SQL before execution
    """
    
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)
        self.max_retries = 3  # Maximum number of retries for rate limiting
    
    def generate_sql(
        self, 
        query: str, 
        schema: str,
        understanding_and_plan: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Generate a SQL query based on the understanding and plan
        
        Args:
            query: The user's natural language query
            schema: Relevant schema information
            understanding_and_plan: Dictionary containing understanding and plan
            
        Returns:
            Dictionary containing the generated SQL and any notes
        """
        understanding = understanding_and_plan.get("understanding", "")
        plan = understanding_and_plan.get("plan", "")
        
        print(f"\n[RefinerAgent] Inputs:")
        print(f"  - Query: {query}")
        print(f"  - Schema length: {len(str(schema)) if schema else 0} chars")
        print(f"  - Understanding length: {len(understanding)} chars")
        print(f"  - Plan length: {len(plan)} chars")
        
        sql_generation_prompt = f"""You are an expert SQL query generator. Generate a valid SQL query based on the user question, your understanding, and the query plan.

USER QUESTION: {query}

UNDERSTANDING: 
{understanding}

QUERY PLAN:
{plan}

RELEVANT DATABASE SCHEMA:
{schema}

Generate a valid, executable SQL query that correctly answers the user's question. Follow these guidelines:
1. Use only tables and columns that exist in the schema
2. Use proper joins based on the foreign key relationships
3. Handle any necessary filtering, grouping, or aggregation
4. Ensure the query is syntactically correct

YOUR SQL QUERY (PROVIDE ONLY THE EXECUTABLE SQL WITHOUT ANY EXPLANATION OR MARKDOWN):
"""
        
        retries = 0
        while retries <= self.max_retries:
            try:
                print(f"[RefinerAgent] Sending SQL generation prompt to LLM ({len(sql_generation_prompt)} chars)")
                response = self.llm.invoke(sql_generation_prompt)
                print(f"[RefinerAgent] SQL Generation Output: {len(response)} chars")
                
                # Extract just the SQL query, removing any markdown code blocks
                sql_query = response.strip()
                sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.MULTILINE)
                sql_query = re.sub(r'\s*```$', '', sql_query, flags=re.MULTILINE)
                sql_query = sql_query.strip()
                
                # Clean up common SQL issues
                sql_query = self._clean_sql(sql_query)
                
                return {
                    "sql": sql_query,
                    "notes": "SQL generated based on understanding and plan."
                }
            except Exception as e:
                error_str = str(e).lower()
                retries += 1
                
                # Check if it's a rate limit error
                if any(phrase in error_str for phrase in ['rate limit', 'too many requests', 'server overloaded', '429', '503', 'timeout']):
                    wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8 seconds
                    print(f"[RefinerAgent] Rate limit error, retrying in {wait_time} seconds... ({retries}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Not a rate limit error, break the loop
                    logger.error(f"Error in SQL generation: {e}")
                    print(f"[RefinerAgent] SQL Generation Error: {e}, returning fallback")
                    break
        
        # If we've exhausted retries or encountered a non-rate-limit error
        return {
            "sql": f"SELECT * FROM {self._extract_table_name(schema)} LIMIT 5",
            "notes": f"Error occurred during SQL generation. Using fallback query."
        }
    
    def refine_sql(
        self, 
        query: str, 
        schema: str,
        understanding_and_plan: Dict[str, str],
        sql_query: str,
        error_message: str
    ) -> Dict[str, str]:
        """
        Refine a SQL query based on error feedback
        
        Args:
            query: The user's natural language query
            schema: Relevant schema information
            understanding_and_plan: Dictionary containing understanding and plan
            sql_query: The original SQL query that failed
            error_message: The error message from the database
            
        Returns:
            Dictionary containing the refined SQL and notes
        """
        understanding = understanding_and_plan.get("understanding", "")
        plan = understanding_and_plan.get("plan", "")
        
        print(f"\n[RefinerAgent] Refinement Inputs:")
        print(f"  - Query: {query}")
        print(f"  - Original SQL: {sql_query}")
        print(f"  - Error: {error_message}")
        
        # Check if transaction is aborted
        if "current transaction is aborted" in error_message.lower():
            print("[RefinerAgent] Transaction is aborted. Suggesting ROLLBACK before retrying.")
            return {
                "sql": "ROLLBACK; " + sql_query,
                "notes": "Added ROLLBACK to reset aborted transaction."
            }
        
        refinement_prompt = f"""You are an expert SQL query debugger. The SQL query below failed with an error. Fix the query to make it work correctly.

USER QUESTION: {query}

UNDERSTANDING: 
{understanding}

QUERY PLAN:
{plan}

RELEVANT DATABASE SCHEMA:
{schema}

ORIGINAL SQL QUERY:
{sql_query}

ERROR MESSAGE:
{error_message}

Analyze the error and fix the SQL query. Common issues include:
1. Incorrect table or column names
2. Missing or incorrect joins
3. Syntax errors
4. Type mismatches
5. Missing GROUP BY clauses for aggregations

FIXED SQL QUERY (PROVIDE ONLY THE EXECUTABLE SQL WITHOUT ANY EXPLANATION OR MARKDOWN):
"""
        
        retries = 0
        while retries <= self.max_retries:
            try:
                print(f"[RefinerAgent] Sending refinement prompt to LLM ({len(refinement_prompt)} chars)")
                response = self.llm.invoke(refinement_prompt)
                print(f"[RefinerAgent] Refinement Output: {len(response)} chars")
                
                # Extract just the SQL query, removing any markdown code blocks
                refined_sql = response.strip()
                refined_sql = re.sub(r'^```sql\s*', '', refined_sql, flags=re.MULTILINE)
                refined_sql = re.sub(r'\s*```$', '', refined_sql, flags=re.MULTILINE)
                refined_sql = refined_sql.strip()
                
                # Clean up common SQL issues
                refined_sql = self._clean_sql(refined_sql)
                
                return {
                    "sql": refined_sql,
                    "notes": f"SQL refined to fix error: {error_message}"
                }
            except Exception as e:
                error_str = str(e).lower()
                retries += 1
                
                # Check if it's a rate limit error
                if any(phrase in error_str for phrase in ['rate limit', 'too many requests', 'server overloaded', '429', '503', 'timeout']):
                    wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8 seconds
                    print(f"[RefinerAgent] Rate limit error, retrying in {wait_time} seconds... ({retries}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Not a rate limit error, break the loop
                    logger.error(f"Error in SQL refinement: {e}")
                    print(f"[RefinerAgent] SQL Refinement Error: {e}, returning original query")
                    break
        
        # If we've exhausted retries or encountered a non-rate-limit error
        return {
            "sql": sql_query,
            "notes": f"Error occurred during SQL refinement. Keeping original query."
        }
    
    def _extract_table_name(self, schema: str) -> str:
        """Extract the first table name from the schema"""
        if not schema:
            return "unknown_table"
        
        # Try to extract table name from schema
        table_match = re.search(r'Table: ([a-zA-Z0-9_]+)', schema)
        if table_match:
            return table_match.group(1)
        
        return "unknown_table"
    
    def _clean_sql(self, sql: str) -> str:
        """Clean up common SQL issues"""
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        
        # Ensure semicolon at the end
        if not sql.strip().endswith(';'):
            sql = sql.strip() + ';'
        
        # Fix common date syntax issues for PostgreSQL
        
        # Replace EXTRACT issues with LIKE for PostgreSQL compatibility
        if 'EXTRACT(YEAR FROM' in sql:
            # Replace EXTRACT(YEAR FROM field) = year with field LIKE 'year-%'
            sql = re.sub(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+([a-zA-Z0-9_]+)\s*\)\s*=\s*(\d{4})", 
                         r"\1 LIKE '\2-%'", sql)
        
        # Replace SQLite's strftime with PostgreSQL's date parts
        if 'STRFTIME' in sql.upper():
            sql = re.sub(r"STRFTIME\s*\(\s*'%Y'\s*,\s*([a-zA-Z0-9_]+)\s*\)\s*=\s*'(\d{4})'", 
                        r"\1 LIKE '\2-%'", sql)
        
        # Fix LIKE patterns for date matching
        sql = re.sub(r"date_received LIKE '(\d{4})-%'", r"date_received LIKE '\1-%'", sql)
        sql = re.sub(r"join_date LIKE '(\d{4})-%'", r"join_date LIKE '\1-%'", sql)
        sql = re.sub(r"donation_date LIKE '(\d{4})-%'", r"donation_date LIKE '\1-%'", sql)
        
        # Fix casting issues when using date functions
        if 'EXTRACT' in sql.upper() and '::date' not in sql:
            sql = re.sub(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+([a-zA-Z0-9_]+)\s*\)", 
                       r"EXTRACT(YEAR FROM \1::date)", sql)
        
        # Fix date range comparison
        sql = re.sub(r"([a-zA-Z0-9_]+)\s+>=\s+'(\d{4})-(\d{2})-(\d{2})'\s+AND\s+([a-zA-Z0-9_]+)\s+<\s+'(\d{4})-(\d{2})-(\d{2})'", 
                   r"\1 LIKE '\2-%'", sql)
        
        # Remove extra whitespace 
        sql = re.sub(r'\s+', ' ', sql)
        sql = re.sub(r'\s*;\s*', ';', sql)
        
        return sql
    
    def _clean_sql_response(self, sql: str) -> str:
        """Clean up the SQL response to extract just the SQL query"""
        # Remove markdown code blocks if present
        sql_query = re.search(r'```(?:sql)?\s*(.*?)\s*```', sql, re.DOTALL)
        if sql_query:
            return sql_query.group(1).strip()
        
        # If no code blocks, try to extract SELECT statement
        sql_query = re.search(r'(SELECT.*?);', sql, re.DOTALL | re.IGNORECASE)
        if sql_query:
            return sql_query.group(1).strip() + ";"
        
        # If still no clear SQL, try a more general regex for SQL statements
        sql_query = re.search(r'(?:SELECT|WITH|CREATE|INSERT|UPDATE|DELETE).*?;', sql, re.DOTALL | re.IGNORECASE)
        if sql_query:
            return sql_query.group(0).strip()
        
        # If all else fails, return the original response
        return sql.strip() 