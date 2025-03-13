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
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_together import Together
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import traceback

from core.config import DB_CONFIG, PROMPTS

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base agent class with common functionality for all agents
    """
    
    def __init__(self, chat_manager):
        """
        Initialize the agent with a chat manager
        
        Args:
            chat_manager: The chat manager to use for communication with the LLM
        """
        self.chat_manager = chat_manager
    
    def _clean_response(self, response: str) -> str:
        """
        Clean the response by removing markdown artifacts and normalizing whitespace
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response
        """
        # Remove code blocks if present
        if "```" in response:
            # Extract content from code blocks
            matches = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", response)
            if matches:
                # Join all code blocks found
                response = "\n\n".join(matches)
        
        # Normalize whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response


class SelectorAgent(BaseAgent):
    """
    Agent responsible for selecting relevant schema and examples
    """
    
    def __init__(self, chat_manager):
        """
        Initialize the selector agent
        
        Args:
            chat_manager: The chat manager to use for communication with the LLM
        """
        super().__init__(chat_manager)
    
    def select_schema(self, query, schema_knowledge):
        """
        Select relevant parts of the schema based on the query
        
        Args:
            query: Natural language query
            schema_knowledge: Full schema information
            
        Returns:
            Selected schema parts relevant to the query
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
            result = self.chat_manager.generate_llm_response(prompt)
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
            selection_result = self.chat_manager.generate_llm_response(examples_prompt)
            
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
    
    def __init__(self, chat_manager):
        """
        Initialize the decomposer agent
        
        Args:
            chat_manager: The chat manager to use for communication with the LLM
        """
        super().__init__(chat_manager)
    
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
            response = self.chat_manager.generate_llm_response(understanding_and_planning_prompt)
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
            understanding = self.chat_manager.generate_llm_response(understanding_prompt)
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
            plan = self.chat_manager.generate_llm_response(planning_prompt)
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
    1. Generating initial SQL queries based on the understanding and plan
    2. Refining SQL queries if errors are encountered
    3. Ensuring the final SQL query is valid and accurate
    """
    
    def __init__(self, chat_manager):
        """
        Initialize the refiner agent
        
        Args:
            chat_manager: The chat manager to use for communication with the LLM
        """
        super().__init__(chat_manager)
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
        
        # Determine if we're working with SQLite
        is_sqlite = True  # Default to SQLite as it's safest for the BIRD benchmark
        if hasattr(self.chat_manager, 'connection'):
            import sqlite3
            is_sqlite = isinstance(self.chat_manager.connection, sqlite3.Connection)
        
        # Adjust prompt based on database type
        db_specific_guidance = """
        For SQLite databases:
        1. DO NOT use 'information_schema' tables - they don't exist in SQLite
        2. Use PRAGMA statements to get schema information
        3. Use single quotes for string literals, not double quotes
        4. Date functions use strftime(), not EXTRACT
        5. Remember that SQLite is case-sensitive for table and column names
        """
        
        if not is_sqlite:
            db_specific_guidance = """
            For PostgreSQL databases:
            1. You can use information_schema tables to get metadata
            2. Use double quotes for identifiers and single quotes for string literals
            3. Date functions use EXTRACT() and ::date casting
            4. Remember that PostgreSQL is case-sensitive unless quoted
            """
        
        sql_generation_prompt = f"""You are an expert SQL query generator. Generate a valid SQL query based on the user question, your understanding, and the query plan.

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
"""
        
        retries = 0
        while retries <= self.max_retries:
            try:
                print(f"[RefinerAgent] Sending SQL generation prompt to LLM ({len(sql_generation_prompt)} chars)")
                response = self.chat_manager.generate_llm_response(sql_generation_prompt)
                print(f"[RefinerAgent] SQL Generation Output: {len(response)} chars")
                
                # Extract just the SQL query and clean it
                sql_query = self._clean_sql_response(response)
                
                # Check for SQL repetition and fix it
                if sql_query:
                    repeated_statements = self._detect_repeated_statements(sql_query)
                    if repeated_statements:
                        # Take only the first statement
                        sql_query = repeated_statements[0]
                
                # If SQL query is empty or obviously not SQL, use a fallback
                if not sql_query or len(sql_query) < 10 or not self._looks_like_sql(sql_query):
                    print(f"[RefinerAgent] Extracted SQL doesn't look valid: {sql_query}")
                    sql_query = f"SELECT * FROM {self._extract_table_name(schema)} LIMIT 5;"
                    notes = "Generated invalid SQL. Using fallback query."
                else:
                    notes = "SQL generated based on understanding and plan."
                
                return {
                    "sql": sql_query,
                    "notes": notes
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
            "sql": f"SELECT * FROM {self._extract_table_name(schema)} LIMIT 5;",
            "notes": f"Error occurred during SQL generation. Using fallback query."
        }
    
    def refine_sql(
        self, 
        query: str, 
        sql_query: str,
        error_message: str
    ) -> str:
        """
        Refine SQL query based on error messages
        
        Args:
            query: The original natural language query
            sql_query: The SQL query to refine
            error_message: Error message from execution
            
        Returns:
            Refined SQL query as a string
        """
        if not error_message:
            return sql_query
        
        print(f"\n[RefinerAgent] Refinement Inputs:")
        print(f"  - Query: {query}")
        print(f"  - Original SQL: {sql_query}")
        print(f"  - Error: {error_message}")

        # Create a prompt for SQL refinement
        refinement_prompt = f"""
        You are an expert SQL debugger. The following SQL query was generated to answer a question but encountered an error during execution.
        
        Question: {query}
        Original SQL: {sql_query}
        Error Message: {error_message}
        
        Please fix the SQL query to address the error and ensure it correctly answers the original question.
        Only return the fixed SQL query without any explanations.
        """
        
        max_retries = self.max_retries
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"[RefinerAgent] Sending refinement prompt to LLM ({len(refinement_prompt)} chars)")
                response = self.chat_manager.generate_llm_response(refinement_prompt)
                print(f"[RefinerAgent] Refinement Output: {len(response)} chars")
                
                # Extract SQL from the response
                refined_sql = self._clean_sql_response(response)
                
                # Validate the refined SQL
                if not self._looks_like_sql(refined_sql):
                    # If it doesn't look like SQL, try to extract it
                    match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
                    if match:
                        refined_sql = match.group(1).strip()
                
                # If still no valid SQL, return the original
                if not refined_sql or not self._looks_like_sql(refined_sql):
                    print(f"Warning: Could not extract valid SQL from refinement response. Using original SQL.")
                    return sql_query
                
                return refined_sql
            except Exception as e:
                retry_count += 1
                print(f"Error in SQL refinement (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    print(f"Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                else:
                    print(f"Max retries reached. Returning original SQL.")
                    return sql_query
        
        # If we get here, return the original SQL
        return sql_query
    
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
        """
        Clean and extract SQL query from the LLM response
        
        Args:
            sql: Raw response from LLM that may contain SQL
            
        Returns:
            Cleaned SQL query
        """
        if not sql:
            return ""
            
        # Remove markdown code blocks if present
        if "```" in sql:
            # Try to extract from markdown code blocks first
            matches = re.findall(r"```(?:sql|postgresql|)?\s*([\s\S]*?)```", sql, re.IGNORECASE)
            if matches:
                # Use the first code block that looks like SQL
                for match in matches:
                    cleaned = match.strip()
                    if self._looks_like_sql(cleaned):
                        return cleaned
        
        # If no code blocks found, try to extract SQL using keyword patterns
        lines = sql.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comment lines
            if not line or line.startswith('--') or line.startswith('#') or line.startswith('/*'):
                continue
                
            # Check if line contains SQL keywords to start capturing
            if re.match(r'^(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', line, re.IGNORECASE):
                in_sql = True
                sql_lines.append(line)
            # If we're already capturing SQL, continue until we hit a line that seems to end it
            elif in_sql:
                # Stop capturing if we hit a line that looks like explanation text
                if re.match(r'^(In this query|This query|Here|The|Note|This|I|As you can see)', line):
                    break
                sql_lines.append(line)
        
        # If we found SQL lines, join them
        if sql_lines:
            cleaned_sql = ' '.join(sql_lines)
            
            # Make sure the query ends with a semicolon
            if not cleaned_sql.rstrip().endswith(';'):
                cleaned_sql += ';'
                
            return cleaned_sql
        
        # If all else fails, try a more aggressive approach to find SQL-like content
        sql_match = re.search(r'(SELECT\s+.+?;)', sql, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
            
        # If we still can't find SQL, return the original text cleaned up
        # but only if it looks like it might be SQL
        cleaned = re.sub(r'[`""]', '"', sql)  # Normalize quotes
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize whitespace
        
        if self._looks_like_sql(cleaned):
            return cleaned
            
        # Nothing worked, return empty string
        return ""
    
    def _looks_like_sql(self, text: str) -> bool:
        """Check if a string looks like a valid SQL query"""
        # Check for common SQL keywords
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT']
        
        # Must have SELECT (for the queries we're generating)
        if 'SELECT' not in text.upper():
            return False
            
        # Must have FROM (for the queries we're generating) 
        if 'FROM' not in text.upper():
            return False
            
        # Check if it contains explanatory language
        non_sql_indicators = ['I would', 'please note', 'this query', 'explanation', 'hope this helps']
        if any(indicator in text.lower() for indicator in non_sql_indicators):
            return False
        
        # Count how many SQL keywords are present
        keyword_count = sum(1 for keyword in sql_keywords if keyword in text.upper())
        
        # A real SQL query should have multiple keywords and appropriate length
        return keyword_count >= 2 and len(text) > 20 and len(text) < 1000 

    def _detect_repeated_statements(self, sql: str) -> List[str]:
        """
        Detect and split repeated SQL statements
        
        Args:
            sql: SQL query that might contain repeated statements
            
        Returns:
            List of unique SQL statements
        """
        # Check if there are multiple statements (separated by semicolons)
        if sql.count(';') > 1:
            # Split by semicolon and filter out empty statements
            statements = [stmt.strip() + ';' for stmt in sql.split(';') if stmt.strip()]
            
            # Deduplicate statements
            unique_statements = []
            for stmt in statements:
                normalized = re.sub(r'\s+', ' ', stmt.lower()).strip()
                if not any(re.sub(r'\s+', ' ', existing.lower()).strip() == normalized for existing in unique_statements):
                    unique_statements.append(stmt)
            
            return unique_statements
        
        # No repetition detected
        return [sql]


class QuestionAgent:
    """Agent for understanding user questions and extracting key components."""
    
    def __init__(self, chat_manager):
        """
        Initialize the QuestionAgent.
        
        Args:
            chat_manager: The chat manager to use for communication with the LLM
        """
        self.chat_manager = chat_manager
    
    def process_question(self, query: str, schema_name: str) -> Dict[str, Any]:
        """
        Process a user question to understand its intent and identify key components.
        
        Args:
            query: The user's natural language query
            schema_name: The database schema context
            
        Returns:
            Dict with understanding of the question
        """
        # Create a prompt for understanding the question
        prompt = f"""
        I need to understand this database question: "{query}"
        
        The question is about the database: {schema_name}
        
        Please analyze the question and identify:
        1. The main intent (e.g., filtering, aggregation, comparison)
        2. The entities or tables that are likely involved
        3. Any specific conditions or filters
        4. Any time-related constraints
        5. The type of information requested (count, list, calculation, etc.)
        
        Return your analysis as a JSON object with these components.
        """
        
        try:
            # Get response from the LLM
            response = self.chat_manager.generate_llm_response(prompt)
            
            # Try to parse the response as JSON
            try:
                import json
                result = json.loads(response)
                return result
            except:
                # If parsing fails, return as text
                return {"understanding": response}
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return {"understanding": "Failed to process question", "error": str(e)}


class SQLGenerator:
    """Agent for generating SQL queries based on user questions."""
    
    def __init__(self, chat_manager):
        """
        Initialize the SQLGenerator.
        
        Args:
            chat_manager: The chat manager to use for communication with the LLM
        """
        self.chat_manager = chat_manager
    
    def generate_sql(self, query: str, schema_name: str) -> str:
        """
        Generate SQL from a natural language query.
        
        Args:
            query: The user's natural language query
            schema_name: The database schema context
            
        Returns:
            SQL query string
        """
        # Get the schema information
        schema_info = self._get_schema_info(schema_name)
        
        # Create a prompt for generating SQL
        prompt = f"""
        You are an expert SQL query generator. Your task is to translate a natural language question into a valid SQL query.
        
        Question: "{query}"
        
        Database context: {schema_name}
        
        Database schema information:
        {schema_info}
        
        Generate a single SQL query that answers this question. The SQL must be compatible with PostgreSQL.
        Follow these rules:
        1. Do not use any fictional tables or columns not mentioned in the schema
        2. ONLY return the SQL query without any explanations, comments, or markdown formatting
        3. Ensure your query is complete with necessary joins, conditions, and ordering
        4. Use explicit JOIN syntax rather than comma-separated tables
        5. Make sure the query ends with a semicolon
        
        SQL:
        """
        
        try:
            # Get response from the LLM
            sql = self.chat_manager.generate_llm_response(prompt)
            return sql
        except Exception as e:
            print(f"Error generating SQL: {str(e)}")
            return ""
    
    def _get_schema_info(self, schema_name: str) -> str:
        """
        Get schema information for the specified database.
        
        Args:
            schema_name: The database schema name
            
        Returns:
            Schema information as a string
        """
        try:
            # First try to get schema from database
            schema_query = f"""
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
            """
            
            result = self.chat_manager.execute_sql_query(schema_query)
            
            if result.get('success', False) and 'rows' in result and 'columns' in result:
                # Format the schema information
                schema_info = []
                current_table = None
                table_columns = []
                
                for row in result['rows']:
                    table_name = row[0]
                    column_name = row[1]
                    data_type = row[2]
                    is_nullable = row[3]
                    
                    if current_table != table_name:
                        if current_table:
                            schema_info.append(f"Table: {current_table}")
                            schema_info.append("Columns:")
                            for col in table_columns:
                                schema_info.append(f"  - {col}")
                            schema_info.append("")
                        
                        current_table = table_name
                        table_columns = []
                    
                    nullable_str = "NULL" if is_nullable == "YES" else "NOT NULL"
                    table_columns.append(f"{column_name} ({data_type}, {nullable_str})")
                
                # Add the last table
                if current_table:
                    schema_info.append(f"Table: {current_table}")
                    schema_info.append("Columns:")
                    for col in table_columns:
                        schema_info.append(f"  - {col}")
                
                return "\n".join(schema_info)
            
            # Fallback: Get table names at minimum
            tables_query = f"""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
            
            tables_result = self.chat_manager.execute_sql_query(tables_query)
            
            if tables_result.get('success', False) and 'rows' in tables_result:
                table_names = [row[0] for row in tables_result['rows']]
                return f"Tables in database: {', '.join(table_names)}"
            
            # Final fallback based on database name
            if schema_name == 'student_club':
                return "Tables: member (member_id, first_name, last_name, position), event (event_id, event_name, type, date), attendance (attendance_id, link_to_member, link_to_event), income (income_id, amount, date_received, link_to_member)"
            elif schema_name == 'formula_1':
                return "Tables: drivers (driverId, forename, surname, nationality, dob), races (raceId, name, year, date), constructors (constructorId, name, nationality), results (resultId, raceId, driverId, constructorId, position, points)"
            else:
                return f"Database: {schema_name} (schema information not available)"
        
        except Exception as e:
            print(f"Error retrieving schema info: {str(e)}")
            return f"Database: {schema_name} (schema information not available due to error: {str(e)})" 