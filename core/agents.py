"""
MAC-SQL Agent Classes
====================

This module defines the three agents in the MAC-SQL framework:
1. Selector Agent: Handles schema and example selection
2. Decomposer Agent: Handles question understanding and SQL planning
3. Refiner Agent: Handles SQL generation and refinement

Adapted from the original MAC-SQL implementation but modified for PostgreSQL.
"""

import os
import re
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import traceback
import psycopg2
from copy import deepcopy

logger = logging.getLogger(__name__)

# Constants for agent communication
SYSTEM_NAME = "System"
SELECTOR_NAME = "SelectorAgent"
DECOMPOSER_NAME = "DecomposerAgent"
REFINER_NAME = "RefinerAgent"
MAX_ROUNDS = 3


class BaseAgent:
    """Base agent class with common functionality"""
    
    def __init__(self, chat_manager):
        """
        Initialize the agent with a chat manager
        
        Args:
            chat_manager: The chat manager for LLM communication
        """
        self.chat_manager = chat_manager
        self._message = {}
    
    def talk(self, message: dict):
        """
        Process a message - to be implemented by subclasses
        
        Args:
            message: Message to process
        """
        pass


class SelectorAgent(BaseAgent):
    """
    Agent responsible for selecting relevant schema elements
    
    This agent analyzes the database schema and selects the relevant 
    tables and columns based on the user query.
    """
    
    name = SELECTOR_NAME
    description = "Get database description and extract relevant tables & columns"
    
    def __init__(self, chat_manager):
        """
        Initialize the selector agent
        
        Args:
            chat_manager: The chat manager to use
        """
        super().__init__(chat_manager)
        self.schema_cache = {}
        self.db_schema_info = {}
    
    def select_schema(self, query, schema_knowledge):
        """
        Select relevant parts of the schema based on the query
        
        Args:
            query: Natural language query
            schema_knowledge: Full schema information
            
        Returns:
            Selected schema parts relevant to the query
        """
        print(f"\n[SelectorAgent] Analyzing schema for query: {query}")
        
        if not schema_knowledge:
            logger.warning("No schema knowledge available")
            fallback_schema = self.get_detailed_schema(self.chat_manager.db_name)
            if fallback_schema:
                return self._extract_tables_as_dict(fallback_schema)
            return {}
        
        # Extract all tables and their columns from the schema
        tables_info = self._parse_schema_knowledge(schema_knowledge)
        
        if not tables_info:
            print("[SelectorAgent] No tables found in schema, checking fallbacks")
            db_name = self.chat_manager.db_name.lower() if self.chat_manager.db_name else None
            if db_name:
                fallback_schema = self.get_detailed_schema(db_name)
                if fallback_schema:
                    return self._extract_tables_as_dict(fallback_schema)
            return {}
        
        # For small schemas (few tables or columns), include everything
        total_tables = len(tables_info)
        total_columns = sum(len(columns) for columns in tables_info.values())
        
        print(f"[SelectorAgent] Schema stats: {total_tables} tables, {total_columns} total columns")
        
        if total_tables <= 3 or total_columns <= 30:
            # For small schemas, include all tables and columns
            selected_schema = {table: 'keep_all' for table in tables_info.keys()}
            print(f"[SelectorAgent] Small schema detected, including all {total_tables} tables")
            return selected_schema
        
        # For larger schemas, use the LLM to select relevant parts
        return self._select_relevant_tables(query, tables_info, schema_knowledge)
    
    def _is_need_prune(self, table_count, column_count):
        """
        Determine if schema pruning is needed based on size
        
        Args:
            table_count: Number of tables
            column_count: Number of columns
            
        Returns:
            True if pruning is needed, False otherwise
        """
        if table_count <= 3 and column_count <= 30:
            return False
        return True
    
    def _parse_schema_knowledge(self, schema_knowledge):
        """
        Parse schema knowledge to extract tables and their columns
        
        Args:
            schema_knowledge: Schema string
            
        Returns:
            Dictionary of table names to column lists
        """
        tables_info = {}
        current_table = None
        
        # Process the schema line by line
        lines = schema_knowledge.split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract table name
            if line.startswith('Table:'):
                current_table = line.replace('Table:', '').strip()
                tables_info[current_table] = []
            
            # Extract column information for the current table
            elif current_table and line.startswith('-') and ':' not in line:
                # Line format: "  - column_name (data_type, nullable)"
                # Extract column name, which is the first part before parentheses
                column_match = re.search(r'\s*-\s*([^\s(]+)', line)
                if column_match:
                    column_name = column_match.group(1).strip()
                    tables_info[current_table].append(column_name)
        
        print(f"[SelectorAgent] Parsed schema: {len(tables_info)} tables with {sum(len(cols) for cols in tables_info.values())} total columns")
        return tables_info
    
    def _select_relevant_tables(self, query, tables_info, schema_knowledge):
        """
        Select relevant tables and columns based on the query using LLM
        
        Args:
            query: User query
            tables_info: Dictionary of table names to column lists
            schema_knowledge: Full schema string
            
        Returns:
            Dictionary with selected tables and columns
        """
        try:
            # Prepare the prompt for the LLM to analyze schema relevance
            prompt = f"""
You are a database expert analyzing a user question and a database schema to identify which tables and columns are most relevant.

USER QUESTION: {query}

DATABASE SCHEMA:
{schema_knowledge[:3000]}  # Limit schema size to avoid token limits

Based on the user question, determine which tables and columns are relevant. For each table, decide:
1. If it's highly relevant with 10 or fewer columns, mark it as "keep_all"
2. If it's relevant but has more than 10 columns, list only the most relevant columns (up to 6)
3. If it's completely irrelevant, mark it as "drop_all"

Ensure you include at least 3 tables in your selection unless the schema has fewer tables.

Format your response as a JSON object where keys are table names and values are either "keep_all", "drop_all", or a list of column names. Example:
{{
  "employees": "keep_all",
  "departments": ["dept_id", "dept_name", "manager_id"],
  "irrelevant_table": "drop_all"
}}

Return only the JSON object, no additional text.
"""
            
            # Get response from LLM
            llm_response = self.chat_manager.generate_llm_response(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    selected_schema = json.loads(json_str)
                    
                    # Ensure we have at least some tables
                    if not selected_schema or all(value == "drop_all" for value in selected_schema.values()):
                        # Fallback: include all tables
                        selected_schema = {table: "keep_all" for table in tables_info.keys()}
                    
                    print(f"[SelectorAgent] LLM selected {len(selected_schema)} relevant tables")
                    return selected_schema
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from LLM response")
            
            # Fallback: if we couldn't get valid JSON, include all tables
            print("[SelectorAgent] Could not parse LLM response as valid JSON, using all tables")
            return {table: "keep_all" for table in tables_info.keys()}
            
        except Exception as e:
            logger.error(f"Error in table selection: {str(e)}")
            # Fallback: include all tables
            return {table: "keep_all" for table in tables_info.keys()}
    
    def _extract_tables_as_dict(self, schema):
        """
        Extract tables from schema and return as a dict mapping to 'keep_all'
        
        Args:
            schema: Schema string
            
        Returns:
            Dictionary mapping table names to 'keep_all'
        """
        tables = []
        lines = schema.split('\n')
        for line in lines:
            if line.strip().startswith('Table:'):
                table_name = line.replace('Table:', '').strip()
                tables.append(table_name)
        return {table: 'keep_all' for table in tables}
        
    def get_detailed_schema(self, db_name):
        """
        Provide a more detailed schema description for specific databases
        
        Args:
            db_name: Name of the database
            
        Returns:
            Detailed schema description if available, None otherwise
        """
        if db_name is None:
            return None
            
        db_name_lower = db_name.lower()
        
        # BIRD benchmark database schema templates
        if 'student_club' in db_name_lower:
            return """
Table: member
Columns:
  - member_id (int, NOT NULL, PRIMARY KEY)
  - first_name (varchar, NULL)
  - last_name (varchar, NULL)
  - position (varchar, NULL)
  - email (varchar, NULL)
  - phone (varchar, NULL)

Table: event
Columns:
  - event_id (int, NOT NULL, PRIMARY KEY)
  - event_name (varchar, NULL)
  - type (varchar, NULL)
  - date (date, NULL)
  - location (varchar, NULL)

Table: attendance
Columns:
  - attendance_id (int, NOT NULL, PRIMARY KEY)
  - link_to_member (int, NULL)
  - link_to_event (int, NULL)
Foreign Keys:
  - link_to_member -> member.member_id
  - link_to_event -> event.event_id

Table: income
Columns:
  - income_id (int, NOT NULL, PRIMARY KEY)
  - amount (decimal, NULL)
  - date_received (date, NULL)
  - link_to_member (int, NULL)
Foreign Keys:
  - link_to_member -> member.member_id

Table: budget
Columns:
  - budget_id (int, NOT NULL, PRIMARY KEY)
  - category (varchar, NULL)
  - amount (decimal, NULL)
  - link_to_event (int, NULL)
Foreign Keys:
  - link_to_event -> event.event_id
"""
        elif 'formula_1' in db_name_lower:
            return """
Table: drivers
Columns:
  - driverId (int, NOT NULL, PRIMARY KEY)
  - driverRef (varchar, NULL)
  - number (int, NULL)
  - code (varchar, NULL)
  - forename (varchar, NULL)
  - surname (varchar, NULL)
  - dob (date, NULL)
  - nationality (varchar, NULL)
  - url (varchar, NULL)

Table: races
Columns:
  - raceId (int, NOT NULL, PRIMARY KEY)
  - year (int, NULL)
  - round (int, NULL)
  - circuitId (int, NULL)
  - name (varchar, NULL)
  - date (date, NULL)
  - time (time, NULL)
  - url (varchar, NULL)
Foreign Keys:
  - circuitId -> circuits.circuitId

Table: constructors
Columns:
  - constructorId (int, NOT NULL, PRIMARY KEY)
  - constructorRef (varchar, NULL)
  - name (varchar, NULL)
  - nationality (varchar, NULL)
  - url (varchar, NULL)

Table: results
Columns:
  - resultId (int, NOT NULL, PRIMARY KEY)
  - raceId (int, NULL)
  - driverId (int, NULL)
  - constructorId (int, NULL)
  - number (int, NULL)
  - grid (int, NULL)
  - position (int, NULL)
  - positionText (varchar, NULL)
  - positionOrder (int, NULL)
  - points (float, NULL)
  - laps (int, NULL)
  - time (varchar, NULL)
  - milliseconds (int, NULL)
  - fastestLap (int, NULL)
  - rank (int, NULL)
  - fastestLapTime (varchar, NULL)
  - fastestLapSpeed (varchar, NULL)
  - statusId (int, NULL)
Foreign Keys:
  - raceId -> races.raceId
  - driverId -> drivers.driverId
  - constructorId -> constructors.constructorId
  - statusId -> status.statusId

Table: circuits
Columns:
  - circuitId (int, NOT NULL, PRIMARY KEY)
  - circuitRef (varchar, NULL)
  - name (varchar, NULL)
  - location (varchar, NULL)
  - country (varchar, NULL)
  - lat (float, NULL)
  - lng (float, NULL)
  - alt (int, NULL)
  - url (varchar, NULL)

Table: status
Columns:
  - statusId (int, NOT NULL, PRIMARY KEY)
  - status (varchar, NULL)
"""
        elif 'european_football_2' in db_name_lower:
            return """
Table: player
Columns:
  - player_id (int, NOT NULL, PRIMARY KEY)
  - player_name (varchar, NULL)
  - team_id (int, NULL)
  - birthday (date, NULL)
  - height (int, NULL)
  - weight (int, NULL)
Foreign Keys:
  - team_id -> team.team_id

Table: team
Columns:
  - team_id (int, NOT NULL, PRIMARY KEY)
  - team_name (varchar, NULL)
  - league (varchar, NULL)

Table: match
Columns:
  - match_id (int, NOT NULL, PRIMARY KEY)
  - season (varchar, NULL)
  - date (date, NULL)
  - home_team_id (int, NULL)
  - away_team_id (int, NULL)
  - home_goals (int, NULL)
  - away_goals (int, NULL)
Foreign Keys:
  - home_team_id -> team.team_id
  - away_team_id -> team.team_id
"""
        # Add more hardcoded schemas as needed
            
        return None
        
    def talk(self, message: dict):
        """
        Process a message to extract schema information
        
        Args:
            message: The message to process
            
        Returns:
            Processed message with schema information
        """
        if message.get('send_to') != self.name:
            return
            
        self._message = message
        db_id = message.get('db_id')
        ext_sch = message.get('extracted_schema', {})
        query = message.get('query')
        evidence = message.get('evidence')
        
        # Get schema information from database
        schema_info = self.chat_manager.get_schema()
        
        # Check if schema pruning is needed
        tables_info = self._parse_schema_knowledge(schema_info)
        total_tables = len(tables_info)
        total_columns = sum(len(columns) for columns in tables_info.values())
        
        need_prune = self._is_need_prune(total_tables, total_columns)
        
        # Option to bypass selector
        without_selector = message.get('without_selector', False)
        if without_selector:
            need_prune = False
            
        if ext_sch == {} and need_prune:
            try:
                # Prune schema based on query
                raw_extracted_schema_dict = self._select_relevant_tables(query, tables_info, schema_info)
            except Exception as e:
                logger.error(f"Error pruning schema: {e}")
                raw_extracted_schema_dict = {}
                
            # Update message with selected schema
            message['extracted_schema'] = raw_extracted_schema_dict
            message['desc_str'] = schema_info
            message['fk_str'] = ""  # PostgreSQL FK info would be included in schema_info
            message['pruned'] = True
            message['send_to'] = DECOMPOSER_NAME
        else:
            # No pruning needed, use full schema
            message['desc_str'] = schema_info
            message['fk_str'] = ""  # PostgreSQL FK info would be included in schema_info
            message['pruned'] = False
            message['send_to'] = DECOMPOSER_NAME
            
        return message


class DecomposerAgent(BaseAgent):
    """
    Agent responsible for understanding the question and planning the query
    """
    
    name = DECOMPOSER_NAME
    description = "Decompose the question and solve them using CoT"
    
    def __init__(self, chat_manager):
        """
        Initialize the decomposer agent
        
        Args:
            chat_manager: The chat manager to use
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
            print(f"[DecomposerAgent] Processing: {query}")
            response = self.chat_manager.generate_llm_response(understanding_and_planning_prompt)
            
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
            return {
                "understanding": f"Attempting to extract information about {query}",
                "plan": f"Search for data related to '{query}' in the most relevant tables."
            }
    
    def talk(self, message: dict):
        """
        Process a message to create understanding and plan
        
        Args:
            message: The message to process
            
        Returns:
            Processed message with understanding and plan
        """
        if message.get('send_to') != self.name:
            return
            
        self._message = message
        query = message.get('query')
        evidence = message.get('evidence', '')
        schema_info = message.get('desc_str', '')
        fk_info = message.get('fk_str', '')
        
        # Augment query with evidence if provided
        augmented_query = query
        if evidence and evidence not in query:
            augmented_query = f"{query}\nContext: {evidence}"
        
        # Get understanding and plan
        result = self.understand_and_plan(augmented_query, schema_info)
        
        # Generate SQL based on understanding and plan
        try:
            # Create a prompt for SQL generation
            sql_prompt = f"""You are an expert SQL developer using PostgreSQL. 
Generate a valid SQL query that answers this question based on the understanding and plan provided.

QUESTION: {augmented_query}

DATABASE SCHEMA:
{schema_info}

UNDERSTANDING:
{result['understanding']}

PLAN:
{result['plan']}

Generate a valid, executable SQL query for PostgreSQL. Return ONLY the SQL query without any explanations or markdown formatting.
"""
            sql_response = self.chat_manager.generate_llm_response(sql_prompt)
            
            # Clean the response to extract just the SQL
            sql_query = self._extract_sql(sql_response)
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            sql_query = f"-- Error generating SQL: {str(e)}\nSELECT 1;"
        
        message['understanding'] = result.get('understanding', '')
        message['plan'] = result.get('plan', '')
        message['final_sql'] = sql_query
        message['fixed'] = False
        message['send_to'] = REFINER_NAME
        
        return message
    
    def _extract_sql(self, response):
        """
        Extract SQL query from LLM response
        
        Args:
            response: Response from LLM
            
        Returns:
            Extracted SQL query
        """
        if not response:
            return "SELECT 1;"
        
        # First try to extract SQL from code blocks
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Try to extract from any code blocks
        sql_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # If no code blocks, try to extract SQL by looking for common SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        for keyword in sql_keywords:
            sql_match = re.search(f'({keyword}[^;]*;)', response, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
        
        # If all else fails, just return the entire response
        return response.strip()


class RefinerAgent(BaseAgent):
    """
    Agent responsible for refining and fixing SQL queries
    """
    
    name = REFINER_NAME
    description = "Execute SQL and perform validation"
    
    def __init__(self, chat_manager):
        """
        Initialize the refiner agent
        
        Args:
            chat_manager: The chat manager to use
        """
        super().__init__(chat_manager)
    
    def generate_sql(self, query, schema, understanding=None, plan=None):
        """
        Generate SQL from a natural language query
        
        Args:
            query: The natural language query
            schema: The database schema
            understanding: Optional query understanding
            plan: Optional query plan
            
        Returns:
            Generated SQL
        """
        # Format understanding and plan
        understanding_str = understanding if understanding else ''
        plan_str = plan if plan else ''
        
        # Create a prompt for SQL generation
        prompt = f"""
        You are an expert SQL developer. Generate a PostgreSQL query for the following question.
        
        Database Schema:
        {schema}
        
        Question: {query}
        
        Understanding: {understanding_str}
        Plan: {plan_str}
        
        Based on the database schema and the question, generate a valid PostgreSQL SQL query.
        Return only the SQL query, nothing else.
        """
        
        try:
            # Generate SQL using LLM
            sql_result = self.chat_manager.generate_llm_response(prompt)
            
            # Clean up the SQL
            import re
            sql_match = re.search(r'```sql\s*(.*?)\s*```', sql_result, re.DOTALL)
            if sql_match:
                sql = sql_match.group(1).strip()
            else:
                # Try to extract without code blocks
                sql_match = re.search(r'SELECT\s+.*?;', sql_result, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    sql = sql_match.group(0).strip()
                else:
                    # Just use the response as is
                    sql = sql_result.strip()
            
            # Ensure SQL ends with semicolon
            if not sql.strip().endswith(';'):
                sql = sql.strip() + ';'
            
            return sql
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return "SELECT * FROM information_schema.tables LIMIT 5;"
    
    def _is_need_refine(self, exec_result):
        """
        Determine if SQL needs refinement based on execution result
        
        Args:
            exec_result: Execution result
            
        Returns:
            True if refinement is needed, False otherwise
        """
        if exec_result.get('success', False):
            return False
            
        # Check for specific error messages that indicate refinement is needed
        error_message = exec_result.get('error', '')
        
        # Common SQL errors that can be fixed
        fixable_errors = [
            'syntax error',
            'no such table',
            'no such column',
            'ambiguous column name',
            'relation',
            'does not exist',
            'column',
            'not found'
        ]
        
        # Check if any of the fixable errors are in the error message
        for err in fixable_errors:
            if err in error_message.lower():
                return True
                
        # Check if there's a specific error_info structure with more details
        if 'error_info' in exec_result and 'error_details' in exec_result['error_info']:
            details = exec_result['error_info']['error_details']
            if details:
                return True
                
        return False
    
    def refine_sql(self, query, schema, understanding, original_sql, error_message):
        """
        Refine SQL based on error message
        
        Args:
            query: The natural language query
            schema: The database schema
            understanding: Query understanding
            original_sql: The original SQL query
            error_message: The error message from execution
            
        Returns:
            Refined SQL
        """
        # Check if the error_message is a dict with execution results
        execution_results = None
        error_info = None
        error_str = ""
        
        if isinstance(error_message, dict):
            # This is a result object from execute_sql_query
            execution_results = error_message
            if not execution_results.get('success', False):
                error_str = execution_results.get('error', 'Unknown error')
                error_info = execution_results.get('error_info', {})
            else:
                # Execution was successful but returned empty results
                if 'rows' in execution_results and len(execution_results['rows']) == 0:
                    error_str = "Query executed successfully but returned no results."
                    # Add this flag to indicate empty results
                    error_info = {'empty_results': True}
        else:
            # Direct error string
            error_str = str(error_message)
        
        # Log the refinement attempt
        logger.info(f"Refining SQL: {original_sql}")
        logger.info(f"Error: {error_str}")
        
        # For empty results, try to fix common issues before sending to LLM
        if error_info and error_info.get('empty_results'):
            # Try to automatically fix common issues
            
            # 1. Case sensitivity issues (e.g., 'Attended' vs 'attended')
            # Convert string equality comparisons to case-insensitive comparisons
            case_insensitive_sql = self._fix_case_sensitivity(original_sql)
            if case_insensitive_sql != original_sql:
                logger.info(f"Attempting to fix case sensitivity: {case_insensitive_sql}")
                # Try executing the case-insensitive query
                try:
                    test_result = self.chat_manager.execute_sql_query(case_insensitive_sql)
                    if test_result.get('success', False) and test_result.get('rows') and len(test_result['rows']) > 0:
                        logger.info("Case sensitivity fix successful!")
                        return case_insensitive_sql
                except Exception as e:
                    logger.error(f"Error testing case sensitivity fix: {e}")
            
            # 2. Get sample data to help with refinement
            sample_data = self._get_sample_data(original_sql)
            
            # 3. Check for date format issues and other common problems
            date_fixed_sql = self._fix_date_formats(original_sql)
            if date_fixed_sql != original_sql:
                logger.info(f"Attempting to fix date formats: {date_fixed_sql}")
                try:
                    test_result = self.chat_manager.execute_sql_query(date_fixed_sql)
                    if test_result.get('success', False) and test_result.get('rows') and len(test_result['rows']) > 0:
                        logger.info("Date format fix successful!")
                        return date_fixed_sql
                except Exception as e:
                    logger.error(f"Error testing date format fix: {e}")
        
        # For relation does not exist errors, try to find the correct table name
        if error_info and error_info.get('error_details', {}).get('missing_table'):
            missing_table = error_info['error_details']['missing_table']
            similar_tables = error_info['error_details'].get('similar_tables', [])
            
            if similar_tables:
                # Try each similar table
                for similar_table in similar_tables:
                    corrected_sql = original_sql.replace(missing_table, similar_table)
                    logger.info(f"Trying with corrected table name: {corrected_sql}")
                    try:
                        test_result = self.chat_manager.execute_sql_query(corrected_sql)
                        if test_result.get('success', False):
                            logger.info(f"Table name correction successful! Using {similar_table} instead of {missing_table}")
                            return corrected_sql
                    except Exception as e:
                        logger.error(f"Error testing table name correction: {e}")
        
        # If the above automatic fixes didn't work, use the LLM for refinement
        prompt = f"""
        You are an expert SQL developer. You need to fix an SQL query that is not working as expected.
        
        Original Query: {query}
        
        Database Schema:
        {schema}
        
        Current SQL:
        {original_sql}
        
        Error or Issue:
        {error_str}
        """
        
        # Add sample data to the prompt if available
        if 'sample_data' in locals() and sample_data:
            prompt += f"""
            Sample Data from the Database:
            {sample_data}
            """
        
        prompt += """
        Please provide a corrected SQL query that fixes the issue. Consider:
        1. Table and column names might be incorrect
        2. Case sensitivity in string comparisons
        3. Date format issues
        4. Join conditions
        5. Syntax errors
        
        Only return the corrected SQL query, without explanation or additional text.
        """
        
        # Generate the refined SQL
        try:
            response = self.chat_manager.generate_llm_response(prompt)
            refined_sql = self._extract_sql(response)
            
            # Add a semicolon at the end if missing
            if not refined_sql.strip().endswith(';'):
                refined_sql = refined_sql.strip() + ';'
                
            logger.info(f"Refined SQL: {refined_sql}")
            
            # Check if refinement made any changes
            if refined_sql == original_sql:
                logger.warning("Refinement did not change the SQL query")
                
            return refined_sql
        except Exception as e:
            logger.error(f"Error refining SQL: {e}")
            return original_sql
    
    def _fix_case_sensitivity(self, sql):
        """Fix case sensitivity issues in SQL queries"""
        import re
        
        # Find all string literals in single quotes
        pattern = r"'([^']*)'"
        matches = re.finditer(pattern, sql)
        
        # Replace = 'String' with ILIKE 'string'
        new_sql = sql
        for match in matches:
            string_literal = match.group(0)
            if string_literal in new_sql and string_literal != "'%'" and string_literal != "''":
                # Skip obvious patterns that shouldn't be case-insensitive
                if " = " + string_literal in new_sql:
                    new_sql = new_sql.replace(" = " + string_literal, " ILIKE " + string_literal)
        
        return new_sql
    
    def _fix_date_formats(self, sql):
        """Fix date format issues in SQL"""
        import re
        
        # Find date literals like '2023-01-01' or '2023/01/01' or '01/01/2023'
        date_pattern = r"'(\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|\d{2}/\d{2}/\d{4})'"
        matches = re.finditer(date_pattern, sql)
        
        # Replace with date conversion functions
        new_sql = sql
        for match in matches:
            date_literal = match.group(0)
            # If using = with dates, use a more flexible comparison
            if " = " + date_literal in new_sql:
                new_sql = new_sql.replace(
                    " = " + date_literal, 
                    f" = TO_DATE({date_literal}, 'YYYY-MM-DD')"
                )
        
        return new_sql
    
    def _get_sample_data(self, sql):
        """Get sample data from tables mentioned in the SQL to help with refinement"""
        try:
            # Extract table names from SQL
            import re
            table_pattern = r'\bFROM\s+([a-zA-Z0-9_]+)|[Jj][Oo][Ii][Nn]\s+([a-zA-Z0-9_]+)'
            matches = re.finditer(table_pattern, sql)
            
            tables = []
            for match in matches:
                if match.group(1):  # FROM clause
                    tables.append(match.group(1))
                elif match.group(2):  # JOIN clause
                    tables.append(match.group(2))
            
            # Get sample data from each table
            sample_data = ""
            for table in set(tables):
                try:
                    result = self.chat_manager.execute_sql_query(f"SELECT * FROM {table} LIMIT 3")
                    if result.get('success', False) and result.get('rows'):
                        sample_data += f"\nTable: {table}\n"
                        sample_data += "Columns: " + ", ".join(result.get('columns', [])) + "\n"
                        sample_data += "Sample rows:\n"
                        for row in result.get('rows', []):
                            sample_data += str(row) + "\n"
                except Exception as e:
                    logger.error(f"Error getting sample data for {table}: {e}")
            
            return sample_data
            
        except Exception as e:
            logger.error(f"Error analyzing SQL for sample data: {e}")
            return ""
    
    def _extract_sql(self, response):
        """
        Extract SQL query from LLM response
        
        Args:
            response: Response from LLM
            
        Returns:
            Extracted SQL query
        """
        if not response:
            return "SELECT 1;"
        
        # First try to extract SQL from code blocks
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Try to extract from any code blocks
        sql_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # If no code blocks, try to extract SQL by looking for common SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        for keyword in sql_keywords:
            sql_match = re.search(f'({keyword}[^;]*;)', response, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
        
        # If all else fails, just return the entire response
        return response.strip()
    
    def talk(self, message: dict):
        """
        Process a message to refine SQL if needed
        
        Args:
            message: The message to process
            
        Returns:
            Processed message with refined SQL
        """
        if message.get('send_to') != self.name:
            return
            
        self._message = message
        db_id = message.get('db_id')
        old_sql = message.get('final_sql', "")
        query = message.get('query')
        evidence = message.get('evidence', '')
        schema_info = message.get('desc_str', '')
        understanding = message.get('understanding', '')

        # Skip if SQL contains "error"
        if 'error' in old_sql.lower():
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = old_sql
            message['send_to'] = SYSTEM_NAME
            return message
        
        # Execute SQL and check for errors
        try:
            exec_result = self.chat_manager.execute_sql_query(old_sql)
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            exec_result = {
                'success': False,
                'error': str(e)
            }
        
        # Check if refinement is needed
        is_need = self._is_need_refine(exec_result)
        
        if not is_need:
            # SQL execution was successful
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = old_sql
            message['send_to'] = SYSTEM_NAME
        else:
            # Refinement needed
            new_sql = self.refine_sql(query, schema_info, understanding, old_sql, exec_result)
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = new_sql
            message['fixed'] = True
            message['send_to'] = REFINER_NAME  # Send back to refiner for another try
        
        return message


class QuestionAgent(BaseAgent):
    """
    Agent for understanding user questions and extracting key components
    """
    
    def __init__(self, chat_manager):
        """
        Initialize the question agent
        
        Args:
            chat_manager: The chat manager to use
        """
        super().__init__(chat_manager)
    
    def process_question(self, query: str) -> Dict[str, Any]:
        """
        Process a user question to extract key components
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with extracted components
        """
        prompt = f"""Analyze this database question and extract key components:
        
QUESTION: {query}
        
Identify:
        1. The main intent (e.g., filtering, aggregation, comparison)
        2. The entities or tables that are likely involved
        3. Any specific conditions or filters
        4. Any time-related constraints
        5. The type of information requested (count, list, calculation, etc.)
        
        Return your analysis as a JSON object with these components.
        """
        
        try:
            response = self.chat_manager.generate_llm_response(prompt)
            
            # Try to parse as JSON
            try:
                parsed_response = json.loads(response)
                return parsed_response
            except json.JSONDecodeError:
                # If parsing fails, return as text
                return {"understanding": response}
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {"understanding": f"Failed to process question: {str(e)}"}


class SQLGenerator(BaseAgent):
    """
    Agent for directly generating SQL from natural language
    """
    
    def __init__(self, chat_manager):
        """
        Initialize the SQL generator
        
        Args:
            chat_manager: The chat manager to use
        """
        super().__init__(chat_manager)
    
    def generate_sql(self, query: str, schema: str) -> str:
        """
        Generate SQL from natural language query
        
        Args:
            query: Natural language query
            schema: Database schema
            
        Returns:
            Generated SQL query
        """
        prompt = f"""You are an expert SQL developer. Generate a PostgreSQL query for this question:

QUESTION: {query}

DATABASE SCHEMA:
{schema}

Generate ONLY the SQL query without explanations. Ensure it's valid PostgreSQL syntax.
        """
        
        try:
            response = self.chat_manager.generate_llm_response(prompt)
            
            # Extract SQL from response
            sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
            
            sql_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
            
            # If no code blocks, return the whole response
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return f"-- Error: {str(e)}\nSELECT 1;" 