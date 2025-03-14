#!/usr/bin/env python
"""
MAC-SQL: Multi-Agent Collaboration for SQL Generation
=====================================================

A multi-agent collaborative framework for Text-to-SQL generation.

This module provides the main interface for the MAC-SQL framework.
"""

import logging
import os
import json
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Union, Tuple
from tqdm import tqdm
import time
import traceback
import re
import sqlite3

from core.chat_manager import ChatManager
from core.agents import SelectorAgent, DecomposerAgent, RefinerAgent, QuestionAgent, SQLGenerator
from core.config import DB_CONFIG

# Define the BIRD database path
BIRD_DATABASE = "minidev/BIRD/BIRD.db"

logger = logging.getLogger(__name__)

class MACSQL:
    """
    Multi-Agent Collaborative framework for Text-to-SQL
    
    This framework implements a collaborative system of three specialized agents:
    1. Selector: Handles schema and example selection
    2. Decomposer: Handles question understanding and query planning
    3. Refiner: Handles SQL generation and refinement
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the MAC-SQL framework
        
        Args:
            model_name: Name of the LLM to use
            api_key: API key for the LLM service (optional)
            temperature: Temperature for the LLM
            max_tokens: Maximum number of tokens for the LLM
            verbose: Whether to print verbose output
            **kwargs: Additional keyword arguments
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.kwargs = kwargs
        
        # Initialize schema cache with timestamp tracking
        self.schema_cache = {}
        self.cache_timestamp = {}
        self.cache_ttl = 3600  # Default cache TTL (1 hour)
        
        # Initialize chat manager with basic parameters
        self.chat_manager = ChatManager(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Initialize agents with basic parameters
        self.selector_agent = SelectorAgent(self.chat_manager)
        self.decomposer_agent = DecomposerAgent(self.chat_manager)
        self.refiner_agent = RefinerAgent(self.chat_manager)
        self.question_agent = QuestionAgent(self.chat_manager)
        self.sql_generator = SQLGenerator(self.chat_manager)
    
    def add_to_schema_cache(self, db_name: str, schema: str) -> None:
        """
        Add schema to cache with timestamp
        
        Args:
            db_name: Database name
            schema: Schema information
        """
        import time
        self.schema_cache[db_name] = schema
        self.cache_timestamp[db_name] = time.time()
        logger.info(f"Added schema for database '{db_name}' to cache")
    
    def get_from_schema_cache(self, db_name: str) -> Optional[str]:
        """
        Get schema from cache if available and not expired
        
        Args:
            db_name: Database name
            
        Returns:
            Schema information or None if not in cache or expired
        """
        import time
        if db_name in self.schema_cache:
            # Check if cache has expired
            if time.time() - self.cache_timestamp.get(db_name, 0) > self.cache_ttl:
                logger.info(f"Cache for database '{db_name}' has expired")
                return None
            
            logger.info(f"Using cached schema for database '{db_name}'")
            return self.schema_cache[db_name]
        
        return None
    
    def clear_schema_cache(self, db_name: str = None) -> None:
        """
        Clear schema cache for a specific database or all databases
        
        Args:
            db_name: Database name or None to clear all
        """
        if db_name:
            if db_name in self.schema_cache:
                del self.schema_cache[db_name]
                if db_name in self.cache_timestamp:
                    del self.cache_timestamp[db_name]
                logger.info(f"Cleared cache for database '{db_name}'")
        else:
            self.schema_cache.clear()
            self.cache_timestamp.clear()
            logger.info("Cleared all schema cache")
    
    def connect_to_database(self, db_name: str) -> bool:
        """
        Connect to a database
        
        Args:
            db_name: Database name
        
        Returns:
            True if connection was successful
        """
        # Try to connect to the database
        if self.chat_manager.connect_to_database(db_name):
            # If connection was successful, cache the schema
            if hasattr(self.chat_manager, 'schema_knowledge') and self.chat_manager.schema_knowledge:
                self.add_to_schema_cache(db_name, self.chat_manager.schema_knowledge)
            return True
        return False
    
    def process_query(self, query: str, db_context: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
        """
        Process a natural language query to generate and execute SQL
        
        Args:
            query: The natural language query to process
            db_context: Optional database context to use
            
        Returns:
            Tuple of (generated SQL, query results)
        """
        import time
        import pandas as pd
        
        start_time = time.time()
        
        # Connect to database if not already connected or context changed
        if db_context and (not self.chat_manager.connected or db_context != self.chat_manager.db_name):
            print(f"Connecting to database: {db_context}")
            if not self.chat_manager.connect_to_database(db_context):
                print(f"Failed to connect to database: {db_context}")
                return "Failed to connect to database", pd.DataFrame()
        
        # Check if we're connected to a database
        if not self.chat_manager.connected:
            print("Not connected to a database. Please connect to a database first.")
            return "Not connected to a database", pd.DataFrame()
        
        schema_name = db_context or self.chat_manager.db_name
        
        print(f"Using schema context: {schema_name}")
        print("Understanding the question...")
        
        # Generate SQL
        print("Generating SQL...")
        
        # Get appropriate schema for the query
        # For SQLite databases, we need to handle this differently
        is_sqlite = isinstance(self.chat_manager.connection, sqlite3.Connection)
        
        try:
            # Initialize schema knowledge if it's empty
            if not self.chat_manager.schema_knowledge:
                if is_sqlite:
                    self.chat_manager._initialize_sqlite_schema()
                else:
                    self.chat_manager._initialize_schema_knowledge()
            
            # Get schema knowledge from chat manager
            schema = self.chat_manager.schema_knowledge
            
            # If still no schema, create a fallback
            if not schema:
                if is_sqlite:
                    # For SQLite, get table list and basic info
                    cursor = self.chat_manager.connection.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
                    tables = [table[0] for table in cursor.fetchall()]
                    
                    schema_parts = []
                    for table in tables:
                        cursor.execute(f"PRAGMA table_info({table});")
                        columns = cursor.fetchall()
                        
                        col_info = [f"{col[1]} ({col[2]})" for col in columns]
                        schema_parts.append(f"Table: {table}\nColumns: {', '.join(col_info)}")
                    
                    schema = "\n\n".join(schema_parts)
                else:
                    # For PostgreSQL
                    cursor = self.chat_manager.connection.cursor()
                    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
                    tables = [table[0] for table in cursor.fetchall()]
                    
                    schema = f"Tables: {', '.join(tables)}"
        except Exception as e:
            print(f"Error retrieving schema: {e}")
            schema = f"Database: {schema_name}"
        
        # Use the Decomposer agent to understand the question and plan the query
        decomposer = DecomposerAgent(self.chat_manager)
        understanding_and_plan = decomposer.understand_and_plan(query, schema)
        
        # Use the Refiner agent to generate the SQL
        refiner = RefinerAgent(self.chat_manager)
        sql_result = refiner.generate_sql(query, schema, understanding_and_plan)
        
        # Clean up the SQL
        sql_query = self._clean_sql(sql_result.get("sql", ""))
        
        # Execute the SQL query
        print("Executing SQL...")
        error_message = None
        attempt = 1
        max_attempts = 3
        results = pd.DataFrame()
        
        # Multiple attempts to execute the query
        while attempt <= max_attempts:
            print(f"Execution attempt {attempt}...")
            result = self.chat_manager.execute_sql_query(sql_query)
            
            if result.get('success', False):
                # Successful execution
                if 'columns' in result and 'rows' in result:
                    # Convert the results to a pandas DataFrame
                    results = pd.DataFrame(result['rows'], columns=result['columns'])
                    error_message = None
                    break
                else:
                    # No results but no error (e.g., UPDATE statement)
                    results = pd.DataFrame()
                    error_message = None
                    break
            else:
                # Execution failed
                error_message = result.get('error', 'Unknown error')
                print(f"SQL execution error: {error_message}")
                
                # Refine the SQL if there's an error
                if "no such column" in error_message.lower() or "column not found" in error_message.lower() or "no such table" in error_message.lower():
                    print("Refining SQL due to execution error...")
                    sql_query = refiner.refine_sql(query, sql_query, error_message)
                    attempt += 1
                else:
                    # For other types of errors, try to handle them
                    sql_query = refiner.refine_sql(query, sql_query, error_message)
                    attempt += 1
        
        # If all attempts failed, try a fallback query
        if error_message and attempt > max_attempts:
            print("Creating fallback query due to continued errors...")
            fallback_sql = self._generate_fallback_query(query, schema_name, error_message)
            
            # Execute the fallback query
            result = self.chat_manager.execute_sql_query(fallback_sql)
            
            if result.get('success', False):
                sql_query = fallback_sql
                if 'columns' in result and 'rows' in result:
                    results = pd.DataFrame(result['rows'], columns=result['columns'])
                else:
                    results = pd.DataFrame()
            else:
                # Even the fallback failed, return empty results
                print("Returning empty results due to continued errors")
                results = pd.DataFrame()
        
        # Execution time
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f}s")
        
        return sql_query, results

    def _clean_sql(self, sql: str) -> str:
        """
        Clean and validate the SQL query
        
        Args:
            sql: The SQL query to clean
            
        Returns:
            str: The cleaned SQL query
        """
        # Extract just the SQL if it's embedded in a larger text
        if not sql:
            return ""
        
        # Remove markdown code blocks if present
        sql = re.sub(r'```sql|```postgresql|```', '', sql)
        
        # Remove any additional explanation or notes
        sql_lines = []
        in_sql = False
        
        for line in sql.split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # If line starts with SELECT, FROM, WHERE, etc. we're in SQL territory
            if re.match(r'^(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|JOIN|LEFT|RIGHT|INNER|OUTER|FULL|USING|ON)\b', line.upper()):
                in_sql = True
                sql_lines.append(line)
            # If we're already in SQL mode and the line doesn't start with common SQL keywords,
            # but looks like part of a SQL query, include it
            elif in_sql and not line.startswith('--') and not line.startswith('#') and not line.startswith('/*'):
                sql_lines.append(line)
        
        # If no SQL was found using the line-by-line approach, use a regex approach
        if not sql_lines:
            # Try to find a SQL query using regex
            match = re.search(r'(SELECT\s+.+?(?:;|$))', sql, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Join all SQL lines
        cleaned_sql = ' '.join(sql_lines).strip()
        
        # Ensure the SQL ends with a semicolon
        if cleaned_sql and not cleaned_sql.endswith(';'):
            cleaned_sql += ';'
        
        return cleaned_sql

    def _generate_fallback_query(self, query: str, schema_name: str, error_message: str) -> str:
        """
        Generate a simplified fallback query when the main query fails repeatedly
        
        Args:
            query: The original natural language query
            schema_name: The database schema context
            error_message: The error message from the failed attempts
            
        Returns:
            str: A simplified SQL query that is more likely to execute successfully
        """
        # Create a prompt for generating a simplified query
        prompt = f"""
        I need a very simple SQL query for the database '{schema_name}' that captures the essence of this question:
        "{query}"
        
        The previous queries failed with this error: "{error_message}"
        
        Please generate a simplified SQL query that:
        1. Uses only basic SELECT, FROM, WHERE clauses
        2. Avoids complex joins or subqueries
        3. Uses only tables and columns you're certain exist in the schema
        4. Returns at least some relevant information, even if it's not complete
        
        Return ONLY the SQL query without any explanation or markdown.
        """
        
        # Get response from the LLM for a simplified query
        try:
            response = self.chat_manager.generate_llm_response(prompt)
            fallback_sql = self._clean_sql(response)
            print(f"Generated fallback SQL: {fallback_sql}")
            return fallback_sql
        except Exception as e:
            print(f"Error generating fallback query: {str(e)}")
            # If all else fails, return a very basic query to at least get something from a main table
            return f"SELECT * FROM (SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1) AS t JOIN (SELECT * FROM {self._get_main_table_name(schema_name)} LIMIT 5) AS main_data;"

    def _get_main_table_name(self, schema_name: str) -> str:
        """
        Get a main table name from the schema as a last resort fallback
        
        Args:
            schema_name: The database schema context
            
        Returns:
            str: A table name that likely exists in the schema
        """
        try:
            # Try to get table names from the database
            result = self.chat_manager.execute_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1;")
            if result.get('success', False) and 'rows' in result and result['rows']:
                return result['rows'][0][0]
        except:
            pass
        
        # Hardcoded fallbacks based on database name
        if schema_name == 'student_club':
            return 'member'
        elif schema_name == 'formula_1':
            return 'drivers'
        elif schema_name == 'california_schools':
            return 'schools'
        else:
            return 'main_table'  # Generic fallback

    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Process a natural language query and return results.
        
        Args:
            question: Natural language question about the database
            verbose: Whether to print verbose output
            
        Returns:
            Dictionary with query results
        """
        if verbose:
            print(f"\nProcessing query: {question}")
        
        if not self.chat_manager.connected:
            if verbose:
                print("Not connected to database. Attempting to connect...")
            
            # Check if we have a database name
            if not self.chat_manager.db_name:
                if verbose:
                    print("No database specified. Cannot execute query.")
                return {"error": "No database specified"}
            
            # Try to connect
            success = self.chat_manager.connect_to_database(self.chat_manager.db_name)
            if not success:
                if verbose:
                    print(f"Failed to connect to database: {self.chat_manager.db_name}")
                return {"error": f"Failed to connect to database: {self.chat_manager.db_name}"}
        
        # Execute query through agents
        try:
            start_time = time.time()
            
            # Step 1: Select relevant schema based on question
            if verbose:
                print("\n=== Step 1: Schema Selection ===")
            
            schema_docs = self.chat_manager.selector_agent.select_schema(
                question, 
                self.chat_manager.schema_knowledge
            )
            
            # Combine schema docs into a single string
            if isinstance(schema_docs, list):
                schema = "\n\n".join([doc.page_content for doc in schema_docs])
            else:
                schema = str(schema_docs)
            
            if verbose:
                print(f"Selected schema with {len(schema)} characters")
            
            # Step 2: Understand and Plan the query
            if verbose:
                print("\n=== Step 2: Question Understanding and Query Planning ===")
            
            understanding_and_plan = self.chat_manager.decomposer_agent.understand_and_plan(
                question,
                schema
            )
            
            if verbose:
                print(f"Understanding: {understanding_and_plan.get('understanding', '')[:150]}...")
                print(f"Plan: {understanding_and_plan.get('plan', '')[:150]}...")
            
            # Step 3: Generate and execute SQL
            if verbose:
                print("\n=== Step 3: SQL Generation and Execution ===")
            
            # Generate initial SQL
            sql_generation = self.chat_manager.refiner_agent.generate_sql(
                question,
                schema,
                understanding_and_plan
            )
            
            sql_query = sql_generation.get("sql", "")
            
            if verbose:
                print(f"Generated SQL: {sql_query}")
                
            # Execute SQL query
            query_success = False
            max_refinement_attempts = 3
            refinement_count = 0
            error_message = None
            query_result = None
            
            while not query_success and refinement_count < max_refinement_attempts:
                try:
                    # Execute the query with automatic transaction handling
                    query_result = self.chat_manager.execute_sql_query(sql_query)
                    
                    if query_result.get('success', False):
                        query_success = True
                        if verbose:
                            print(f"Query executed successfully with {len(query_result.get('rows', []))} results")
                    else:
                        error_message = query_result.get('error', 'Unknown error')
                        if verbose:
                            print(f"Query execution failed: {error_message}")
                        
                        # Refine the SQL
                        refinement = self.chat_manager.refiner_agent.refine_sql(
                            question,
                            schema,
                            understanding_and_plan,
                            sql_query,
                            error_message
                        )
                        
                        # Update SQL query for next attempt
                        refined_sql = refinement.get("sql", "")
                        
                        # Don't retry with the same query
                        if refined_sql == sql_query:
                            if verbose:
                                print("Refinement produced the same SQL. Stopping refinement process.")
                            break
                        
                        sql_query = refined_sql
                        
                        if verbose:
                            print(f"Refined SQL (attempt {refinement_count+1}): {sql_query}")
                
                except Exception as e:
                    error_message = str(e)
                    if verbose:
                        print(f"Error during query execution: {error_message}")
                
                refinement_count += 1
                
                # Add delay between refinement attempts (to avoid rate limits)
                if not query_success and refinement_count < max_refinement_attempts:
                    time.sleep(1)  # 1 second delay
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Return results
            return {
                "question": question,
                "understanding": understanding_and_plan.get("understanding", ""),
                "plan": understanding_and_plan.get("plan", ""),
                "sql_query": sql_query,
                "query_result": query_result.get('rows', []) if query_success else None,
                "success": query_success,
                "error": error_message if not query_success else None,
                "execution_time": execution_time,
                "refinement_attempts": refinement_count
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "error": str(e),
                "success": False
            }

    def get_available_databases(self) -> List[str]:
        """
        Get a list of available databases
        
        Returns:
            List of database names
        """
        try:
            # Temporarily connect to the default database
            import psycopg2
            conn = psycopg2.connect(
                dbname=self.chat_manager.db_config.get("dbname", "postgres"),
                user=self.chat_manager.db_config.get("user", "postgres"),
                password=self.chat_manager.db_config.get("password", "postgres"),
                host=self.chat_manager.db_config.get("host", "localhost"),
                port=self.chat_manager.db_config.get("port", "5432")
            )
            cursor = conn.cursor()
            
            # Get a list of databases
            cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false")
            databases = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return databases
        except Exception as e:
            print(f"Error getting available databases: {e}")
            return []

    def evaluate_benchmark(self, benchmark_file, num_samples=2, output_file=None):
        """
        Evaluate the model on a benchmark dataset
        
        Args:
            benchmark_file: Path to benchmark JSON file
            num_samples: Number of samples per database to evaluate
            output_file: Path to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            print(f"Starting evaluation with {self.model_name} model")
            start_time = time.time()
            
            # Load benchmark from file
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                benchmark = json.load(f)
            
            print(f"Loaded benchmark data from {benchmark_file}")
            
            # Sample benchmark items by database
            db_items = {}
            for item in benchmark:
                db_id = item.get("db_id")
                if db_id:
                    if db_id not in db_items:
                        db_items[db_id] = []
                    db_items[db_id].append(item)
            
            # Sample items from each database
            sampled_items = []
            for db_id, items in db_items.items():
                # Take at most num_samples items from each database
                sample_size = min(num_samples, len(items))
                sampled_items.extend(random.sample(items, sample_size))
            
            print(f"Sampled {len(sampled_items)} items from {len(db_items)} databases")
            
            # Initialize results
            results = []
            
            # Process each benchmark item
            for i, item in enumerate(sampled_items):
                db_id = item.get("db_id", "")
                question = item.get("question", "")
                gold_sql = item.get("SQL", "")  # Note: Using uppercase "SQL" key as in BIRD
                evidence = item.get("evidence", "")
                
                print(f"\n[{i+1}/{len(sampled_items)}] Processing question for database '{db_id}':")
                print(f"  Question: {question}")
                print(f"  Evidence: {evidence}")
                print(f"  Gold SQL: {gold_sql}")
                
                # Check if the database exists in our PostgreSQL instance
                available_dbs = self.get_available_databases()
                db_exists = db_id in available_dbs
                
                if not db_exists:
                    print(f"  WARNING: Database '{db_id}' not found in PostgreSQL. Using hardcoded schema if available.")
                    print(f"  Available databases: {', '.join(available_dbs[:10])}{'...' if len(available_dbs) > 10 else ''}")
                
                # Connect to the correct database before processing
                print(f"  Connecting to database: {db_id}...")
                connection_success = self.chat_manager.connect_to_database(db_id)
                if not connection_success:
                    print(f"  WARNING: Failed to connect to database '{db_id}', will use hardcoded schema if available")
                    
                    # Check if this is a BIRD benchmark database
                    is_bird_db = self.chat_manager._is_bird_benchmark_db(db_id)
                    if is_bird_db:
                        print(f"  INFO: '{db_id}' is a BIRD benchmark database, hardcoded schema should be available")
                    else:
                        print(f"  ERROR: '{db_id}' is not recognized as a BIRD benchmark database")
                
                # Add schema debugging output
                schema_info = self.chat_manager.get_schema(force_refresh=True)
                if not schema_info:
                    print(f"  ERROR: No schema available for database '{db_id}'. Results will likely be incorrect.")
                else:
                    schema_length = len(schema_info)
                    schema_preview = schema_info[:200] + "..." if len(schema_info) > 200 else schema_info
                    print(f"\n[DEBUG] Schema for database '{db_id}' (length: {schema_length} chars):")
                    print("=" * 80)
                    print(f"{schema_preview}")
                    print("=" * 80)
                
                # Using our improved agent workflow
                result = self.chat_manager.process_query(
                    user_query=question,
                    db_id=db_id,  # Explicitly pass db_id to ensure correct database
                    evidence=evidence
                )
                
                # Check for success or failure
                if result.get('success', False):
                    generated_sql = result.get('sql', '')
                    print(f"  Generated SQL: {generated_sql}")
                    
                    # Compare generated SQL with gold SQL
                    sql_match, similarity_score, comparison_notes = self._compare_sql(generated_sql, gold_sql)
                    
                    # Execute generated SQL to check for execution errors
                    try:
                        execution_result = self.chat_manager.execute_sql_query(generated_sql)
                        execution_success = execution_result.get('success', False)
                        
                        # Now also execute the gold SQL to compare results
                        gold_execution_result = self.chat_manager.execute_sql_query(gold_sql)
                        gold_execution_success = gold_execution_result.get('success', False)
                        
                        # Compare execution results if both executed successfully
                        results_match = False
                        result_similarity = 0.0
                        result_metrics = {}
                        
                        if execution_success and gold_execution_success:
                            results_match, result_similarity, result_metrics = self._compare_results(
                                execution_result, gold_execution_result
                            )
                            print(f"  Results Match: {results_match}, Similarity: {result_similarity:.2f}")
                            if result_metrics.get('f1_score') is not None:
                                print(f"  Results F1 Score: {result_metrics.get('f1_score'):.2f}")
                    except Exception as e:
                        execution_success = False
                        results_match = False
                        result_similarity = 0.0
                        result_metrics = {}
                        print(f"  Execution error: {e}")
                    
                    # Add result to results list
                    results.append({
                        'db_id': db_id,
                        'question': question,
                        'gold_sql': gold_sql,
                        'generated_sql': generated_sql,
                        'execution_success': execution_success,
                        'sql_match': sql_match,
                        'similarity_score': similarity_score,
                        'results_match': results_match,
                        'result_similarity': result_similarity,
                        'result_metrics': result_metrics,
                        'understanding': result.get('understanding', ''),
                        'plan': result.get('plan', ''),
                        'evidence': evidence,
                        'attempts': result.get('attempts', 1),
                        'execution_time': result.get('execution_time', 0)
                    })
                    
                    print(f"  SQL Match: {sql_match}, Execution Success: {execution_success}, Similarity Score: {similarity_score:.2f}")
                    if results_match:
                        print(f"  Results Match: True, F1 Score: {result_similarity:.2f}")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"  Error: {error}")
                    
                    # Add failed result to results list
                    results.append({
                        'db_id': db_id,
                        'question': question,
                        'gold_sql': gold_sql,
                        'generated_sql': result.get('sql', ''),
                        'execution_success': False,
                        'sql_match': False,
                        'similarity_score': 0.0,
                        'results_match': False,
                        'result_similarity': 0.0,
                        'result_metrics': {},
                        'understanding': result.get('understanding', ''),
                        'plan': result.get('plan', ''),
                        'evidence': evidence,
                        'error': error,
                        'attempts': result.get('attempts', 1),
                        'execution_time': result.get('execution_time', 0)
                    })
            
            # Calculate aggregate statistics
            total_items = len(results)
            sql_match_count = sum(1 for item in results if item.get('sql_match', False))
            execution_success_count = sum(1 for item in results if item.get('execution_success', False))
            results_match_count = sum(1 for item in results if item.get('results_match', False))
            avg_similarity = sum(item.get('similarity_score', 0) for item in results) / total_items if total_items > 0 else 0
            avg_result_similarity = sum(item.get('result_similarity', 0) for item in results) / total_items if total_items > 0 else 0
            avg_attempts = sum(item.get('attempts', 1) for item in results) / total_items if total_items > 0 else 0
            
            # Group results by database
            db_results = {}
            for result in results:
                db_id = result.get('db_id', '')
                if db_id not in db_results:
                    db_results[db_id] = []
                db_results[db_id].append(result)
            
            # Calculate statistics by database
            db_stats = {}
            for db_id, db_result_list in db_results.items():
                total_db_items = len(db_result_list)
                db_sql_match_count = sum(1 for item in db_result_list if item.get('sql_match', False))
                db_execution_success_count = sum(1 for item in db_result_list if item.get('execution_success', False))
                db_results_match_count = sum(1 for item in db_result_list if item.get('results_match', False))
                db_avg_similarity = sum(item.get('similarity_score', 0) for item in db_result_list) / total_db_items if total_db_items > 0 else 0
                db_avg_result_similarity = sum(item.get('result_similarity', 0) for item in db_result_list) / total_db_items if total_db_items > 0 else 0
                
                db_stats[db_id] = {
                    'total_items': total_db_items,
                    'sql_match_count': db_sql_match_count,
                    'sql_match_rate': db_sql_match_count / total_db_items if total_db_items > 0 else 0,
                    'execution_success_count': db_execution_success_count,
                    'execution_success_rate': db_execution_success_count / total_db_items if total_db_items > 0 else 0,
                    'results_match_count': db_results_match_count,
                    'results_match_rate': db_results_match_count / total_db_items if total_db_items > 0 else 0,
                    'avg_similarity': db_avg_similarity,
                    'avg_result_similarity': db_avg_result_similarity
                }
            
            # Create evaluation summary
            evaluation_summary = {
                'model_name': self.model_name,
                'benchmark_file': benchmark_file,
                'total_items': total_items,
                'sql_match_count': sql_match_count,
                'sql_match_rate': sql_match_count / total_items if total_items > 0 else 0,
                'execution_success_count': execution_success_count,
                'execution_success_rate': execution_success_count / total_items if total_items > 0 else 0,
                'results_match_count': results_match_count,
                'results_match_rate': results_match_count / total_items if total_items > 0 else 0,
                'avg_similarity': avg_similarity,
                'avg_result_similarity': avg_result_similarity,
                'avg_attempts': avg_attempts,
                'database_stats': db_stats,
                'results': results,
                'evaluation_time': time.time() - start_time
            }
            
            # Save results to file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_summary, f, indent=2)
                print(f"Saved evaluation results to {output_file}")
            
            print("\nEvaluation Summary:")
            print(f"  Total Items: {total_items}")
            print(f"  SQL Match Rate: {sql_match_count}/{total_items} ({evaluation_summary['sql_match_rate']:.2%})")
            print(f"  Execution Success Rate: {execution_success_count}/{total_items} ({evaluation_summary['execution_success_rate']:.2%})")
            print(f"  Results Match Rate: {results_match_count}/{total_items} ({evaluation_summary['results_match_rate']:.2%})")
            print(f"  Average SQL Similarity Score: {avg_similarity:.4f}")
            print(f"  Average Results Similarity Score: {avg_result_similarity:.4f}")
            print(f"  Average Attempts: {avg_attempts:.2f}")
            print(f"  Evaluation Time: {evaluation_summary['evaluation_time']:.2f} seconds")
            
            return evaluation_summary
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'success': False
            }

    def _compare_sql(self, generated_sql: str, gold_sql: str) -> Tuple[bool, float, str]:
        """
        Compare two SQL queries and return a tuple indicating whether they match,
        the similarity score, and any notes about the comparison.
        
        Args:
            generated_sql: The generated SQL query
            gold_sql: The gold SQL query
            
        Returns:
            Tuple containing:
            - bool: Whether the queries match
            - float: Similarity score between 0 and 1
            - str: Notes about the comparison
        """
        if not generated_sql or not gold_sql:
            return False, 0.0, "One or both SQL queries are empty"
        
        # Normalize SQL for comparison
        def normalize_sql(sql):
            if not sql:
                return ""
            # Convert to lowercase
            sql = sql.lower()
            # Remove extra whitespace
            sql = ' '.join(sql.split())
            # Remove trailing semicolons
            sql = sql.rstrip(';')
            # Normalize quotes (replace double quotes with single quotes)
            sql = sql.replace('"', "'")
            return sql
        
        normalized_generated = normalize_sql(generated_sql)
        normalized_gold = normalize_sql(gold_sql)
        
        # Check for exact match after normalization
        if normalized_generated == normalized_gold:
            return True, 1.0, "SQL queries match exactly after normalization"
        
        # Calculate similarity based on keywords and structure
        # Extract main components from both queries
        def extract_components(sql):
            components = {
                'select': [],
                'from': [],
                'where': [],
                'group_by': [],
                'order_by': [],
                'having': [],
                'limit': None
            }
            
            # Extract SELECT clause
            select_match = re.search(r'select\s+(.*?)(?:\s+from\s+|\s*$)', sql, re.IGNORECASE)
            if select_match:
                select_clause = select_match.group(1).strip()
                components['select'] = [col.strip() for col in select_clause.split(',')]
            
            # Extract FROM clause
            from_match = re.search(r'from\s+(.*?)(?:\s+where\s+|\s+group\s+by\s+|\s+order\s+by\s+|\s+having\s+|\s+limit\s+|\s*$)', sql, re.IGNORECASE)
            if from_match:
                from_clause = from_match.group(1).strip()
                # Handle joins in FROM clause
                if 'join' in from_clause.lower():
                    # This is a simplified approach - a real parser would be more robust
                    components['from'] = [table.strip() for table in re.split(r'\s+(?:left|right|inner|outer|full|cross)?\s+join\s+', from_clause, flags=re.IGNORECASE) if table.strip()]
                else:
                    components['from'] = [table.strip() for table in from_clause.split(',')]
            
            # Extract WHERE clause
            where_match = re.search(r'where\s+(.*?)(?:\s+group\s+by\s+|\s+order\s+by\s+|\s+having\s+|\s+limit\s+|\s*$)', sql, re.IGNORECASE)
            if where_match:
                where_clause = where_match.group(1).strip()
                # Split conditions by AND/OR
                conditions = re.split(r'\s+(?:and|or)\s+', where_clause, flags=re.IGNORECASE)
                components['where'] = [cond.strip() for cond in conditions]
            
            # Extract GROUP BY clause
            group_match = re.search(r'group\s+by\s+(.*?)(?:\s+order\s+by\s+|\s+having\s+|\s+limit\s+|\s*$)', sql, re.IGNORECASE)
            if group_match:
                group_clause = group_match.group(1).strip()
                components['group_by'] = [col.strip() for col in group_clause.split(',')]
            
            # Extract ORDER BY clause
            order_match = re.search(r'order\s+by\s+(.*?)(?:\s+limit\s+|\s*$)', sql, re.IGNORECASE)
            if order_match:
                order_clause = order_match.group(1).strip()
                components['order_by'] = [col.strip() for col in order_clause.split(',')]
            
            # Extract HAVING clause
            having_match = re.search(r'having\s+(.*?)(?:\s+order\s+by\s+|\s+limit\s+|\s*$)', sql, re.IGNORECASE)
            if having_match:
                having_clause = having_match.group(1).strip()
                components['having'] = [cond.strip() for cond in re.split(r'\s+(?:and|or)\s+', having_clause, flags=re.IGNORECASE)]
            
            # Extract LIMIT clause
            limit_match = re.search(r'limit\s+(\d+)', sql, re.IGNORECASE)
            if limit_match:
                components['limit'] = int(limit_match.group(1))
            
            return components
        
        # Extract components from both queries
        generated_components = extract_components(normalized_generated)
        gold_components = extract_components(normalized_gold)
        
        # Calculate similarity scores for each component
        similarity_scores = {}
        total_weight = 0.0
        
        # Weights for each component (adjust as needed)
        weights = {
            'select': 0.3,
            'from': 0.2,
            'where': 0.25,
            'group_by': 0.1,
            'order_by': 0.05,
            'having': 0.05,
            'limit': 0.05
        }
        
        # Compare SELECT clauses
        if generated_components['select'] and gold_components['select']:
            select_sim = self._compare_lists(generated_components['select'], gold_components['select'])
            similarity_scores['select'] = select_sim
            total_weight += weights['select']
        
        # Compare FROM clauses
        if generated_components['from'] and gold_components['from']:
            from_sim = self._compare_lists(generated_components['from'], gold_components['from'])
            similarity_scores['from'] = from_sim
            total_weight += weights['from']
        
        # Compare WHERE clauses
        if generated_components['where'] and gold_components['where']:
            where_sim = self._compare_lists(generated_components['where'], gold_components['where'])
            similarity_scores['where'] = where_sim
            total_weight += weights['where']
        
        # Compare GROUP BY clauses
        if generated_components['group_by'] and gold_components['group_by']:
            group_sim = self._compare_lists(generated_components['group_by'], gold_components['group_by'])
            similarity_scores['group_by'] = group_sim
            total_weight += weights['group_by']
        
        # Compare ORDER BY clauses
        if generated_components['order_by'] and gold_components['order_by']:
            order_sim = self._compare_lists(generated_components['order_by'], gold_components['order_by'])
            similarity_scores['order_by'] = order_sim
            total_weight += weights['order_by']
        
        # Compare HAVING clauses
        if generated_components['having'] and gold_components['having']:
            having_sim = self._compare_lists(generated_components['having'], gold_components['having'])
            similarity_scores['having'] = having_sim
            total_weight += weights['having']
        
        # Compare LIMIT clauses
        if generated_components['limit'] is not None and gold_components['limit'] is not None:
            limit_sim = 1.0 if generated_components['limit'] == gold_components['limit'] else 0.0
            similarity_scores['limit'] = limit_sim
            total_weight += weights['limit']
        
        # Calculate weighted similarity score
        weighted_score = 0.0
        for component, score in similarity_scores.items():
            weighted_score += score * weights[component]
        
        # Normalize score based on total weight
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0
        
        # Generate comparison notes
        comparison_notes = []
        for component, score in similarity_scores.items():
            comparison_notes.append(f"{component.upper()}: {score:.2f}")
        
        # Determine match status
        is_match = final_score > 0.85  # Threshold for considering a match
        
        return is_match, final_score, ", ".join(comparison_notes)
    
    def _compare_results(self, generated_results, gold_results):
        """
        Compare the results of executing the generated and gold SQL queries
        
        Args:
            generated_results: Results from executing the generated SQL (rows and columns)
            gold_results: Results from executing the gold SQL (rows and columns)
            
        Returns:
            Tuple containing:
            - bool: Whether the results match
            - float: Result similarity score between 0 and 1
            - dict: Detailed comparison metrics
        """
        # Initialize default return values
        exact_match = False
        similarity_score = 0.0
        comparison_metrics = {
            'rows_match': False,
            'columns_match': False,
            'total_rows_generated': 0,
            'total_rows_gold': 0,
            'common_rows': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        # Handle None or empty results
        if generated_results is None or gold_results is None:
            return exact_match, similarity_score, comparison_metrics
            
        # Extract rows and columns
        generated_rows = generated_results.get('rows', [])
        generated_cols = generated_results.get('columns', [])
        gold_rows = gold_results.get('rows', [])
        gold_cols = gold_results.get('columns', [])
        
        # Update metrics
        comparison_metrics['total_rows_generated'] = len(generated_rows)
        comparison_metrics['total_rows_gold'] = len(gold_rows)
        
        # Handle empty results case
        if len(generated_rows) == 0 and len(gold_rows) == 0:
            return True, 1.0, {**comparison_metrics, 'rows_match': True, 'columns_match': True, 'f1_score': 1.0}
        
        # Check if columns match (order might be different)
        if set(generated_cols) == set(gold_cols):
            comparison_metrics['columns_match'] = True
        
        # If no rows in one or both results, we can't do much comparison
        if len(generated_rows) == 0 or len(gold_rows) == 0:
            # If both are empty, that's a match
            if len(generated_rows) == 0 and len(gold_rows) == 0:
                return True, 1.0, {**comparison_metrics, 'rows_match': True}
            # Otherwise, no match
            return False, 0.0, comparison_metrics
        
        # Convert rows to sets of tuples for comparison
        # This allows us to compare results regardless of row order
        generated_row_set = set(tuple(row) for row in generated_rows)
        gold_row_set = set(tuple(row) for row in gold_rows)
        
        # Find common rows
        common_rows = generated_row_set.intersection(gold_row_set)
        comparison_metrics['common_rows'] = len(common_rows)
        
        # Calculate precision and recall
        if len(generated_rows) > 0:
            comparison_metrics['precision'] = len(common_rows) / len(generated_rows)
        if len(gold_rows) > 0:
            comparison_metrics['recall'] = len(common_rows) / len(gold_rows)
        
        # Calculate F1 score
        precision = comparison_metrics['precision']
        recall = comparison_metrics['recall']
        if precision + recall > 0:
            comparison_metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
        
        # Determine if results match (using F1 score threshold)
        f1_score = comparison_metrics['f1_score']
        exact_match = f1_score > 0.95  # High threshold for considering a match
        similarity_score = f1_score  # Use F1 as the similarity score
        
        # Set rows_match based on exact match
        comparison_metrics['rows_match'] = exact_match
        
        return exact_match, similarity_score, comparison_metrics

    def _compare_lists(self, list1, list2):
        """
        Calculate similarity between two lists (set-based comparison)
        
        Args:
            list1: First list
            list2: Second list
            
        Returns:
            Similarity score between 0 and 1
        """
        if not list1 and not list2:
            return 1.0
        
        if not list1 or not list2:
            return 0.0
        
        set1 = set(list1)
        set2 = set(list2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAC-SQL: Multi-Agent Collaborative Framework for Text-to-SQL")
    parser.add_argument("--question", help="Natural language question to convert to SQL")
    parser.add_argument("--db", help="Database to connect to")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model to use")
    parser.add_argument("--api_key", help="API key for the LLM service")
    parser.add_argument("--verbose", action="store_true", help="Print detailed processing steps")
    
    args = parser.parse_args()
    
    if args.question:
        # Initialize MAC-SQL
        mac_sql = MACSQL(args.model, args.api_key)
        
        # Connect to specified database if provided
        if args.db:
            mac_sql.connect_to_database(args.db)
        
        # Process the query
        result = mac_sql.query(args.question, args.verbose)
        
        # Print the result
        print(f"\nSQL Query: {result['sql_query']}")
        print("\nQuery Result:")
        print(result["query_result"])
    else:
        parser.print_help() 