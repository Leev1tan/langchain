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

from core.chat_manager import ChatManager
from core.agents import SelectorAgent, DecomposerAgent, RefinerAgent
from core.config import DB_CONFIG

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
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """
        Initialize the MAC-SQL framework
        
        Args:
            model_name: Name of the LLM to use
            api_key: API key for the LLM service (optional)
            temperature: Temperature for the LLM
            max_tokens: Maximum number of tokens for the LLM
            **kwargs: Additional keyword arguments
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
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
        self.selector_agent = SelectorAgent(model_name, api_key)
        self.decomposer_agent = DecomposerAgent(model_name, api_key)
        self.refiner_agent = RefinerAgent(model_name, api_key)
    
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
        if self.chat_manager.update_database(db_name):
            # If connection was successful, cache the schema
            if self.chat_manager.schema_knowledge:
                self.add_to_schema_cache(db_name, self.chat_manager.schema_knowledge)
            return True
        return False
    
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
            success = self.chat_manager.update_database(self.chat_manager.db_name)
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
                db_id = item.get("db_id", "").lower()
                if db_id not in db_items:
                    db_items[db_id] = []
                db_items[db_id].append(item)
            
            # Sample items from each database
            sampled_items = []
            for db_id, items in db_items.items():
                n = min(num_samples, len(items))
                sampled_items.extend(random.sample(items, n))
            
            print(f"Sampled {len(sampled_items)} benchmark items from {len(db_items)} databases")
            
            # Initialize results
            results = {
                "model": self.model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_accuracy": 0.0,
                "correct": 0,
                "total": len(sampled_items),
                "error_count": 0,
                "detailed_results": [],
                "results_by_database": {}
            }
            
            # Track current database to avoid reconnecting if unnecessary
            current_db = None
            current_chat_manager = None
            
            # Iterate through sampled items
            for i, item in enumerate(sampled_items):
                db_id = item.get("db_id", "").lower()
                question = item["question"]
                gold_sql = item.get("query", "")
                question_id = item.get("question_id", i)
                
                print(f"\nEvaluating item {i+1}/{len(sampled_items)}: Database: {db_id}")
                print(f"Question: {question}")
                
                # Initialize result item
                result_item = {
                    "db_id": db_id,
                    "question": question,
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "results_match": False
                }
                
                try:
                    # Connect to the appropriate database if needed
                    if current_db != db_id:
                        print(f"Connecting to database: {db_id}")
                        
                        # Initialize a new ChatManager for the current database
                        current_chat_manager = ChatManager(
                            model_name=self.model_name,
                            api_key=self.api_key,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            **self.kwargs
                        )
                        
                        success = current_chat_manager.update_database(db_id)
                        if not success:
                            print(f"Failed to connect to database: {db_id}")
                            result_item["error"] = f"Failed to connect to database: {db_id}"
                            results["detailed_results"].append(result_item)
                            results["error_count"] += 1
                            continue
                        
                        current_db = db_id
                    
                    # Measure query execution time
                    query_start_time = time.time()
                    
                    # Try to get schema from cache first
                    schema_knowledge = self.get_from_schema_cache(db_id)
                    if schema_knowledge:
                        current_chat_manager.schema_knowledge = schema_knowledge
                    
                    # Process query with the chat manager
                    result = current_chat_manager.process_query(question)
                    generated_sql = result.get("sql", "")
                    understanding = result.get("understanding", "")
                    plan = result.get("plan", "")
                    
                    # Record generated SQL and reasoning
                    result_item["generated_sql"] = generated_sql
                    result_item["understanding"] = understanding
                    result_item["plan"] = plan
                    
                    # Initialize variables for results
                    generated_result = None
                    gold_result = None
                    results_match = False
                    
                    # Execute generated SQL if available
                    if generated_sql:
                        try:
                            generated_result = current_chat_manager.execute_sql_query(generated_sql)
                            result_item["generated_result"] = generated_result
                        except Exception as sql_error:
                            print(f"Error executing generated SQL: {sql_error}")
                            result_item["error"] = f"Generated SQL execution error: {str(sql_error)}"
                            results["error_count"] += 1
                    else:
                        result_item["error"] = "No SQL was generated"
                        results["error_count"] += 1
                    
                    # Execute gold SQL if available
                    if gold_sql:
                        try:
                            gold_result = current_chat_manager.execute_sql_query(gold_sql)
                            result_item["gold_result"] = gold_result
                        except Exception as gold_error:
                            print(f"Error executing gold SQL: {gold_error}")
                            result_item["error"] = f"Gold SQL execution error: {str(gold_error)}"
                    
                    # Compare results if both executed successfully
                    if (generated_result and gold_result and 
                        generated_result.get('success', False) and 
                        gold_result.get('success', False)):
                        
                        # Get result sets
                        gen_rows = generated_result.get('rows', [])
                        gold_rows = gold_result.get('rows', [])
                        
                        # Special case for count queries (single value results)
                        if (len(gen_rows) == 1 and len(gold_rows) == 1 and
                            isinstance(gen_rows[0], dict) and isinstance(gold_rows[0], dict)):
                            
                            # Extract the values (first value of each row)
                            gen_value = list(gen_rows[0].values())[0] if gen_rows[0] else None
                            gold_value = list(gold_rows[0].values())[0] if gold_rows[0] else None
                            
                            # Try to convert strings to numbers if needed
                            if gen_value is not None and gold_value is not None:
                                try:
                                    if isinstance(gen_value, str) and isinstance(gold_value, (int, float)):
                                        gen_value = float(gen_value)
                                    elif isinstance(gold_value, str) and isinstance(gen_value, (int, float)):
                                        gold_value = float(gold_value)
                                except Exception:
                                    # If conversion fails, just use the original values
                                    pass
                            
                            # Compare values with tolerance for numeric values
                            if isinstance(gen_value, (int, float)) and isinstance(gold_value, (int, float)):
                                results_match = abs(gen_value - gold_value) < 0.001
                            else:
                                results_match = gen_value == gold_value
                        
                        # For more complex results with multiple rows
                        elif len(gen_rows) == len(gold_rows):
                            # Start with assumption they match
                            results_match = True
                            
                            # For single column results, compare values
                            if gen_rows and isinstance(gen_rows[0], dict) and isinstance(gold_rows[0], dict):
                                gen_keys = set(gen_rows[0].keys())
                                gold_keys = set(gold_rows[0].keys())
                                
                                # If key sets are different but both have only one column,
                                # we can still try to compare values
                                if len(gen_keys) == 1 and len(gold_keys) == 1:
                                    gen_values = [list(row.values())[0] for row in gen_rows]
                                    gold_values = [list(row.values())[0] for row in gold_rows]
                                    
                                    # Sort values for comparison if they're all the same type
                                    if all(isinstance(v, (int, float)) for v in gen_values + gold_values):
                                        gen_values.sort()
                                        gold_values.sort()
                                        
                                        # Compare each value with tolerance
                                        for gv, gr in zip(gen_values, gold_values):
                                            if abs(gv - gr) > 0.001 * max(abs(gv), abs(gr), 1.0):
                                                results_match = False
                                                break
                                    else:
                                        # For non-numeric values, sort and compare directly
                                        gen_values.sort(key=str)
                                        gold_values.sort(key=str)
                                        results_match = gen_values == gold_values
                
                    # Update result with match status
                    result_item["results_match"] = results_match
                    
                    # Update correct count if results match
                    if results_match:
                        results["correct"] += 1
                    
                    # Measure execution time
                    query_end_time = time.time()
                    execution_time = query_end_time - query_start_time
                    result_item["execution_time"] = execution_time
                    
                    # Categorize query complexity and type if gold SQL is available
                    if gold_sql:
                        try:
                            result_item["complexity"] = self._categorize_query_complexity(gold_sql)
                            result_item["query_type"] = self._categorize_query_type(gold_sql)
                        except Exception as e:
                            print(f"Error categorizing query: {e}")
                    
                    # Print result
                    status = "✓" if results_match else "✗"
                    print(f"Result: {status} ({execution_time:.2f}s)")
                    print(f"Generated SQL: {generated_sql}")
                    
                except Exception as e:
                    # Handle any other errors during processing
                    print(f"Error processing question: {e}")
                    traceback.print_exc()
                    
                    # Record the error
                    result_item["error"] = str(e)
                    results["error_count"] += 1
                
                # Add result to detailed results (happens regardless of success/failure)
                results["detailed_results"].append(result_item)
                
                # Update results by database
                if db_id not in results["results_by_database"]:
                    results["results_by_database"][db_id] = {
                        "correct": 0,
                        "total": 0,
                        "accuracy": 0.0
                    }
                
                results["results_by_database"][db_id]["total"] += 1
                if result_item.get("results_match", False):
                    results["results_by_database"][db_id]["correct"] += 1
                
                # Calculate accuracy for this database
                db_total = results["results_by_database"][db_id]["total"]
                db_correct = results["results_by_database"][db_id]["correct"]
                
                if db_total > 0:
                    accuracy = (db_correct / db_total) * 100
                    results["results_by_database"][db_id]["accuracy"] = accuracy
            
            # Calculate overall accuracy
            if results["total"] > 0:
                results["execution_accuracy"] = (results["correct"] / results["total"]) * 100
            
            # Calculate total evaluation time
            end_time = time.time()
            results["evaluation_time"] = end_time - start_time
            
            # Save results to file if specified
            if output_file:
                os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                print(f"Evaluation results saved to {output_file}")
            
            return results
        
        except Exception as e:
            print(f"Error in benchmark evaluation: {e}")
            traceback.print_exc()
            return {
                "error": str(e),
                "model": self.model_name,
                "execution_accuracy": 0.0,
                "correct": 0,
                "total": 0
            }

    def _categorize_query_complexity(self, query):
        """
        Categorize SQL query by complexity
        
        Args:
            query: SQL query string
            
        Returns:
            Complexity category: 'simple', 'medium', or 'complex'
        """
        query = query.lower()
        
        # Count specific SQL features
        joins = query.count('join')
        aggregations = sum(1 for agg in ['count(', 'sum(', 'avg(', 'min(', 'max('] if agg in query)
        subqueries = query.count('select') - 1  # Subtract 1 for the main query
        group_by = 1 if 'group by' in query else 0
        having = 1 if 'having' in query else 0
        order_by = 1 if 'order by' in query else 0
        distinct = 1 if 'distinct' in query else 0
        
        # Calculate complexity score
        complexity_score = joins + aggregations*1.5 + subqueries*2 + group_by + having*1.5 + order_by*0.5 + distinct*0.5
        
        # Categorize based on score
        if complexity_score <= 1:
            return 'simple'
        elif complexity_score <= 4:
            return 'medium'
        else:
            return 'complex'

    def _categorize_query_type(self, query):
        """
        Categorize SQL query by type
        
        Args:
            query: SQL query string
            
        Returns:
            Query type: 'select', 'aggregation', 'join', etc.
        """
        query = query.lower()
        
        # Identify the primary type
        if 'join' in query:
            if any(agg in query for agg in ['count(', 'sum(', 'avg(', 'min(', 'max(']):
                return 'join_with_aggregation'
            return 'join'
        elif any(agg in query for agg in ['count(', 'sum(', 'avg(', 'min(', 'max(']):
            if 'group by' in query:
                return 'group_aggregation'
            return 'aggregation'
        elif 'where' in query:
            return 'filtered_select'
        else:
            return 'simple_select'

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