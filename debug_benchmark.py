#!/usr/bin/env python
"""
MAC-SQL Debug Benchmark
=======================

A minimal benchmark run with 1-2 samples to debug SQL generation.
"""

import json
import os
import sys
import time
from mac_sql import MACSQL

def debug_benchmark(benchmark_file, db_id=None, num_samples=2, api_key=None):
    """
    Run a minimal benchmark for debugging purposes
    
    Args:
        benchmark_file: Path to benchmark JSON file
        db_id: Optional specific database to test (if None, randomly samples)
        num_samples: Number of samples to test (default: 2)
        api_key: API key for LLM service
    """
    print(f"Loading benchmark file: {benchmark_file}")
    try:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
    except Exception as e:
        print(f"Error loading benchmark file: {e}")
        return
    
    print(f"Loaded {len(benchmark)} items from benchmark file")
    
    # Filter by database ID if specified
    if db_id:
        benchmark = [item for item in benchmark if item.get("db_id") == db_id]
        print(f"Filtered to {len(benchmark)} items for database '{db_id}'")
    
    # Take only the requested number of samples
    import random
    if len(benchmark) > num_samples:
        samples = random.sample(benchmark, num_samples)
    else:
        samples = benchmark
    
    print(f"Selected {len(samples)} samples for debugging")
    
    # Initialize MAC-SQL with a lower rate of API calls
    mac_sql = MACSQL(
        api_key=api_key,
        temperature=0.1,  # Lower temperature for more deterministic results
        verbose=True      # Enable verbose mode
    )
    
    # Add a configuration to avoid rate limiting
    if hasattr(mac_sql.chat_manager, 'generate_llm_response'):
        original_generate_llm_response = mac_sql.chat_manager.generate_llm_response
        
        def rate_limited_generate_llm_response(*args, **kwargs):
            """
            Wrapper around generate_llm_response with rate limiting
            """
            print("Applying rate limiting (10-second delay) for API call...")
            time.sleep(10)  # 10-second delay between API calls
            return original_generate_llm_response(*args, **kwargs)
        
        mac_sql.chat_manager.generate_llm_response = rate_limited_generate_llm_response
    
    # Process each sample
    for i, sample in enumerate(samples):
        db_id = sample.get("db_id", "")
        question = sample.get("question", "")
        gold_sql = sample.get("SQL", "")  # Note: Using uppercase "SQL" key as in BIRD
        evidence = sample.get("evidence", "")
        
        print("\n" + "="*80)
        print(f"SAMPLE {i+1}/{len(samples)}")
        print("="*80)
        print(f"Database: {db_id}")
        print(f"Question: {question}")
        print(f"Evidence: {evidence}")
        print(f"Gold SQL: {gold_sql}")
        print("-"*80)
        
        # Connect to the database
        print(f"Connecting to database: {db_id}...")
        connection_success = mac_sql.chat_manager.connect_to_database(db_id)
        if not connection_success:
            print(f"WARNING: Failed to connect to database '{db_id}', will use hardcoded schema if available")
        
        # Get and display schema
        schema_info = mac_sql.chat_manager.get_schema(force_refresh=True)
        if schema_info:
            print(f"\nSCHEMA for '{db_id}' (showing first 500 chars):")
            print("-"*80)
            print(schema_info[:500] + "..." if len(schema_info) > 500 else schema_info)
            print("-"*80)
        else:
            print(f"WARNING: No schema available for database '{db_id}'")
        
        # Process the query
        print("\nPROCESSING QUERY...")
        start_time = time.time()
        
        try:
            result = mac_sql.chat_manager.process_query(
                user_query=question,
                db_id=db_id,
                evidence=evidence
            )
            
            execution_time = time.time() - start_time
            print(f"Query processing completed in {execution_time:.2f} seconds")
            
            if result.get('success', False):
                # Successfully generated SQL
                generated_sql = result.get('sql', '')
                
                print("\nGENERATED SQL:")
                print("-"*80)
                print(generated_sql)
                print("-"*80)
                
                # Compare with gold SQL
                sql_match, similarity_score, comparison_notes = mac_sql._compare_sql(generated_sql, gold_sql)
                
                print("\nSQL COMPARISON:")
                print(f"Match: {sql_match}")
                print(f"Similarity Score: {similarity_score:.4f}")
                print(f"Component Scores: {comparison_notes}")
                
                # Execute the generated SQL
                print("\nEXECUTING GENERATED SQL...")
                try:
                    execution_result = mac_sql.chat_manager.execute_sql_query(generated_sql)
                    execution_success = execution_result.get('success', False)
                    
                    if execution_success:
                        rows = execution_result.get('rows', [])
                        columns = execution_result.get('columns', [])
                        
                        print(f"Execution succeeded with {len(rows)} rows and {len(columns)} columns")
                        
                        # Display results (limited to 5 rows for readability)
                        max_rows = min(5, len(rows))
                        if columns and rows:
                            print("\nRESULTS (first 5 rows):")
                            print("-"*80)
                            print(" | ".join(columns))
                            print("-" * 80)
                            for i in range(max_rows):
                                print(" | ".join(str(cell) for cell in rows[i]))
                            if len(rows) > max_rows:
                                print(f"... and {len(rows) - max_rows} more rows")
                            print("-"*80)
                    else:
                        error = execution_result.get('error', 'Unknown error')
                        print(f"Execution failed: {error}")
                except Exception as e:
                    print(f"Error during execution: {e}")
                    execution_success = False
                
                # Execute the gold SQL for comparison
                print("\nEXECUTING GOLD SQL...")
                try:
                    gold_result = mac_sql.chat_manager.execute_sql_query(gold_sql)
                    gold_execution_success = gold_result.get('success', False)
                    
                    if gold_execution_success:
                        gold_rows = gold_result.get('rows', [])
                        gold_columns = gold_result.get('columns', [])
                        
                        print(f"Gold SQL execution succeeded with {len(gold_rows)} rows and {len(gold_columns)} columns")
                        
                        # Display gold results (limited to 5 rows for readability)
                        max_rows = min(5, len(gold_rows))
                        if gold_columns and gold_rows:
                            print("\nGOLD RESULTS (first 5 rows):")
                            print("-"*80)
                            print(" | ".join(gold_columns))
                            print("-" * 80)
                            for i in range(max_rows):
                                print(" | ".join(str(cell) for cell in gold_rows[i]))
                            if len(gold_rows) > max_rows:
                                print(f"... and {len(gold_rows) - max_rows} more rows")
                            print("-"*80)
                    else:
                        error = gold_result.get('error', 'Unknown error')
                        print(f"Gold SQL execution failed: {error}")
                except Exception as e:
                    print(f"Error during gold SQL execution: {e}")
                    gold_execution_success = False
                
                # Compare results if both executed successfully
                if execution_success and gold_execution_success:
                    # Add a safety check for the _compare_results method
                    if hasattr(mac_sql, '_compare_results'):
                        results_match, result_similarity, result_metrics = mac_sql._compare_results(
                            execution_result, gold_result
                        )
                        
                        print("\nRESULTS COMPARISON:")
                        print(f"Match: {results_match}")
                        print(f"Similarity (F1 Score): {result_similarity:.4f}")
                        print(f"Precision: {result_metrics.get('precision', 0):.4f}")
                        print(f"Recall: {result_metrics.get('recall', 0):.4f}")
                        print(f"Common Rows: {result_metrics.get('common_rows', 0)}/{result_metrics.get('total_rows_generated', 0)} (generated) vs {result_metrics.get('total_rows_gold', 0)} (gold)")
                    else:
                        print("\nResults comparison skipped: _compare_results method not found")
            else:
                # Failed to generate SQL
                error = result.get('error', 'Unknown error')
                generated_sql = result.get('sql', '')
                
                print("\nFAILED TO PROCESS QUERY")
                print(f"Error: {error}")
                
                if generated_sql:
                    print("\nPartial SQL generated:")
                    print(generated_sql)
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Query processing failed in {execution_time:.2f} seconds")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("DEBUG BENCHMARK COMPLETED")
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_benchmark.py <benchmark_file> [db_id] [num_samples] [api_key]")
        sys.exit(1)
    
    benchmark_file = sys.argv[1]
    db_id = sys.argv[2] if len(sys.argv) > 2 else None
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    api_key = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("TOGETHER_API_KEY")
    
    debug_benchmark(benchmark_file, db_id, num_samples, api_key) 