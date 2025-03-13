#!/usr/bin/env python
"""
Run the MAC-SQL framework

This script provides a CLI interface for running the MAC-SQL framework.
"""

import argparse
import json
import os
from mac_sql import MACSQL

def run_evaluation(mac_sql, args):
    """
    Run evaluation of MAC-SQL on benchmark data
    
    Args:
        mac_sql: MACSQL instance
        args: Command-line arguments
    """
    # Get benchmark file
    benchmark_file = input(f"Enter path to benchmark file [minidev/MINIDEV/mini_dev_postgresql.json]: ")
    benchmark_file = benchmark_file or "minidev/MINIDEV/mini_dev_postgresql.json"
    
    # Get number of samples per database
    samples_per_db = input("Enter number of samples per database [2]: ")
    samples_per_db = int(samples_per_db) if samples_per_db.isdigit() else 2
    
    print(f"Loading benchmark data from {benchmark_file}...")
    
    # Load benchmark data
    try:
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
            
        # Get unique database IDs
        db_ids = set(item.get("db_id", "").lower() for item in benchmark)
        print(f"Available databases: {', '.join(sorted(db_ids))}")
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Run evaluation
    print("Evaluating MAC-SQL on benchmark data...")
    results = mac_sql.evaluate_benchmark(
        benchmark_file, 
        num_samples=samples_per_db, 
        output_file="results/evaluation_results.json"
    )
    
    # Visualize results
    try:
        mac_sql.visualize_results(results, output_dir="results")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Print results summary
    try:
        print_results_summary(results)
    except Exception as e:
        print(f"Error printing results summary: {e}")
        print("\nRaw results:")
        print(results)

def print_results_summary(results):
    """
    Print a summary of evaluation results
    
    Args:
        results: Dictionary with evaluation results
    """
    print("\n===== MAC-SQL Evaluation Results =====")
    print(f"Model: {results.get('model', 'Unknown')}")
    print(f"Execution Accuracy: {results.get('execution_accuracy', 0.0):.2f}%")
    print(f"Total correct: {results.get('correct', 0)} / {results.get('total', 0)}")
    
    if 'error_count' in results:
        print(f"Errors: {results['error_count']}/{results['total']}")
    
    # Print results by database if available
    if 'results_by_database' in results:
        print("\nResults by database:")
        for db_id, db_result in results['results_by_database'].items():
            print(f"- {db_id}: {db_result['accuracy']:.2f}% ({db_result['correct']}/{db_result['total']})")
    
    # Print detailed results if available
    if 'detailed_results' in results:
        print("\nDetailed results:")
        for item in results['detailed_results']:
            db_id = item.get('db_id', 'Unknown')
            question = item.get('question', 'Unknown')
            results_match = item.get('results_match', False)
            status = "✓" if results_match else "✗"
            print(f"{status} {db_id}: {question[:50]}..." if len(question) > 50 else question)
    
    if 'error' in results:
        print(f"\nError in evaluation: {results['error']}")

def main():
    parser = argparse.ArgumentParser(description="Run MAC-SQL framework")
    
    # Add arguments
    parser.add_argument("--db", help="Database to connect to")
    parser.add_argument("--question", help="Natural language question to convert to SQL")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model to use")
    parser.add_argument("--api_key", help="API key for the model")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize MAC-SQL
    mac_sql = MACSQL(
        model_name=args.model,
        api_key=args.api_key,
    )
    
    # Handle evaluation mode
    if args.evaluate:
        print("Running evaluation...")
        run_evaluation(mac_sql, args)
        return
    
    # Handle direct query mode
    if args.question:
        # Connect to database if specified
        if args.db:
            success = mac_sql.connect_to_database(args.db)
            if not success:
                print(f"Failed to connect to database: {args.db}")
                return
        
        # Process the query
        result = mac_sql.query(args.question, verbose=args.verbose)
        
        # Print the results
        print(f"\nSQL Query: {result['sql_query']}")
        print("\nQuery Result:")
        print(result["query_result"])
    else:
        # Interactive mode
        if not args.db:
            print("No database specified. Please use --db to specify a database.")
            return
        
        # Connect to database
        success = mac_sql.connect_to_database(args.db)
        if not success:
            print(f"Failed to connect to database: {args.db}")
            return
        
        print(f"Connected to database: {args.db}")
        print("Enter 'exit' to quit.")
        
        # Interactive query loop
        while True:
            question = input("\nEnter your question: ")
            if question.lower() in ["exit", "quit", "q"]:
                break
            
            result = mac_sql.query(question, verbose=args.verbose)
            
            print(f"\nSQL Query: {result['sql_query']}")
            print("\nQuery Result:")
            print(result["query_result"])

if __name__ == "__main__":
    main() 