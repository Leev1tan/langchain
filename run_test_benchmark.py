#!/usr/bin/env python
"""
Script to run a benchmark test using the test databases
"""

import os
import json
import time
import tabulate
from core.chat_manager import ChatManager

# Test queries for each database
TEST_QUERIES = {
    "student_club": [
        "How many members are in the club?",
        "List all events that happened in 2023",
        "Who attended the Annual Meeting?",
        "How many members registered for the Summer Picnic?",
        "What is the most popular event based on attendance?"
    ],
    "formula_1": [
        "Who won the Monaco Grand Prix?",
        "How many points did Lewis Hamilton score in total?",
        "Which driver has the most wins?",
        "List all races where a driver did not finish (DNF)",
        "What is the average position of Charles Leclerc across all races?"
    ]
}

def format_table(columns, rows):
    """Format results as a table"""
    if not columns or not rows:
        return "No results"
    
    return tabulate.tabulate(rows, headers=columns, tablefmt="grid")

def run_benchmark():
    """
    Run a benchmark test using the test databases
    """
    print("Running benchmark test using test databases...")
    
    # Initialize ChatManager
    chat_manager = ChatManager(verbose=True)
    
    results = {}
    summary = {"total": 0, "success": 0, "with_results": 0, "empty_results": 0, "failed": 0}
    
    # Test each database
    for db_name, queries in TEST_QUERIES.items():
        print(f"\n{'-'*80}")
        print(f"Testing database: {db_name}")
        print(f"{'-'*80}")
        
        # Connect to database
        success = chat_manager.connect_to_database(db_name)
        if not success:
            print(f"Failed to connect to database: {db_name}")
            continue
        
        db_results = []
        
        # Run each query
        for i, query in enumerate(queries):
            summary["total"] += 1
            
            print(f"\n{'-'*40}")
            print(f"Query {i+1}: {query}")
            print(f"{'-'*40}")
            
            # Record start time
            start_time = time.time()
            
            # Process query
            result = chat_manager.process_query(query)
            
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Print result
            print(f"\nSQL: {result.get('sql', 'No SQL generated')}")
            print(f"Execution time: {execution_time:.2f} seconds")
            
            # Check if execution was successful
            if result.get('success', False):
                summary["success"] += 1
                print("Query executed successfully")
                
                # Check if we got any results
                rows = result.get('results', {}).get('rows', [])
                columns = result.get('results', {}).get('columns', [])
                
                if rows:
                    summary["with_results"] += 1
                    print(f"Results ({len(rows)} rows):")
                    print(format_table(columns, rows))
                else:
                    summary["empty_results"] += 1
                    print("Query returned no results")
            else:
                summary["failed"] += 1
                print(f"Query execution failed: {result.get('error', 'Unknown error')}")
            
            # Add to results
            db_results.append({
                "query": query,
                "sql": result.get('sql', ''),
                "success": result.get('success', False),
                "has_results": bool(result.get('results', {}).get('rows', [])),
                "error": result.get('error', ''),
                "execution_time": execution_time,
                "understanding": result.get('understanding', ''),
                "plan": result.get('plan', '')
            })
        
        # Add database results to overall results
        results[db_name] = db_results
    
    # Save results to file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'-'*80}")
    print("Benchmark Summary:")
    print(f"{'-'*80}")
    print(f"Total queries: {summary['total']}")
    print(f"Successful: {summary['success']} ({summary['success']/summary['total']*100:.1f}%)")
    print(f"  - With results: {summary['with_results']} ({summary['with_results']/summary['total']*100:.1f}%)")
    print(f"  - Empty results: {summary['empty_results']} ({summary['empty_results']/summary['total']*100:.1f}%)")
    print(f"Failed: {summary['failed']} ({summary['failed']/summary['total']*100:.1f}%)")
    
    print("\nBenchmark test complete. Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark() 