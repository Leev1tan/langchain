#!/usr/bin/env python
"""
MAC-SQL Evaluation Script
========================

This script evaluates the MAC-SQL framework on benchmark datasets.

Usage:
    python evaluate.py --benchmark <benchmark_file> --samples_per_db <num_samples> --output <output_file>
"""

import os
import json
import argparse
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import traceback
import csv
from datetime import datetime
from collections import defaultdict

from mac_sql import MACSQL

def load_benchmark_data(benchmark_path):
    """Load benchmark data from JSON file"""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def filter_benchmark_by_databases(benchmark_data, databases):
    """Filter benchmark data to only include specific databases"""
    return [item for item in benchmark_data if item.get('db_id', '').lower() in [db.lower() for db in databases]]

def group_benchmark_by_database(benchmark_data):
    """Group benchmark questions by database"""
    grouped = {}
    for item in benchmark_data:
        db_id = item.get('db_id', '').lower()
        if db_id not in grouped:
            grouped[db_id] = []
        grouped[db_id].append(item)
    return grouped

def sample_benchmark_items(grouped_benchmark, samples_per_db=5):
    """Sample a specified number of items from each database"""
    sampled_items = []
    
    for db_id, items in grouped_benchmark.items():
        # Sample up to samples_per_db items (or all items if fewer)
        sample_size = min(samples_per_db, len(items))
        sampled = random.sample(items, sample_size)
        sampled_items.extend(sampled)
    
    return sampled_items

def categorize_query_complexity(query):
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

def categorize_query_type(query):
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

def print_results_summary(results):
    """
    Print a summary of evaluation results
    
    Args:
        results: Dictionary with evaluation results
    """
    if not results:
        print("No results to display")
        return
        
    print("\n===== MAC-SQL Evaluation Results =====")
    print(f"Model: {results.get('model', 'Unknown')}")
    print(f"Execution Accuracy: {results.get('execution_accuracy', 0.0):.2f}%")
    print(f"Total correct: {results.get('correct', 0)} / {results.get('total', 0)}")
    
    if 'error_count' in results:
        print(f"Errors: {results['error_count']}/{results['total']}")
    
    if 'avg_execution_time' in results:
        print(f"Average execution time: {results['avg_execution_time']:.2f} seconds")
    
    # Print results by complexity if available
    if 'results_by_complexity' in results:
        print("\nResults by query complexity:")
        for complexity, data in results['results_by_complexity'].items():
            print(f"- {complexity}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")
    
    # Print results by query type if available
    if 'results_by_type' in results:
        print("\nResults by query type:")
        for query_type, data in results['results_by_type'].items():
            print(f"- {query_type}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")
    
    # Print results by database if available
    if 'results_by_database' in results:
        print("\nResults by database:")
        for db_id, db_result in results['results_by_database'].items():
            print(f"- {db_id}: {db_result['accuracy']:.2f}% ({db_result['correct']}/{db_result['total']})")
    
    # Print detailed results if available and requested
    if 'detailed_results' in results and len(results['detailed_results']) < 10:
        print("\nDetailed results:")
        for item in results['detailed_results']:
            db_id = item.get('db_id', 'Unknown')
            question = item.get('question', 'Unknown')
            results_match = item.get('results_match', False)
            exec_time = item.get('execution_time', 0)
            status = "✓" if results_match else "✗"
            question_short = question[:50] + "..." if len(question) > 50 else question
            print(f"{status} [{exec_time:.2f}s] {db_id}: {question_short}")
    
    if 'error' in results:
        print(f"\nError in evaluation: {results['error']}")

def create_visualizations(results, output_dir="results"):
    """
    Create visualizations of evaluation results
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Create a DataFrame from detailed results
        if 'detailed_results' not in results or not results['detailed_results']:
            print("No detailed results available for visualization")
            return
            
        df = pd.DataFrame(results['detailed_results'])
        
        # Ensure required columns exist
        if 'results_match' not in df.columns:
            print("Results data does not contain 'results_match' column")
            return
            
        # Convert to boolean and fill NaN
        df['success'] = df['results_match'].fillna(False).astype(bool)
        
        # Overall success rate pie chart
        plt.figure(figsize=(8, 6))
        success_count = df['success'].sum()
        failure_count = len(df) - success_count
        plt.pie(
            [success_count, failure_count], 
            labels=['Success', 'Failure'], 
            autopct='%1.1f%%', 
            colors=['#66b3ff', '#ff9999'],
            explode=(0.1, 0)
        )
        plt.title(f'Overall Success Rate: {results.get("execution_accuracy", 0):.2f}%')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overall_success_{timestamp}.png'), dpi=300)
        
        # Success rate by database
        if 'db_id' in df.columns:
            plt.figure(figsize=(12, 6))
            db_success = df.groupby('db_id')['success'].agg(['mean', 'count']).reset_index()
            db_success.columns = ['Database', 'Success Rate', 'Count']
            
            # Sort by success rate
            db_success = db_success.sort_values('Success Rate', ascending=False)
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            bars = plt.bar(db_success['Database'], db_success['Success Rate'] * 100, color='skyblue')
            
            # Add data labels
            for bar, rate, count in zip(bars, db_success['Success Rate'], db_success['Count']):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{rate*100:.1f}% (n={count})",
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontsize=8
                )
            
            plt.title('Success Rate by Database')
            plt.xlabel('Database')
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 105)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'success_by_db_{timestamp}.png'), dpi=300)
        
        # Success rate by complexity
        if 'complexity' in df.columns:
            plt.figure(figsize=(8, 6))
            complexity_success = df.groupby('complexity')['success'].agg(['mean', 'count']).reset_index()
            complexity_success.columns = ['Complexity', 'Success Rate', 'Count']
            
            # Ensure consistent ordering
            order = ['simple', 'medium', 'complex']
            complexity_success['Complexity'] = pd.Categorical(
                complexity_success['Complexity'], 
                categories=order, 
                ordered=True
            )
            complexity_success = complexity_success.sort_values('Complexity')
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(complexity_success['Complexity'], complexity_success['Success Rate'] * 100, color='lightgreen')
            
            # Add data labels
            for bar, rate, count in zip(bars, complexity_success['Success Rate'], complexity_success['Count']):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{rate*100:.1f}% (n={count})",
                    ha='center',
                    va='bottom'
                )
            
            plt.title('Success Rate by Query Complexity')
            plt.xlabel('Complexity')
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 105)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'success_by_complexity_{timestamp}.png'), dpi=300)
        
        # Execution time boxplot
        if 'execution_time' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.boxplot([df[df['success'] == True]['execution_time'], 
                        df[df['success'] == False]['execution_time']], 
                        labels=['Successful Queries', 'Failed Queries'])
            plt.title('Execution Time Distribution')
            plt.ylabel('Time (seconds)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'execution_time_{timestamp}.png'), dpi=300)
        
        print(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()

def export_results_to_csv(results, output_file="results/evaluation_results.csv"):
    """
    Export evaluation results to CSV
    
    Args:
        results: Dictionary with evaluation results
        output_file: Path to save CSV file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        if 'detailed_results' not in results or not results['detailed_results']:
            print("No detailed results to export")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(results['detailed_results'])
        
        # Add timestamp
        df['evaluation_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add model name
        df['model'] = results.get('model', 'Unknown')
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")
        
    except Exception as e:
        print(f"Error exporting results to CSV: {e}")

def analyze_errors(results):
    """
    Analyze error patterns in evaluation results
    
    Args:
        results: Dictionary with evaluation results
        
    Returns:
        Dictionary with error analysis
    """
    if 'detailed_results' not in results or not results['detailed_results']:
        return {"error": "No detailed results available for analysis"}
    
    error_items = [item for item in results['detailed_results'] 
                  if not item.get('results_match', False)]
    
    if not error_items:
        return {"error_count": 0, "message": "No errors found"}
    
    # Error counts by database
    errors_by_db = defaultdict(int)
    for item in error_items:
        db_id = item.get('db_id', 'Unknown')
        errors_by_db[db_id] += 1
    
    # Error counts by query complexity
    errors_by_complexity = defaultdict(int)
    for item in error_items:
        complexity = item.get('complexity', 'Unknown')
        errors_by_complexity[complexity] += 1
    
    # Common error messages
    error_messages = [item.get('error', '') for item in error_items if 'error' in item]
    common_errors = defaultdict(int)
    for msg in error_messages:
        # Get first line of error or first 50 chars
        short_msg = msg.split('\n')[0][:50] if msg else "Unknown error"
        common_errors[short_msg] += 1
    
    return {
        "error_count": len(error_items),
        "errors_by_database": dict(errors_by_db),
        "errors_by_complexity": dict(errors_by_complexity),
        "common_errors": dict(sorted(common_errors.items(), key=lambda x: x[1], reverse=True)[:5])
    }

def run_evaluation(
    benchmark_file, 
    output_file="results/evaluation_results.json", 
    num_samples=2,
    visualize=True,
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Default model
    api_key=None
):
    """
    Run evaluation on benchmark data
    
    Args:
        benchmark_file: Path to benchmark JSON file
        output_file: Path to save evaluation results
        num_samples: Number of samples per database to evaluate
        visualize: Whether to visualize results
        model_name: Name of the LLM model to use
        api_key: API key for the LLM service
    """
    print(f"Loading benchmark data from {benchmark_file}...")
    
    try:
        # Create results directory
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # Load benchmark data
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        # Get available databases
        db_ids = set()
        for item in benchmark_data:
            db_id = item.get("db_id", "").lower()
            if db_id:
                db_ids.add(db_id)
        
        print(f"Available databases: {', '.join(sorted(db_ids))}")
        
        # Initialize MAC-SQL
        mac_sql = MACSQL(model_name=model_name, api_key=api_key)
        
        # Run evaluation
        print("Evaluating MAC-SQL on benchmark data...")
        results = mac_sql.evaluate_benchmark(
            benchmark_file=benchmark_file,
            num_samples=num_samples,
            output_file=output_file
        )
        
        # Add additional analysis
        if results and 'detailed_results' in results:
            # Calculate additional metrics
            detailed_results = results['detailed_results']
            
            # Add complexity categorization
            for item in detailed_results:
                if 'gold_sql' in item:
                    item['complexity'] = categorize_query_complexity(item['gold_sql'])
                    item['query_type'] = categorize_query_type(item['gold_sql'])
            
            # Results by complexity
            results_by_complexity = defaultdict(lambda: {"correct": 0, "total": 0})
            for item in detailed_results:
                complexity = item.get('complexity', 'unknown')
                results_by_complexity[complexity]["total"] += 1
                if item.get('results_match', False):
                    results_by_complexity[complexity]["correct"] += 1
            
            # Calculate accuracy for each complexity
            for complexity, data in results_by_complexity.items():
                if data["total"] > 0:
                    data["accuracy"] = (data["correct"] / data["total"]) * 100
                else:
                    data["accuracy"] = 0
            
            results['results_by_complexity'] = dict(results_by_complexity)
            
            # Results by query type
            results_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
            for item in detailed_results:
                query_type = item.get('query_type', 'unknown')
                results_by_type[query_type]["total"] += 1
                if item.get('results_match', False):
                    results_by_type[query_type]["correct"] += 1
            
            # Calculate accuracy for each query type
            for query_type, data in results_by_type.items():
                if data["total"] > 0:
                    data["accuracy"] = (data["correct"] / data["total"]) * 100
                else:
                    data["accuracy"] = 0
            
            results['results_by_type'] = dict(results_by_type)
            
            # Calculate average execution time
            execution_times = [item.get('execution_time', 0) for item in detailed_results]
            if execution_times:
                results['avg_execution_time'] = sum(execution_times) / len(execution_times)
            
            # Analyze errors
            results['error_analysis'] = analyze_errors(results)
            
            # Save the updated results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        if visualize:
            # Create visualizations
            create_visualizations(results)
        
        # Export to CSV
        csv_output = output_file.replace('.json', '.csv')
        export_results_to_csv(results, csv_output)
        
        return results
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate MAC-SQL on benchmark datasets")
    parser.add_argument("--benchmark", default="minidev/MINIDEV/mini_dev_postgresql.json",
                       help="Path to benchmark file (JSON)")
    parser.add_argument("--samples-per-db", type=int, default=2,
                       help="Number of samples per database")
    parser.add_argument("--output", default="results/evaluation_results.json",
                       help="Path to save evaluation results")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                       help="Model name")
    parser.add_argument("--api-key", default=None,
                       help="API key for LLM service")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualizations of results")
    parser.add_argument("--db", nargs="+", default=None,
                       help="Filter to specific databases")
    parser.add_argument("--export-csv", action="store_true",
                       help="Export results to CSV")
    args = parser.parse_args()
    
    # Ensure output directory exists
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Run evaluation
    results = run_evaluation(
        args.benchmark,
        args.output,
        args.samples_per_db,
        args.visualize,
        args.model,
        args.api_key
    )
    
    # Print summary
    if results:
        print_results_summary(results)

if __name__ == "__main__":
    main() 