#!/usr/bin/env python
"""
BIRD Benchmark Evaluation Script for MAC-SQL

This script evaluates the MAC-SQL framework on the BIRD benchmark.
It loads benchmark data from a JSON file and evaluates the framework
on a specified number of samples per database.

All tables are stored in a single PostgreSQL database called 'BIRD'.

Usage:
    python evaluate_bird.py [--benchmark BENCHMARK_FILE] [--samples-per-db SAMPLES] [--db DB_NAMES [DB_NAMES ...]] [--visualize]
"""

import os
import json
import time
import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from mac_sql import MACSQL
from core.config import DB_CONFIG

# Constants
DEFAULT_BENCHMARK_PATH = "minidev/MINIDEV/mini_dev_postgresql.json"
DEFAULT_SAMPLES_PER_DB = 5
RESULTS_DIR = "results"
BIRD_DATABASE = "BIRD"  # Name of the main BIRD database

def load_benchmark_data(benchmark_path: str) -> List[Dict[str, Any]]:
    """Load benchmark data from JSON file"""
    try:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return []

def get_database_ids(benchmark_data: List[Dict[str, Any]]) -> List[str]:
    """Extract unique database IDs from benchmark data"""
    db_ids = set()
    for item in benchmark_data:
        db_id = item.get("db_id", "")
        if db_id:
            db_ids.add(db_id)
    return sorted(list(db_ids))

def filter_benchmark_data(benchmark_data: List[Dict[str, Any]], db_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter benchmark data based on database IDs"""
    if not db_filter:
        return benchmark_data
    return [item for item in benchmark_data if item.get("db_id", "").lower() in [db.lower() for db in db_filter]]

def sample_benchmark_data(benchmark_data: List[Dict[str, Any]], samples_per_db: int) -> List[Dict[str, Any]]:
    """Sample benchmark data based on database IDs"""
    # Group by database ID
    db_groups = {}
    for item in benchmark_data:
        db_id = item.get("db_id", "")
        if db_id:
            if db_id not in db_groups:
                db_groups[db_id] = []
            db_groups[db_id].append(item)
    
    # Sample from each group
    sampled_data = []
    for db_id, items in db_groups.items():
        # Take at most samples_per_db items from each database
        sample_size = min(samples_per_db, len(items))
        sampled_data.extend(random.sample(items, sample_size))
    
    return sampled_data

def create_results_dir():
    """Create results directory if it doesn't exist"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_mac_sql(benchmark_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate MAC-SQL on benchmark data"""
    # Initialize MAC-SQL with default model
    mac_sql = MACSQL()
    print(f"Initializing MAC-SQL with model: {mac_sql.model_name}")
    
    # Connect to the BIRD database (which contains all tables)
    mac_sql.connect_to_database(BIRD_DATABASE)
    
    results = {
        "model": mac_sql.model_name,
        "total": len(benchmark_data),
        "execution_success": 0,  # Queries that execute without errors
        "semantic_correct": 0,   # Queries that return correct results
        "execution_errors": 0,
        "execution_times": [],
        "detailed_results": []
    }
    
    # Process each benchmark item
    for i, item in enumerate(tqdm(benchmark_data, desc="Evaluating queries")):
        db_id = item.get("db_id", "")
        question_id = item.get("question_id", i)
        question = item.get("question", "")
        gold_sql = item.get("SQL", "")
        difficulty = item.get("difficulty", "unknown")
        hardness = item.get("hardness", "unknown")
        
        print(f"\nEvaluating question {question_id} on database: {db_id}")
        print(f"Question: {question}")
        print(f"Gold SQL: {gold_sql}")
        
        # We're already connected to the BIRD database which contains all tables
        # Just need to make sure the schema knowledge is properly initialized for the current context
        
        # Process the query
        start_time = time.time()
        try:
            # We're using the same BIRD database, but we need to tell the system which 
            # database context we're evaluating for query understanding
            sql, results_df = mac_sql.process_query(question, db_context=db_id)
            execution_time = time.time() - start_time
            
            # Mark as execution success
            execution_success = True
            error_msg = None
            
            # Store the generated results
            generated_results = {
                'success': True,
                'sql': sql,
                'results': results_df
            }
        except Exception as e:
            execution_time = time.time() - start_time
            execution_success = False
            error_msg = str(e)
            sql = None
            results_df = None
            generated_results = {
                'success': False,
                'sql': sql,
                'error': error_msg
            }
        
        # Execute gold SQL to get ground truth results
        gold_results = None
        if gold_sql:
            try:
                # Use the chat manager to execute the gold SQL
                gold_query_result = mac_sql.chat_manager.execute_sql_query(gold_sql)
                if gold_query_result.get('success', False):
                    # Convert to DataFrame for comparison
                    if 'rows' in gold_query_result and 'columns' in gold_query_result:
                        gold_df = pd.DataFrame(gold_query_result['rows'], columns=gold_query_result['columns'])
                        gold_results = {
                            'success': True,
                            'sql': gold_sql,
                            'results': gold_df
                        }
                    else:
                        gold_results = {
                            'success': True,
                            'sql': gold_sql,
                            'results': pd.DataFrame()
                        }
                else:
                    gold_results = {
                        'success': False,
                        'sql': gold_sql,
                        'error': gold_query_result.get('error', 'Unknown error executing gold SQL')
                    }
            except Exception as e:
                gold_results = {
                    'success': False,
                    'sql': gold_sql,
                    'error': str(e)
                }
        
        # Determine if results match (semantic correctness)
        semantic_correct = False
        comparison_notes = ""
        
        if execution_success and gold_results and gold_results['success']:
            semantic_correct, comparison_notes = compare_query_results(results_df, gold_results['results'])
        
        # Record detailed results
        detail = {
            "db_id": db_id,
            "question_id": question_id,
            "question": question,
            "gold_sql": gold_sql,
            "generated_sql": sql,
            "execution_success": execution_success,
            "semantic_correct": semantic_correct,
            "comparison_notes": comparison_notes,
            "error_message": error_msg,
            "execution_time": execution_time,
            "difficulty": difficulty,
            "hardness": hardness
        }
        
        results["detailed_results"].append(detail)
        results["execution_times"].append(execution_time)
        
        if execution_success:
            results["execution_success"] += 1
            print(f"Generated SQL: {sql}")
            print(f"Execution Success: True")
        else:
            results["execution_errors"] += 1
            print(f"Query execution failed: {error_msg}")
            print(f"Execution Success: False")
        
        if semantic_correct:
            results["semantic_correct"] += 1
            print(f"Results match gold standard: True")
        else:
            print(f"Results match gold standard: False")
            if comparison_notes:
                print(f"Comparison notes: {comparison_notes}")
        
        print(f"Execution time: {execution_time:.2f}s")
    
    # Calculate accuracies
    results["execution_accuracy"] = results["execution_success"] / results["total"] if results["total"] > 0 else 0
    results["semantic_accuracy"] = results["semantic_correct"] / results["total"] if results["total"] > 0 else 0
    
    # Calculate results by database
    results["results_by_db"] = {}
    for detail in results["detailed_results"]:
        db_id = detail["db_id"]
        if db_id not in results["results_by_db"]:
            results["results_by_db"][db_id] = {
                "total": 0,
                "execution_success": 0,
                "semantic_correct": 0,
                "execution_accuracy": 0,
                "semantic_accuracy": 0
            }
        results["results_by_db"][db_id]["total"] += 1
        if detail["execution_success"]:
            results["results_by_db"][db_id]["execution_success"] += 1
        if detail["semantic_correct"]:
            results["results_by_db"][db_id]["semantic_correct"] += 1
    
    # Calculate accuracy for each database
    for db_id, db_result in results["results_by_db"].items():
        db_result["execution_accuracy"] = db_result["execution_success"] / db_result["total"] if db_result["total"] > 0 else 0
        db_result["semantic_accuracy"] = db_result["semantic_correct"] / db_result["total"] if db_result["total"] > 0 else 0
    
    # Calculate results by difficulty
    results["results_by_difficulty"] = {}
    for detail in results["detailed_results"]:
        difficulty = detail["difficulty"]
        if difficulty not in results["results_by_difficulty"]:
            results["results_by_difficulty"][difficulty] = {
                "total": 0,
                "execution_success": 0,
                "semantic_correct": 0,
                "execution_accuracy": 0,
                "semantic_accuracy": 0
            }
        results["results_by_difficulty"][difficulty]["total"] += 1
        if detail["execution_success"]:
            results["results_by_difficulty"][difficulty]["execution_success"] += 1
        if detail["semantic_correct"]:
            results["results_by_difficulty"][difficulty]["semantic_correct"] += 1
    
    # Calculate accuracy for each difficulty
    for difficulty, diff_result in results["results_by_difficulty"].items():
        diff_result["execution_accuracy"] = diff_result["execution_success"] / diff_result["total"] if diff_result["total"] > 0 else 0
        diff_result["semantic_accuracy"] = diff_result["semantic_correct"] / diff_result["total"] if diff_result["total"] > 0 else 0
    
    # Calculate results by hardness
    results["results_by_hardness"] = {}
    for detail in results["detailed_results"]:
        hardness = detail["hardness"]
        if hardness not in results["results_by_hardness"]:
            results["results_by_hardness"][hardness] = {
                "total": 0,
                "execution_success": 0,
                "semantic_correct": 0,
                "execution_accuracy": 0,
                "semantic_accuracy": 0
            }
        results["results_by_hardness"][hardness]["total"] += 1
        if detail["execution_success"]:
            results["results_by_hardness"][hardness]["execution_success"] += 1
        if detail["semantic_correct"]:
            results["results_by_hardness"][hardness]["semantic_correct"] += 1
    
    # Calculate accuracy for each hardness
    for hardness, hard_result in results["results_by_hardness"].items():
        hard_result["execution_accuracy"] = hard_result["execution_success"] / hard_result["total"] if hard_result["total"] > 0 else 0
        hard_result["semantic_accuracy"] = hard_result["semantic_correct"] / hard_result["total"] if hard_result["total"] > 0 else 0
    
    return results

def compare_query_results(generated_df: pd.DataFrame, gold_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Compare results from generated SQL and gold SQL to determine if they match
    
    Args:
        generated_df: DataFrame with results from generated SQL
        gold_df: DataFrame with results from gold SQL
        
    Returns:
        Tuple of (match_result, comparison_notes)
    """
    if generated_df is None or gold_df is None:
        return False, "One or both result sets are None"
    
    # Check if both are empty (this could be a valid match)
    if len(generated_df) == 0 and len(gold_df) == 0:
        return True, "Both result sets are empty"
    
    # Check row count match
    if len(generated_df) != len(gold_df):
        return False, f"Row count mismatch: generated={len(generated_df)}, gold={len(gold_df)}"
    
    # Handle single value result (common for COUNT, AVG, etc.)
    if len(generated_df) == 1 and len(gold_df) == 1 and len(generated_df.columns) == 1 and len(gold_df.columns) == 1:
        gen_value = generated_df.iloc[0, 0]
        gold_value = gold_df.iloc[0, 0]
        
        # Try numeric comparison with tolerance
        try:
            gen_num = float(gen_value) if gen_value is not None else None
            gold_num = float(gold_value) if gold_value is not None else None
            
            if gen_num is not None and gold_num is not None:
                # Use relative tolerance for large numbers
                if abs(gold_num) > 1.0:
                    relative_diff = abs(gen_num - gold_num) / abs(gold_num)
                    if relative_diff < 0.01:  # 1% tolerance
                        return True, f"Single numeric values match within tolerance: {gen_num} ≈ {gold_num}"
                # Use absolute tolerance for small numbers
                elif abs(gen_num - gold_num) < 0.001:
                    return True, f"Single numeric values match within tolerance: {gen_num} ≈ {gold_num}"
                else:
                    return False, f"Single numeric values don't match: {gen_num} != {gold_num}"
        except (ValueError, TypeError):
            # Non-numeric comparison
            if gen_value == gold_value:
                return True, f"Single values match exactly: {gen_value}"
            else:
                return False, f"Single values don't match: {gen_value} != {gold_value}"
    
    # Multi-row results: first try direct comparison
    if generated_df.equals(gold_df):
        return True, "DataFrames match exactly"
    
    # Check if columns match (may be in different order)
    gen_cols = set(generated_df.columns)
    gold_cols = set(gold_df.columns)
    
    if gen_cols != gold_cols:
        return False, f"Column names don't match: generated={gen_cols}, gold={gold_cols}"
    
    # For single column results, try sorting and comparing
    if len(gen_cols) == 1:
        gen_sorted = generated_df.sort_values(by=generated_df.columns[0]).reset_index(drop=True)
        gold_sorted = gold_df.sort_values(by=gold_df.columns[0]).reset_index(drop=True)
        
        if gen_sorted.equals(gold_sorted):
            return True, "Single column results match after sorting"
    
    # For multi-column results, try more sophisticated comparison
    # This is complex and may need to be customized based on the specific use case
    
    # As a fallback, check if sets of tuples match (ignoring order)
    try:
        gen_tuples = set(map(tuple, generated_df.values))
        gold_tuples = set(map(tuple, gold_df.values))
        
        if gen_tuples == gold_tuples:
            return True, "Results match as sets (ignoring row order)"
        else:
            return False, "Results don't match as sets"
    except:
        # If tuple conversion fails (e.g., due to unhashable types)
        return False, "Unable to compare results as sets"

def visualize_results(results: Dict[str, Any]):
    """Visualize evaluation results"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy by database
    db_names = list(results["results_by_db"].keys())
    db_exec_accuracy = [results["results_by_db"][db]["execution_accuracy"] * 100 for db in db_names]
    db_semantic_accuracy = [results["results_by_db"][db]["semantic_accuracy"] * 100 for db in db_names]
    
    x = np.arange(len(db_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, db_exec_accuracy, width, label='Execution')
    axes[0, 0].bar(x + width/2, db_semantic_accuracy, width, label='Semantic')
    axes[0, 0].set_title("Accuracy by Database")
    axes[0, 0].set_xlabel("Database")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(db_names, rotation=45, ha='right')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Accuracy by difficulty
    if "results_by_difficulty" in results:
        diff_names = list(results["results_by_difficulty"].keys())
        diff_exec_accuracy = [results["results_by_difficulty"][diff]["execution_accuracy"] * 100 for diff in diff_names]
        diff_semantic_accuracy = [results["results_by_difficulty"][diff]["semantic_accuracy"] * 100 for diff in diff_names]
        
        x = np.arange(len(diff_names))
        
        axes[0, 1].bar(x - width/2, diff_exec_accuracy, width, label='Execution')
        axes[0, 1].bar(x + width/2, diff_semantic_accuracy, width, label='Semantic')
        axes[0, 1].set_title("Accuracy by Complexity")
        axes[0, 1].set_xlabel("Complexity")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(diff_names)
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Accuracy by hardness
    if "results_by_hardness" in results:
        hard_names = list(results["results_by_hardness"].keys())
        hard_exec_accuracy = [results["results_by_hardness"][hard]["execution_accuracy"] * 100 for hard in hard_names]
        hard_semantic_accuracy = [results["results_by_hardness"][hard]["semantic_accuracy"] * 100 for hard in hard_names]
        
        x = np.arange(len(hard_names))
        
        axes[1, 0].bar(x - width/2, hard_exec_accuracy, width, label='Execution')
        axes[1, 0].bar(x + width/2, hard_semantic_accuracy, width, label='Semantic')
        axes[1, 0].set_title("Accuracy by Hardness")
        axes[1, 0].set_xlabel("Hardness")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(hard_names)
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: Overall comparison
    labels = ['Execution', 'Semantic']
    overall_accuracy = [results["execution_accuracy"] * 100, results["semantic_accuracy"] * 100]
    
    axes[1, 1].bar(labels, overall_accuracy)
    axes[1, 1].set_title("Overall Accuracy")
    axes[1, 1].set_xlabel("Metric")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylim(0, 100)
    for i, v in enumerate(overall_accuracy):
        axes[1, 1].text(i, v + 2, f"{v:.1f}%", ha='center')
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bird_results.png"))
    plt.close()

def export_results_to_csv(results: Dict[str, Any]):
    """Export detailed results to CSV"""
    df = pd.DataFrame(results["detailed_results"])
    df.to_csv(os.path.join(RESULTS_DIR, "bird_results.csv"), index=False)

def print_results_summary(results: Dict[str, Any]):
    """Print a summary of the evaluation results"""
    print("\n===== BIRD EVALUATION SUMMARY =====")
    print(f"Model: {results['model']}")
    print(f"Execution Accuracy: {results['execution_accuracy'] * 100:.2f}%")
    print(f"Semantic Accuracy: {results['semantic_accuracy'] * 100:.2f}%")
    print(f"Total queries: {results['total']}")
    print(f"Execution successes: {results['execution_success']} / {results['total']}")
    print(f"Semantically correct: {results['semantic_correct']} / {results['total']}")
    
    print("\nResults by Database:")
    for db, db_result in results["results_by_db"].items():
        print(f"  - {db}: Exec={db_result['execution_accuracy'] * 100:.2f}%, Semantic={db_result['semantic_accuracy'] * 100:.2f}% ({db_result['semantic_correct']}/{db_result['total']})")
    
    if "results_by_difficulty" in results:
        print("\nResults by Complexity:")
        for diff, diff_result in results["results_by_difficulty"].items():
            print(f"  - {diff}: Exec={diff_result['execution_accuracy'] * 100:.2f}%, Semantic={diff_result['semantic_accuracy'] * 100:.2f}% ({diff_result['semantic_correct']}/{diff_result['total']})")
    
    if "results_by_hardness" in results:
        print("\nResults by Difficulty:")
        for hard, hard_result in results["results_by_hardness"].items():
            print(f"  - {hard}: Exec={hard_result['execution_accuracy'] * 100:.2f}%, Semantic={hard_result['semantic_accuracy'] * 100:.2f}% ({hard_result['semantic_correct']}/{hard_result['total']})")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate MAC-SQL on BIRD benchmark")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_PATH,
                      help="Path to benchmark file (JSON)")
    parser.add_argument("--samples-per-db", type=int, default=DEFAULT_SAMPLES_PER_DB,
                      help="Number of samples per database")
    parser.add_argument("--db", nargs="+", default=None,
                      help="Filter to specific databases")
    parser.add_argument("--visualize", action="store_true",
                      help="Visualize evaluation results")
    args = parser.parse_args()
    
    # Create results directory
    create_results_dir()
    
    # Load benchmark data
    print(f"Loading BIRD benchmark data from {args.benchmark}...")
    benchmark_data = load_benchmark_data(args.benchmark)
    if not benchmark_data:
        print("No benchmark data loaded, aborting evaluation")
        return
    
    # Filter benchmark data if specified
    if args.db:
        print(f"Filtering to databases: {', '.join(args.db)}")
        benchmark_data = filter_benchmark_data(benchmark_data, args.db)
    
    # Get available databases
    db_ids = get_database_ids(benchmark_data)
    print(f"Available databases: {', '.join(db_ids)}")
    
    # Sample benchmark data
    benchmark_data = sample_benchmark_data(benchmark_data, args.samples_per_db)
    print(f"Sampled {len(benchmark_data)} items from {len(db_ids)} databases")
    
    # Evaluate MAC-SQL
    results = evaluate_mac_sql(benchmark_data)
    
    # Save results
    with open(os.path.join(RESULTS_DIR, "bird_evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Print results summary
    print_results_summary(results)
    
    # Export results to CSV
    export_results_to_csv(results)
    
    # Visualize results if specified
    if args.visualize:
        visualize_results(results)
        print("Visualizations saved to", RESULTS_DIR)
    
    print(f"Results exported to {os.path.join(RESULTS_DIR, 'bird_results.csv')}")

if __name__ == "__main__":
    main() 