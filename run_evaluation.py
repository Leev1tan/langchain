#!/usr/bin/env python

"""
Run evaluation for MAC-SQL on BIRD benchmark
"""

import argparse
import os
from mac_sql import MACSQL

def main():
    parser = argparse.ArgumentParser(description="Run MAC-SQL evaluation on BIRD benchmark")
    parser.add_argument("--benchmark", default="minidev/MINIDEV/mini_dev_postgresql.json", 
                      help="Path to benchmark JSON file")
    parser.add_argument("--num_samples", type=int, default=5, 
                      help="Number of samples per database to evaluate")
    parser.add_argument("--output_file", default="results/bird_evaluation.json", 
                      help="Path to save evaluation results")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", 
                      help="Model to use")
    parser.add_argument("--api_key", help="API key for the LLM service")
    parser.add_argument("--verbose", action="store_true", 
                      help="Print detailed processing steps")
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize MAC-SQL
    print(f"Initializing MAC-SQL with model: {args.model}")
    mac_sql = MACSQL(
        model_name=args.model,
        api_key=args.api_key,
        verbose=args.verbose
    )
    
    # Run evaluation
    print(f"Running evaluation on benchmark: {args.benchmark}")
    print(f"Number of samples per database: {args.num_samples}")
    
    results = mac_sql.evaluate_benchmark(
        benchmark_file=args.benchmark,
        num_samples=args.num_samples,
        output_file=args.output_file
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"  Total Items: {results.get('total_items', 0)}")
    print(f"  SQL Match Rate: {results.get('sql_match_count', 0)}/{results.get('total_items', 0)} "
          f"({results.get('sql_match_rate', 0):.2%})")
    print(f"  Execution Success Rate: {results.get('execution_success_count', 0)}/{results.get('total_items', 0)} "
          f"({results.get('execution_success_rate', 0):.2%})")
    print(f"  Average Similarity Score: {results.get('avg_similarity', 0):.4f}")
    print(f"  Results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 