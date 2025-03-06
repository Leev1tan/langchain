import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mac_sql_agent_sqlite import MACSQLAgent, results_are_equivalent

# Available databases in the mini-bird benchmark
AVAILABLE_DBS = [
    "card_games",
    "california_schools",
    "superhero",
    "student_club",
    "toxicology",
    "thrombosis_prediction",
    "codebase_community",
    "debit_card_specializing",
    "european_football_2",
    "formula_1"
]

def load_benchmark_data(benchmark_path):
    """Load benchmark data from JSON file"""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def filter_benchmark_by_databases(benchmark_data, databases):
    """Filter benchmark data to only include specific databases"""
    return [item for item in benchmark_data if item.get('db_id') in databases]

def group_benchmark_by_database(benchmark_data):
    """Group benchmark questions by database"""
    grouped = {}
    for item in benchmark_data:
        db_id = item.get('db_id')
        if db_id not in grouped:
            grouped[db_id] = []
        grouped[db_id].append(item)
    return grouped

def evaluate_benchmark(agent, benchmark_items, output_file=None, rate_limit_delay=0):
    """
    Evaluate the agent on benchmark items with rate limit handling
    
    Args:
        agent: The MAC-SQL agent
        benchmark_items: List of benchmark items to evaluate
        output_file: Path to save results
        rate_limit_delay: Delay between queries to avoid rate limits
    
    Returns:
        List of evaluation results
    """
    results = []
    
    for item in tqdm(benchmark_items, desc="Evaluating"):
        question = item.get('question')
        gold_sql = item.get('query')
        db_id = item.get('db_id')
        
        # Skip if database doesn't match agent's database
        if db_id != agent.db_name:
            continue
        
        print(f"\nEvaluating question: {question}")
        print(f"Database: {db_id}")
        
        try:
            # Generate SQL
            generated_sql, understanding, plan, verification = agent.generate_sql(question)
            
            # Execute generated SQL
            generated_results, generated_error = agent.execute_sql_query(generated_sql)
            
            # Execute gold SQL
            gold_results, gold_error = agent.execute_sql_query(gold_sql)
            
            # Check if results match
            results_match = results_are_equivalent(generated_results, gold_results)
            
            result = {
                "question": question,
                "db_id": db_id,
                "generated_sql": generated_sql,
                "gold_sql": gold_sql,
                "results_match": results_match,
                "generated_error": generated_error,
                "gold_error": gold_error,
                "understanding": understanding,
                "plan": plan,
                "verification": verification
            }
            
            results.append(result)
            
            print(f"Results match: {results_match}")
            print(f"Generated SQL: {generated_sql}")
            print(f"Gold SQL: {gold_sql}")
            
            # Add delay to avoid rate limits
            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)
                
        except Exception as e:
            print(f"Error evaluating question: {e}")
            results.append({
                "question": question,
                "db_id": db_id,
                "error": str(e),
                "results_match": False
            })
    
    # Save results if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    return results

def print_results_summary(results):
    """Print a summary of evaluation results"""
    if not results:
        print("No results to summarize")
        return
    
    # Group results by database
    db_results = {}
    for result in results:
        db_id = result.get('db_id')
        if db_id not in db_results:
            db_results[db_id] = []
        db_results[db_id].append(result)
    
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total questions evaluated: {len(results)}")
    
    # Overall accuracy
    correct = sum(1 for r in results if r.get('results_match', False))
    accuracy = correct / len(results) if results else 0
    print(f"Overall accuracy: {correct}/{len(results)} ({accuracy:.2%})")
    
    # Accuracy by database
    print("\nAccuracy by database:")
    for db_id, db_result in db_results.items():
        db_correct = sum(1 for r in db_result if r.get('results_match', False))
        db_accuracy = db_correct / len(db_result) if db_result else 0
        print(f"  {db_id}: {db_correct}/{len(db_result)} ({db_accuracy:.2%})")
    
    # Error analysis
    generated_errors = sum(1 for r in results if r.get('generated_error'))
    gold_errors = sum(1 for r in results if r.get('gold_error'))
    print(f"\nGenerated queries with errors: {generated_errors}/{len(results)} ({generated_errors/len(results):.2%})")
    print(f"Gold queries with errors: {gold_errors}/{len(results)} ({gold_errors/len(results):.2%})")

def create_visualizations(results, output_prefix=None):
    """Create visualizations of evaluation results"""
    if not results:
        print("No results to visualize")
        return
    
    # Group results by database
    db_results = {}
    for result in results:
        db_id = result.get('db_id')
        if db_id not in db_results:
            db_results[db_id] = []
        db_results[db_id].append(result)
    
    # Create accuracy by database bar chart
    plt.figure(figsize=(12, 6))
    db_names = []
    db_accuracies = []
    
    for db_id, db_result in db_results.items():
        db_correct = sum(1 for r in db_result if r.get('results_match', False))
        db_accuracy = db_correct / len(db_result) if db_result else 0
        db_names.append(db_id)
        db_accuracies.append(db_accuracy)
    
    plt.bar(db_names, db_accuracies)
    plt.xlabel('Database')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Database')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_accuracy_by_db.png")
    else:
        plt.show()
    
    # Create error analysis pie chart
    plt.figure(figsize=(10, 6))
    error_categories = {
        'Correct': sum(1 for r in results if r.get('results_match', False)),
        'Generated SQL Error': sum(1 for r in results if r.get('generated_error') and not r.get('results_match', False)),
        'Incorrect Results': sum(1 for r in results if not r.get('generated_error') and not r.get('results_match', False))
    }
    
    plt.pie(error_categories.values(), labels=error_categories.keys(), autopct='%1.1f%%')
    plt.title('Error Analysis')
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_error_analysis.png")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate MAC-SQL Agent on mini-bird benchmark")
    parser.add_argument("--benchmark", default="dev_20240627/dev.json", help="Path to benchmark file")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model to use")
    parser.add_argument("--db", action="append", help="Specific databases to evaluate (defaults to all)")
    parser.add_argument("--samples_per_db", type=int, default=5, help="Number of samples to evaluate per database")
    parser.add_argument("--output", default="results/eval_results_sqlite.json", help="Path to save results")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--delay", type=float, default=0, help="Delay between queries to avoid rate limits (seconds)")
    
    args = parser.parse_args()
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(args.benchmark)
    print(f"Loaded {len(benchmark_data)} benchmark items")
    
    # Filter by databases if specified
    if args.db:
        benchmark_data = filter_benchmark_by_databases(benchmark_data, args.db)
        print(f"Filtered to {len(benchmark_data)} items for databases: {', '.join(args.db)}")
    
    # Group by database
    grouped_data = group_benchmark_by_database(benchmark_data)
    
    # Sample items from each database
    sampled_data = []
    for db_id, items in grouped_data.items():
        if args.samples_per_db > 0:
            # Sample randomly if samples_per_db is specified
            sample_size = min(args.samples_per_db, len(items))
            db_samples = random.sample(items, sample_size)
        else:
            # Use all items if samples_per_db is 0 or negative
            db_samples = items
        
        sampled_data.extend(db_samples)
        print(f"Selected {len(db_samples)} samples from {db_id}")
    
    # Evaluate each database separately
    all_results = []
    
    for db_id in sorted(grouped_data.keys()):
        if args.db and db_id not in args.db:
            continue
        
        print(f"\n===== Evaluating database: {db_id} =====")
        
        # Filter samples for this database
        db_samples = [item for item in sampled_data if item.get('db_id') == db_id]
        
        if not db_samples:
            print(f"No samples for database {db_id}")
            continue
        
        # Initialize agent for this database
        agent = MACSQLAgent(args.model, db_name=db_id)
        
        # Evaluate
        db_output = args.output.replace('.json', f'_{db_id}.json')
        db_results = evaluate_benchmark(agent, db_samples, db_output, args.delay)
        
        # Print summary
        print_results_summary(db_results)
        
        # Create visualizations
        if args.visualize:
            output_prefix = args.output.replace('.json', f'_{db_id}')
            create_visualizations(db_results, output_prefix)
        
        all_results.extend(db_results)
    
    # Save combined results
    if all_results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nSaved combined results to {args.output}")
        
        # Print overall summary
        print("\n===== OVERALL EVALUATION SUMMARY =====")
        print_results_summary(all_results)
        
        # Create overall visualizations
        if args.visualize:
            output_prefix = args.output.replace('.json', '')
            create_visualizations(all_results, output_prefix)

if __name__ == "__main__":
    main() 