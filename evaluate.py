import os
import json
import argparse
import psycopg2
import pandas as pd
from tqdm import tqdm
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mac_sql_agent import MACSQLAgent, adapt_sql_dialect, results_are_equivalent, DB_CONFIG

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
        agent: MACSQLAgent instance
        benchmark_items: List of benchmark items to evaluate
        output_file: Path to save results (optional)
        rate_limit_delay: Delay between queries to avoid rate limits
    
    Returns:
        Dictionary with evaluation results
    """
    correct = 0
    total = len(benchmark_items)
    results = []
    
    # Create a progress bar
    pbar = tqdm(total=total, desc="Evaluating queries")
    
    # Setup for partial results saving
    if output_file:
        temp_output_file = output_file.replace('.json', '_partial.json')
    
    for i, item in enumerate(benchmark_items):
        db_id = item.get('db_id')
        question = item.get('question')
        gold_sql = item.get('SQL')
        question_id = item.get('question_id', i)
        
        # Update agent database if needed
        if agent.db_config['dbname'] != db_id:
            agent.db_config['dbname'] = db_id
            # Reinitialize with new database
            agent.db_uri = f"postgresql://{agent.db_config['user']}:{agent.db_config['password']}@{agent.db_config['host']}:{agent.db_config['port']}/{agent.db_config['dbname']}"
            try:
                agent.db = SQLDatabase.from_uri(agent.db_uri)
                agent.initialize_schema_knowledge()
            except Exception as e:
                print(f"Database connection error for {db_id}: {e}")
                continue
        
        # Add result entry
        result_entry = {
            'db_id': db_id,
            'question_id': question_id,
            'question': question,
            'gold_sql': gold_sql
        }
        
        # Generate SQL with retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Add random delay to avoid rate limiting
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay + random.random() * 2)
                
                # Generate SQL for the question
                generated_sql = agent.generate_sql(question)
                result_entry['generated_sql'] = generated_sql
                
                # Try to execute both queries
                try:
                    # Adapt gold SQL if needed
                    adapted_gold_sql = adapt_sql_dialect(gold_sql, "mysql", "postgresql")
                    result_entry['adapted_gold_sql'] = adapted_gold_sql
                    
                    # Execute generated SQL
                    generated_results = agent.execute_sql_query(generated_sql)
                    
                    # Execute gold SQL
                    gold_results = agent.execute_sql_query(adapted_gold_sql)
                    
                    # Compare results
                    if isinstance(generated_results, pd.DataFrame) and isinstance(gold_results, pd.DataFrame):
                        results_match = results_are_equivalent(generated_results, gold_results)
                    else:
                        results_match = False
                    
                    if results_match:
                        correct += 1
                    
                    result_entry['results_match'] = results_match
                    
                except Exception as e:
                    result_entry['error'] = str(e)
                    result_entry['results_match'] = False
                
                # Success, break retry loop
                break
                
            except ValueError as e:
                # Handle rate limit errors
                error_message = str(e)
                retry_count += 1
                
                if "rate limit" in error_message.lower() and retry_count < max_retries:
                    # Exponential backoff for rate limit
                    wait_time = 5 * (2 ** retry_count)
                    tqdm.write(f"Rate limit hit, retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    # Other error or max retries reached
                    result_entry['error'] = error_message
                    break
        
        # Add result to results list
        results.append(result_entry)
        
        # Update progress bar
        pbar.update(1)
        
        # Save partial results every 5 items
        if output_file and i % 5 == 0 and i > 0:
            partial_results = {
                'execution_accuracy': correct / (i+1) if i+1 > 0 else 0,
                'detailed_results': results
            }
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(partial_results, f, indent=2, default=lambda x: None if isinstance(x, pd.DataFrame) else x)
            tqdm.write(f"Saved partial results ({i+1}/{total} complete)")
    
    pbar.close()
    
    # Calculate execution accuracy
    execution_accuracy = correct / total if total > 0 else 0
    
    final_results = {
        'execution_accuracy': execution_accuracy,
        'detailed_results': results
    }
    
    # Save final results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, default=lambda x: None if isinstance(x, pd.DataFrame) else x)
        print(f"Results saved to {output_file}")
    
    return final_results

def print_results_summary(results):
    """Print a summary of the evaluation results"""
    print(f"\n===== MAC-SQL Agent Evaluation Results =====")
    print(f"Execution Accuracy: {results['execution_accuracy']:.2%}")
    
    # Group by database
    db_results = {}
    for item in results['detailed_results']:
        db_id = item.get('db_id')
        if db_id not in db_results:
            db_results[db_id] = {'correct': 0, 'total': 0}
        
        db_results[db_id]['total'] += 1
        if item.get('results_match', False):
            db_results[db_id]['correct'] += 1
    
    print("\nResults by Database:")
    for db_id, counts in db_results.items():
        accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        print(f"  {db_id}: {accuracy:.2%} ({counts['correct']}/{counts['total']})")
    
    # Count of errors
    error_count = sum(1 for r in results['detailed_results'] if 'error' in r)
    print(f"\nQueries with errors: {error_count}/{len(results['detailed_results'])} ({error_count/len(results['detailed_results']):.2%})")

def create_visualizations(results, output_prefix=None):
    """Create visualizations of the evaluation results"""
    # Prepare data
    df = pd.DataFrame(results['detailed_results'])
    df['success'] = df['results_match'].fillna(False)
    
    # Overall success rate
    plt.figure(figsize=(8, 6))
    success_rate = df['success'].mean()
    plt.pie([success_rate, 1-success_rate], labels=['Success', 'Failure'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    plt.title('Overall Success Rate')
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_overall.png", dpi=300, bbox_inches='tight')
    
    # Success rate by database
    plt.figure(figsize=(12, 6))
    db_success = df.groupby('db_id')['success'].agg(['mean', 'count']).reset_index()
    db_success.columns = ['Database', 'Success Rate', 'Count']
    
    # Sort by success rate
    db_success = db_success.sort_values('Success Rate', ascending=False)
    
    ax = sns.barplot(x='Database', y='Success Rate', data=db_success)
    
    # Add count labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f"n={db_success['Count'].iloc[i]}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 10),
                   textcoords='offset points')
    
    plt.title('Success Rate by Database')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_prefix:
        plt.savefig(f"{output_prefix}_by_database.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate MAC-SQL Agent on mini-bird benchmark')
    parser.add_argument('--benchmark', default='dev_20240627/dev.json', help='Path to benchmark JSON file')
    parser.add_argument('--model', default='meta-llama/Llama-3.3-70B-Instruct-Turbo', help='Model to use')
    parser.add_argument('--api_key', help='Together API key')
    parser.add_argument('--db', nargs='+', choices=AVAILABLE_DBS, help='Specific databases to evaluate (default: all)')
    parser.add_argument('--samples_per_db', type=int, default=5, help='Number of samples per database')
    parser.add_argument('--output', default='results/eval_results.json', help='Output file for results')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between queries (seconds)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load benchmark data
    print(f"Loading benchmark data from {args.benchmark}...")
    benchmark_data = load_benchmark_data(args.benchmark)
    print(f"Loaded {len(benchmark_data)} benchmark examples")
    
    # Filter by databases if specified
    if args.db:
        print(f"Filtering to {len(args.db)} databases: {', '.join(args.db)}")
        benchmark_data = filter_benchmark_by_databases(benchmark_data, args.db)
    else:
        print(f"Using all available databases: {', '.join(AVAILABLE_DBS)}")
        benchmark_data = filter_benchmark_by_databases(benchmark_data, AVAILABLE_DBS)
    
    print(f"Filtered to {len(benchmark_data)} benchmark examples")
    
    # Group by database
    grouped_data = group_benchmark_by_database(benchmark_data)
    print(f"Found {len(grouped_data)} databases in benchmark data")
    
    # Limit samples per database
    if args.samples_per_db > 0:
        eval_data = []
        for db_id, items in grouped_data.items():
            samples = items[:args.samples_per_db]
            eval_data.extend(samples)
            print(f"Selected {len(samples)} samples from {db_id}")
        
        benchmark_data = eval_data
    
    # Initialize agent with default database
    print(f"Initializing MAC-SQL Agent with model {args.model}...")
    agent = MACSQLAgent(model_name=args.model, api_key=args.api_key)
    
    # Run evaluation
    print(f"Running evaluation on {len(benchmark_data)} examples...")
    results = evaluate_benchmark(
        agent=agent, 
        benchmark_items=benchmark_data, 
        output_file=args.output,
        rate_limit_delay=args.delay
    )
    
    # Print summary
    print_results_summary(results)
    
    # Create visualizations
    if args.visualize:
        print("Creating visualizations...")
        output_prefix = args.output.replace('.json', '')
        create_visualizations(results, output_prefix)

if __name__ == "__main__":
    main() 