import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from mac_sql_agent import MACSQLAgent
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

def load_benchmark_data(benchmark_path):
    """Load benchmark data from JSON file"""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def setup_database(db_id):
    """
    Setup the database connection configuration for a specific database
    Returns updated DB_CONFIG
    """
    # This would normally connect to the specific database in the benchmark
    # For simplicity, we're using the default configuration
    from mac_sql_agent import DB_CONFIG
    return DB_CONFIG

def evaluate_agent_with_rate_limits(agent, benchmark_data, num_samples=None, output_file=None):
    """
    Evaluate the agent on benchmark data with rate limit handling
    
    Args:
        agent: The MAC-SQL agent
        benchmark_data: List of benchmark items
        num_samples: Number of samples to evaluate (None = all)
        output_file: File to save results to
    
    Returns:
        Dictionary with evaluation results
    """
    if num_samples:
        benchmark_data = benchmark_data[:num_samples]
    
    correct = 0
    total = len(benchmark_data)
    results = []
    
    # Create a progress bar
    pbar = tqdm(total=total, desc="Evaluating queries")
    
    # Setup for partial results saving
    temp_output_file = output_file.replace('.json', '_partial.json') if output_file else "results_partial.json"
    
    for i, item in enumerate(benchmark_data):
        question = item['question']
        gold_sql = item['SQL']
        
        # Add result entry
        result_entry = {
            'question_id': item.get('question_id', i),
            'question': question,
            'gold_sql': gold_sql,
            'generated_sql': None,
            'results_match': False
        }
        
        # Generate SQL with retry logic
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Add random delay to avoid rate limiting (between 1-3 seconds)
                time.sleep(1 + random.random() * 2)
                
                # Generate SQL for the question
                generated_sql = agent.generate_sql(question)
                result_entry['generated_sql'] = generated_sql
                
                # Execute both queries
                try:
                    generated_results = agent.execute_sql_query(generated_sql)
                    gold_results = agent.execute_sql_query(gold_sql)
                    
                    # Check if results match
                    if isinstance(generated_results, pd.DataFrame) and isinstance(gold_results, pd.DataFrame):
                        results_match = generated_results.equals(gold_results)
                    else:
                        results_match = False
                        
                    if results_match:
                        correct += 1
                        
                    result_entry['results_match'] = results_match
                    
                except Exception as e:
                    result_entry['error'] = str(e)
                
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
        if i % 5 == 0 and i > 0:
            partial_results = {
                'execution_accuracy': correct / (i+1) if i+1 > 0 else 0,
                'detailed_results': results
            }
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(partial_results, f, indent=2)
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
            json.dump(final_results, f, indent=2)
    
    return final_results

def print_results(results):
    """Print evaluation results in a readable format"""
    print(f"\n===== MAC-SQL Agent Evaluation Results =====")
    print(f"Execution Accuracy: {results['execution_accuracy']:.2%}")
    
    # Count success by difficulty level
    difficulty_counts = {}
    for result in results['detailed_results']:
        difficulty = result.get('difficulty', 'unknown')
        if difficulty not in difficulty_counts:
            difficulty_counts[difficulty] = {'correct': 0, 'total': 0}
        
        difficulty_counts[difficulty]['total'] += 1
        if result.get('results_match', False):
            difficulty_counts[difficulty]['correct'] += 1
    
    print("\nResults by Difficulty:")
    for difficulty, counts in difficulty_counts.items():
        accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        print(f"  {difficulty}: {accuracy:.2%} ({counts['correct']}/{counts['total']})")
    
    # Count of errors
    error_count = sum(1 for r in results['detailed_results'] if 'error' in r)
    print(f"\nQueries with errors: {error_count} ({error_count/len(results['detailed_results']):.2%})")

def visualize_results(results, output_file=None):
    """Create visualizations of the evaluation results"""
    # Prepare data
    df_results = pd.DataFrame(results['detailed_results'])
    
    # Add success column
    df_results['success'] = df_results['results_match']
    
    # Count by difficulty
    if 'difficulty' in df_results.columns:
        difficulty_success = df_results.groupby('difficulty')['success'].agg(['mean', 'count'])
        difficulty_success.columns = ['Success Rate', 'Count']
        
        # Plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=difficulty_success.index, y=difficulty_success['Success Rate'])
        
        # Add count labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f"n={difficulty_success['Count'].iloc[i]}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points')
        
        plt.title('Success Rate by Difficulty')
        plt.ylim(0, 1.1)
        plt.ylabel('Success Rate')
        plt.xlabel('Difficulty')
        
        if output_file:
            plt.savefig(f"{output_file}_difficulty.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # Overall success rate
    plt.figure(figsize=(6, 6))
    success_rate = df_results['success'].mean()
    plt.pie([success_rate, 1-success_rate], labels=['Success', 'Failure'], autopct='%1.1f%%')
    plt.title('Overall Success Rate')
    
    if output_file:
        plt.savefig(f"{output_file}_overall.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate MAC-SQL Agent on mini-bird benchmark')
    parser.add_argument('--benchmark', default='dev_20240627/dev.json', help='Path to benchmark JSON file')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-70B-Instruct', help='Model to use')
    parser.add_argument('--api_key', help='Together API key')
    parser.add_argument('--num_samples', type=int, help='Number of samples to evaluate')
    parser.add_argument('--output', help='Output file prefix for results')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    # Load benchmark data
    print(f"Loading benchmark data from {args.benchmark}...")
    benchmark_data = load_benchmark_data(args.benchmark)
    print(f"Loaded {len(benchmark_data)} benchmark samples")
    
    # Initialize agent
    print(f"Initializing MAC-SQL Agent with model {args.model}...")
    agent = MACSQLAgent(model_name=args.model, api_key=args.api_key)
    
    # Run evaluation with rate limit handling
    print(f"Running evaluation...")
    output_file = f"{args.output}_results.json" if args.output else None
    results = evaluate_agent_with_rate_limits(agent, benchmark_data, args.num_samples, output_file)
    
    # Print results
    print_results(results)
    
    # Visualize results
    if args.visualize:
        print("Creating visualizations...")
        visualize_results(results, args.output)

if __name__ == "__main__":
    main() 