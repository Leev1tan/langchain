import os
import json
import argparse
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mac_sql_agent import MACSQLAgent

def load_mini_bird_data(benchmark_path):
    """Load the mini-bird benchmark data from a JSON file"""
    try:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} benchmark examples from {benchmark_path}")
        return data
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return []

def run_evaluation(model_name, benchmark_path, api_key=None, num_samples=None, output_path=None):
    """Run the evaluation of the MAC-SQL agent on the mini-bird benchmark"""
    # Load benchmark data
    benchmark_data = load_mini_bird_data(benchmark_path)
    if not benchmark_data:
        print("No benchmark data found. Exiting.")
        return None
    
    # Limit samples if specified
    if num_samples and num_samples < len(benchmark_data):
        print(f"Limiting evaluation to {num_samples} samples")
        benchmark_data = benchmark_data[:num_samples]
    
    # Initialize agent
    print(f"Initializing MAC-SQL Agent with model: {model_name}")
    agent = MACSQLAgent(model_name=model_name, api_key=api_key)
    
    # Run evaluation
    print("Starting evaluation...")
    results = agent.evaluate_on_benchmark(benchmark_data)
    
    # Save results if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    # Print summary
    print_summary(results)
    
    return results

def print_summary(results):
    """Print a summary of the evaluation results"""
    print("\n===== MAC-SQL Agent Evaluation Summary =====")
    print(f"Execution Accuracy: {results['execution_accuracy']:.2%}")
    
    # Count of successful queries
    total = len(results['detailed_results'])
    successful = sum(1 for r in results['detailed_results'] if r.get('results_match', False))
    print(f"Successful Queries: {successful}/{total} ({successful/total:.2%})")
    
    # Count of errors
    errors = sum(1 for r in results['detailed_results'] if 'error' in r)
    print(f"Queries with Errors: {errors}/{total} ({errors/total:.2%})")

def visualize_results(results):
    """Create visualizations for the evaluation results"""
    # Convert detailed results to DataFrame
    df = pd.DataFrame(results['detailed_results'])
    
    # Create success rate pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate pie chart
    labels = ['Successful', 'Failed']
    sizes = [
        sum(df['results_match']),
        len(df) - sum(df['results_match'])
    ]
    colors = ['#66b3ff', '#ff9999']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Query Execution Success Rate')
    
    # Error types bar chart
    if 'error' in df.columns:
        # Extract error types
        def get_error_type(error_str):
            if not isinstance(error_str, str):
                return 'No Error'
            elif 'syntax error' in error_str.lower():
                return 'Syntax Error'
            elif 'relation' in error_str.lower() and 'does not exist' in error_str.lower():
                return 'Table/Column Not Found'
            elif 'permission denied' in error_str.lower():
                return 'Permission Denied'
            else:
                return 'Other Error'
        
        df['error_type'] = df['error'].apply(get_error_type)
        error_counts = df['error_type'].value_counts()
        
        sns.barplot(x=error_counts.index, y=error_counts.values, ax=ax2)
        ax2.set_title('Error Types')
        ax2.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'No errors found in the evaluation', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    return fig

# Command line interface
def main():
    parser = argparse.ArgumentParser(description='Evaluate MAC-SQL Agent on mini-bird benchmark')
    parser.add_argument('--benchmark', default='data/mini-bird/dev.json', help='Path to benchmark JSON file')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model name')
    parser.add_argument('--api_key', help='Together API key (optional)')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to evaluate')
    parser.add_argument('--output', default='results/evaluation_results.json', help='Output file path')
    parser.add_argument('--ui', action='store_true', help='Launch Streamlit UI')
    
    args = parser.parse_args()
    
    if args.ui:
        # If UI flag is set, run Streamlit app
        print("Launching Streamlit UI...")
        import sys
        sys.argv = ["streamlit", "run", __file__]
        import streamlit.web.cli as stcli
        stcli._main_run_cloned(sys.argv)
    else:
        # Otherwise, run command-line evaluation
        run_evaluation(
            model_name=args.model,
            benchmark_path=args.benchmark,
            api_key=args.api_key,
            num_samples=args.samples,
            output_path=args.output
        )

# Streamlit UI
def streamlit_ui():
    st.set_page_config(page_title="MAC-SQL Benchmark Evaluation", layout="wide")
    
    st.title("MAC-SQL Agent Evaluation on mini-bird Benchmark")
    
    # Sidebar configuration
    st.sidebar.header("Evaluation Settings")
    
    # Model selection
    model_options = {
        "Meta Llama 3.1 8B Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Meta Llama 3.1 70B Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "Mistral 7B Instruct v0.3": "mistral-7B-Instruct-v0.3",
        "Qwen 1.5 72B Chat": "qwen-1.5-72B-Chat",
        "Qwen2.5 7B Instruct": "qwen2.5-7B-Instruct"
    }
    model_choice = st.sidebar.selectbox("Select Model:", list(model_options.keys()), index=0)
    model_name = model_options[model_choice]
    
    # API key input
    api_key = st.sidebar.text_input("Together API Key (optional):", type="password")
    if not api_key:
        api_key = None
    
    # Benchmark file selection
    benchmark_path = st.sidebar.text_input("Benchmark File Path:", "data/mini-bird/dev.json")
    
    # Number of samples
    num_samples = st.sidebar.number_input("Number of Samples (0 = all):", min_value=0, value=5)
    if num_samples == 0:
        num_samples = None
    
    # Output path
    output_path = st.sidebar.text_input("Output File Path:", "results/evaluation_results.json")
    
    # Run evaluation button
    run_button = st.sidebar.button("Run Evaluation")
    
    # Display panel
    if "results" not in st.session_state:
        st.session_state.results = None
    
    # Main content area
    tabs = st.tabs(["Results Summary", "Detailed Results", "Visualizations", "SQL Examples"])
    
    if run_button:
        with st.spinner("Running evaluation..."):
            results = run_evaluation(
                model_name=model_name,
                benchmark_path=benchmark_path,
                api_key=api_key,
                num_samples=num_samples,
                output_path=output_path
            )
            st.session_state.results = results
    
    # Load previously saved results
    load_saved = st.sidebar.button("Load Saved Results")
    if load_saved and os.path.exists(output_path):
        with st.spinner("Loading saved results..."):
            with open(output_path, 'r', encoding='utf-8') as f:
                st.session_state.results = json.load(f)
    
    # Display results in tabs
    if st.session_state.results:
        results = st.session_state.results
        
        # Results Summary tab
        with tabs[0]:
            st.header("Evaluation Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Execution Accuracy", f"{results['execution_accuracy']:.2%}")
            
            with col2:
                successful = sum(1 for r in results['detailed_results'] if r.get('results_match', False))
                total = len(results['detailed_results'])
                st.metric("Successful Queries", f"{successful}/{total}")
            
            with col3:
                errors = sum(1 for r in results['detailed_results'] if 'error' in r)
                st.metric("Queries with Errors", f"{errors}/{total}")
        
        # Detailed Results tab
        with tabs[1]:
            st.header("Detailed Results")
            
            # Convert to DataFrame for easier display
            df = pd.DataFrame(results['detailed_results'])
            if 'results_match' in df.columns:
                df['status'] = df['results_match'].apply(lambda x: "✅ Success" if x else "❌ Failed")
            
            # Add expandable rows for each result
            for i, row in df.iterrows():
                with st.expander(f"Query {i+1}: {row['question'][:100]}..."):
                    st.markdown(f"**Status:** {row.get('status', 'Unknown')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Gold SQL")
                        st.code(row['gold_sql'], language="sql")
                    
                    with col2:
                        st.subheader("Generated SQL")
                        st.code(row['generated_sql'], language="sql")
                    
                    if 'error' in row and row['error']:
                        st.error(f"Error: {row['error']}")
        
        # Visualizations tab
        with tabs[2]:
            st.header("Evaluation Visualizations")
            
            fig = visualize_results(results)
            st.pyplot(fig)
        
        # SQL Examples tab
        with tabs[3]:
            st.header("SQL Query Examples")
            
            # Filter for successful and failed examples
            if 'results_match' in df.columns:
                successful_examples = df[df['results_match'] == True]
                failed_examples = df[df['results_match'] == False]
                
                st.subheader("Successful Examples")
                if not successful_examples.empty:
                    for i, row in successful_examples.iterrows():
                        with st.expander(f"Example {i+1}: {row['question'][:100]}..."):
                            st.markdown(f"**Question:** {row['question']}")
                            st.markdown("**Generated SQL:**")
                            st.code(row['generated_sql'], language="sql")
                            st.markdown("**Gold SQL:**")
                            st.code(row['gold_sql'], language="sql")
                else:
                    st.info("No successful examples found.")
                
                st.subheader("Failed Examples")
                if not failed_examples.empty:
                    for i, row in failed_examples.iterrows():
                        with st.expander(f"Example {i+1}: {row['question'][:100]}..."):
                            st.markdown(f"**Question:** {row['question']}")
                            st.markdown("**Generated SQL:**")
                            st.code(row['generated_sql'], language="sql")
                            st.markdown("**Gold SQL:**")
                            st.code(row['gold_sql'], language="sql")
                            if 'error' in row and row['error']:
                                st.error(f"Error: {row['error']}")
                else:
                    st.info("No failed examples found.")

if __name__ == "__main__":
    # Check if running with Streamlit
    import sys
    if 'streamlit' in sys.argv[0]:
        streamlit_ui()
    else:
        main() 