"""
Streamlit app for MAC-SQL: Multi-Agent Collaboration for SQL Generation
"""

import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import time
from mac_sql import MACSQL

# Setup page config
st.set_page_config(
    page_title="MAC-SQL Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2E86C1;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .error-text {
        color: #E74C3C;
        font-weight: bold;
    }
    .code-box {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_macsql_instance(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key=None):
    """Initialize and cache MAC-SQL instance"""
    return MACSQL(model_name=model_name, api_key=api_key, verbose=True)

def load_evaluation_results(file_path):
    """Load evaluation results from a JSON file"""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    # Sidebar
    st.sidebar.markdown('<div class="main-header">MAC-SQL Dashboard</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="info-text">Multi-Agent Collaboration for SQL Generation</div>', unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Query Interface", "Evaluation Results", "Benchmark Runner"])
    
    # Model selection
    model_options = [
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "TogetherAI/togethercomputer/llama-2-7b",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    
    # API Key
    api_key = st.sidebar.text_input("Together API Key (optional)", type="password")
    
    # Initialize MAC-SQL instance
    macsql = get_macsql_instance(model_name=selected_model, api_key=api_key)
    
    # Pages
    if page == "Query Interface":
        show_query_interface(macsql)
    elif page == "Evaluation Results":
        show_evaluation_results()
    elif page == "Benchmark Runner":
        show_benchmark_runner(macsql)

def show_query_interface(macsql):
    st.markdown('<div class="main-header">MAC-SQL Query Interface</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Enter a natural language question to generate and execute SQL</div>', 
                unsafe_allow_html=True)
    
    # Database selection
    st.markdown('<div class="subheader">1. Select Database</div>', unsafe_allow_html=True)
    
    # Get available databases
    try:
        available_dbs = macsql.get_available_databases()
        if available_dbs:
            selected_db = st.selectbox("Database", available_dbs)
        else:
            st.warning("No databases available. Please check your PostgreSQL connection.")
            selected_db = st.text_input("Enter Database Name")
    except Exception as e:
        st.error(f"Error retrieving databases: {str(e)}")
        selected_db = st.text_input("Enter Database Name")
    
    # Connect to database button
    if st.button("Connect to Database"):
        with st.spinner(f"Connecting to database '{selected_db}'..."):
            success = macsql.connect_to_database(selected_db)
            if success:
                st.success(f"Connected to database: {selected_db}")
            else:
                st.error(f"Failed to connect to database: {selected_db}")
    
    # Query input
    st.markdown('<div class="subheader">2. Enter Your Question</div>', unsafe_allow_html=True)
    question = st.text_area("Natural Language Question", 
                            "List all customers who made purchases in the last month.", 
                            height=100)
    
    # Run query button
    if st.button("Generate SQL & Execute"):
        if not question:
            st.warning("Please enter a question.")
            return
            
        with st.spinner("Processing your question..."):
            # Connect to database if not already connected
            if not macsql.chat_manager.connection:
                if selected_db:
                    macsql.connect_to_database(selected_db)
                else:
                    st.error("Please connect to a database first.")
                    return
            
            # Process the query
            start_time = time.time()
            sql_query, results = macsql.process_query(question, selected_db)
            execution_time = time.time() - start_time
        
        # Display results
        st.markdown('<div class="subheader">Results</div>', unsafe_allow_html=True)
        
        # Display SQL
        st.markdown("**Generated SQL:**")
        st.code(sql_query, language="sql")
        
        # Display execution time
        st.info(f"Execution time: {execution_time:.2f} seconds")
        
        # Display results
        if isinstance(results, pd.DataFrame) and not results.empty:
            st.markdown("**Query Results:**")
            st.dataframe(results)
            
            # Download button for results
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv",
            )
        elif isinstance(results, dict) and 'error' in results:
            st.error(f"Error: {results['error']}")
        else:
            st.info("Query executed successfully but returned no results.")

def show_evaluation_results():
    st.markdown('<div class="main-header">Evaluation Results</div>', unsafe_allow_html=True)
    
    # File selector for evaluation results
    results_dir = "results"
    if not os.path.exists(results_dir):
        st.warning("Results directory not found. Please run an evaluation first.")
        return
        
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    
    if not result_files:
        st.warning("No evaluation result files found. Please run an evaluation first.")
        return
    
    selected_file = st.selectbox("Select Results File", result_files)
    file_path = os.path.join(results_dir, selected_file)
    
    # Load and display results
    results = load_evaluation_results(file_path)
    
    if not results:
        st.error(f"Failed to load results from {file_path}")
        return
    
    # Display summary
    st.markdown('<div class="subheader">Evaluation Summary</div>', unsafe_allow_html=True)
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "SQL Match Rate", 
            f"{results.get('sql_match_rate', 0):.2%}", 
            f"{results.get('sql_match_count', 0)}/{results.get('total_items', 0)}"
        )
    with col2:
        st.metric(
            "Execution Success Rate", 
            f"{results.get('execution_success_rate', 0):.2%}", 
            f"{results.get('execution_success_count', 0)}/{results.get('total_items', 0)}"
        )
    with col3:
        st.metric(
            "Avg Similarity Score", 
            f"{results.get('avg_similarity', 0):.4f}"
        )
    
    # Display model info
    st.info(f"Model: {results.get('model_name', 'Unknown')}")
    st.info(f"Evaluation Time: {results.get('evaluation_time', 0):.2f} seconds")
    
    # Create visualization
    st.markdown('<div class="subheader">Results Visualization</div>', unsafe_allow_html=True)
    
    # Database performance chart
    if 'database_stats' in results:
        db_stats = results['database_stats']
        
        # Create a dataframe for the chart
        db_names = list(db_stats.keys())
        match_rates = [db_stats[db].get('sql_match_rate', 0) for db in db_names]
        exec_rates = [db_stats[db].get('execution_success_rate', 0) for db in db_names]
        
        chart_data = pd.DataFrame({
            'Database': db_names,
            'SQL Match Rate': match_rates,
            'Execution Success Rate': exec_rates
        })
        
        if not chart_data.empty:
            # Plot with matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            chart_data.plot(x='Database', y=['SQL Match Rate', 'Execution Success Rate'], 
                           kind='bar', ax=ax)
            ax.set_title('Performance by Database')
            ax.set_ylabel('Rate')
            ax.set_ylim(0, 1)
            
            # Display in Streamlit
            st.pyplot(fig)
    
    # Detailed results
    st.markdown('<div class="subheader">Detailed Results</div>', unsafe_allow_html=True)
    
    if 'results' in results:
        detailed_results = results['results']
        
        # Convert to DataFrame for easier display
        if detailed_results:
            df_results = pd.DataFrame(detailed_results)
            
            # Add filters
            if not df_results.empty:
                # Filter by success/failure
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=["Successful Matches", "Failed Matches", "Execution Errors"],
                    default=["Successful Matches", "Failed Matches", "Execution Errors"]
                )
                
                filtered_df = df_results.copy()
                
                if "Successful Matches" in status_filter and "Failed Matches" not in status_filter:
                    filtered_df = filtered_df[filtered_df['sql_match'] == True]
                elif "Failed Matches" in status_filter and "Successful Matches" not in status_filter:
                    filtered_df = filtered_df[filtered_df['sql_match'] == False]
                    
                if "Execution Errors" in status_filter:
                    filtered_df = filtered_df[filtered_df['execution_success'] == False]
                
                # Display filtered results
                st.dataframe(filtered_df)
                
                # Detailed view of selected result
                st.markdown('<div class="subheader">Result Details</div>', unsafe_allow_html=True)
                
                # Select a result to view details
                result_indices = list(range(len(filtered_df)))
                if result_indices:
                    selected_idx = st.selectbox("Select Result to View", result_indices)
                    
                    if 0 <= selected_idx < len(filtered_df):
                        selected_result = filtered_df.iloc[selected_idx].to_dict()
                        
                        # Display details
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Question:**")
                            st.markdown(f"{selected_result.get('question', '')}")
                            
                            st.markdown("**Database:**")
                            st.markdown(f"{selected_result.get('db_id', '')}")
                            
                            st.markdown("**Understanding:**")
                            st.markdown(f"{selected_result.get('understanding', '')}")
                            
                            st.markdown("**Plan:**")
                            st.markdown(f"{selected_result.get('plan', '')}")
                        
                        with col2:
                            st.markdown("**Gold SQL:**")
                            st.code(selected_result.get('gold_sql', ''), language="sql")
                            
                            st.markdown("**Generated SQL:**")
                            st.code(selected_result.get('generated_sql', ''), language="sql")
                            
                            # Success indicators
                            sql_match = selected_result.get('sql_match', False)
                            execution_success = selected_result.get('execution_success', False)
                            
                            if sql_match:
                                st.markdown('<div class="success-text">✅ SQL Match</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="error-text">❌ SQL Mismatch</div>', unsafe_allow_html=True)
                                
                            if execution_success:
                                st.markdown('<div class="success-text">✅ Execution Success</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="error-text">❌ Execution Failed</div>', unsafe_allow_html=True)
                            
                            # Similarity score
                            similarity = selected_result.get('similarity_score', 0)
                            st.progress(similarity)
                            st.markdown(f"Similarity Score: {similarity:.4f}")

def show_benchmark_runner(macsql):
    st.markdown('<div class="main-header">Benchmark Runner</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Run evaluation on benchmark datasets</div>', unsafe_allow_html=True)
    
    # Benchmark file selection
    st.markdown('<div class="subheader">1. Select Benchmark</div>', unsafe_allow_html=True)
    
    benchmark_options = [
        "minidev/MINIDEV/mini_dev_postgresql.json",
        "minidev/MINIDEV/mini_dev_mysql.json",
        "minidev/MINIDEV/mini_dev_sqlite.json"
    ]
    
    selected_benchmark = st.selectbox("Benchmark File", benchmark_options)
    
    # Configuration
    st.markdown('<div class="subheader">2. Configure Evaluation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples = st.number_input("Samples per Database", min_value=1, max_value=50, value=5)
        
    with col2:
        output_file = st.text_input("Output File", "results/streamlit_evaluation.json")
    
    # Run evaluation button
    if st.button("Run Evaluation"):
        if not selected_benchmark:
            st.warning("Please select a benchmark file.")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing evaluation...")
            
            # Create results directory
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Start evaluation
            status_text.text(f"Running evaluation on {selected_benchmark} with {num_samples} samples per database...")
            
            with st.spinner("Evaluation in progress..."):
                results = macsql.evaluate_benchmark(
                    benchmark_file=selected_benchmark,
                    num_samples=num_samples,
                    output_file=output_file
                )
                
                progress_bar.progress(1.0)
            
            # Show success message
            st.success(f"Evaluation completed successfully! Results saved to {output_file}")
            
            # Display summary
            st.markdown('<div class="subheader">Evaluation Summary</div>', unsafe_allow_html=True)
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "SQL Match Rate", 
                    f"{results.get('sql_match_rate', 0):.2%}", 
                    f"{results.get('sql_match_count', 0)}/{results.get('total_items', 0)}"
                )
            with col2:
                st.metric(
                    "Execution Success Rate", 
                    f"{results.get('execution_success_rate', 0):.2%}", 
                    f"{results.get('execution_success_count', 0)}/{results.get('total_items', 0)}"
                )
            with col3:
                st.metric(
                    "Avg Similarity Score", 
                    f"{results.get('avg_similarity', 0):.4f}"
                )
            
            # Recommend next steps
            st.info("Go to the 'Evaluation Results' page to view detailed results.")
            
        except Exception as e:
            st.error(f"Error running evaluation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 