import os
import sys
import importlib

def apply_fixes():
    """Apply all the fixes before running the evaluation"""
    print("Applying SQL dialect and schema retrieval fixes...")
    
    # Run the schema retrieval fix
    from fix_schema_retrieval import modify_mac_sql_agent
    modify_mac_sql_agent()
    
    # Update the evaluation method
    # First, import the needed modules and DB_CONFIG
    from mac_sql_agent import MACSQLAgent, DB_CONFIG
    import psycopg2
    import pandas as pd
    from sql_dialect_adapter import adapt_sql_dialect, results_are_equivalent
    
    # Define the patched method
    def patched_execute_sql_query(self, sql_query, refinement_attempts=0):
        """Execute the generated SQL query and handle errors if they arise"""
        MAX_REFINEMENT_ATTEMPTS = 2
        try:
            # First adapt the SQL query to PostgreSQL dialect if needed
            if '`' in sql_query:  # Simple check for MySQL/SQLite dialect
                sql_query = adapt_sql_dialect(sql_query, "mysql", "postgresql")
            
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            return pd.DataFrame(rows, columns=columns)
        except psycopg2.Error as e:
            if refinement_attempts < MAX_REFINEMENT_ATTEMPTS:
                refined_query = self.refine_sql_query(sql_query, str(e))
                return self.execute_sql_query(refined_query, refinement_attempts + 1)
            else:
                return f"Error executing query: {e}"
    
    # Replace the method in the MACSQLAgent class
    print("Patching the execute_sql_query method...")
    MACSQLAgent.execute_sql_query = patched_execute_sql_query
    
    # Also patch the evaluate_on_benchmark method to handle dialect differences in gold SQL
    def patched_evaluate_on_benchmark(self, benchmark_data):
        """
        Evaluate the MAC-SQL agent on a benchmark dataset with SQL dialect adaptation
        
        Args:
            benchmark_data: List of dictionaries with 'question' and 'SQL' keys
            
        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        total = len(benchmark_data)
        results = []
        
        for i, item in enumerate(benchmark_data):
            question = item['question']
            gold_sql = item['SQL']
            
            # Generate SQL for the question
            generated_sql = self.generate_sql(question)
            
            # Adapt gold SQL to PostgreSQL dialect
            adapted_gold_sql = adapt_sql_dialect(gold_sql, "mysql", "postgresql")
            
            # Execute both the generated and gold SQL queries
            try:
                generated_results = self.execute_sql_query(generated_sql)
                gold_results = self.execute_sql_query(adapted_gold_sql)
                
                # Check if results match
                if isinstance(generated_results, pd.DataFrame) and isinstance(gold_results, pd.DataFrame):
                    results_match = results_are_equivalent(generated_results, gold_results)
                else:
                    results_match = False
                    
                if results_match:
                    correct += 1
                    
                results.append({
                    'question_id': item.get('question_id', i),
                    'question': question,
                    'gold_sql': gold_sql,
                    'adapted_gold_sql': adapted_gold_sql,
                    'generated_sql': generated_sql,
                    'results_match': results_match
                })
                
            except Exception as e:
                results.append({
                    'question_id': item.get('question_id', i),
                    'question': question,
                    'gold_sql': gold_sql,
                    'adapted_gold_sql': adapted_gold_sql,
                    'generated_sql': generated_sql,
                    'error': str(e),
                    'results_match': False
                })
        
        # Calculate execution accuracy
        execution_accuracy = correct / total if total > 0 else 0
        
        return {
            'execution_accuracy': execution_accuracy,
            'detailed_results': results
        }
    
    # Replace the evaluate_on_benchmark method
    print("Patching the evaluate_on_benchmark method...")
    MACSQLAgent.evaluate_on_benchmark = patched_evaluate_on_benchmark
    
    print("All fixes applied successfully!")

def run_evaluation():
    """Run the evaluation with the fixes applied"""
    # Import and reload the evaluate_benchmark module to ensure it sees our changes
    import evaluate_benchmark
    importlib.reload(evaluate_benchmark)
    
    # Get the command line arguments for the evaluation (excluding this script's name)
    eval_args = sys.argv[1:]
    
    # If no arguments were provided, use some defaults
    if not eval_args:
        eval_args = ["--benchmark", "dev_20240627/dev.json", 
                     "--model", "meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                     "--num_samples", "5"]
    
    # Run the evaluation
    print(f"Running evaluation with args: {eval_args}")
    sys.argv = ["evaluate_benchmark.py"] + eval_args
    evaluate_benchmark.main()

if __name__ == "__main__":
    # Apply the fixes first
    apply_fixes()
    
    # Then run the evaluation
    run_evaluation() 