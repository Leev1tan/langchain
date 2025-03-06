import os
import json
import pandas as pd
import psycopg2
from tqdm import tqdm
from mac_sql_agent import DB_CONFIG

# PostgreSQL connection configuration
DB_NAME = DB_CONFIG["dbname"]

def load_benchmark_data(benchmark_path):
    """Load benchmark data from JSON file"""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_db_schema():
    """Get a simple schema description for the database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Get table list
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = []
        for table in tables:
            # Get column information
            cursor.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table}'
                AND table_schema = 'public';
            """)
            columns = cursor.fetchall()
            
            # Format for output
            schema_info.append(f"Table: {table}")
            schema_info.append("Columns:")
            for col_name, data_type in columns:
                schema_info.append(f"  - {col_name} ({data_type})")
            schema_info.append("")
            
        conn.close()
        return "\n".join(schema_info)
    except Exception as e:
        print(f"Error getting schema: {e}")
        return "Could not retrieve schema"

def execute_query(sql_query):
    """Execute a SQL query and return the results"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return pd.DataFrame(results, columns=columns)
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

def select_database_for_question(question_data):
    """Select appropriate database based on question metadata"""
    db_id = question_data.get("db_id", "")
    
    # Update DB_CONFIG to point to the correct database
    config = DB_CONFIG.copy()
    config["dbname"] = db_id
    
    return config

def evaluate_questions(question_items, num_samples=5):
    """Evaluate SQL generation for sample questions"""
    from langchain_together import Together
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_core.output_parsers import StrOutputParser
    
    # Get database schema
    schema = get_db_schema()
    db_name = DB_CONFIG["dbname"]
    
    # Create language model
    model = Together(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        together_api_key="6e4593b7c0e0279476b65f144273d1ee972a47e3eb543c9649b36aaf6c114a82",
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    parser = StrOutputParser()
    chain = model | parser
    
    print(f"Evaluating {len(question_items)} questions for database: {db_name}")
    
    results = []
    for i, item in enumerate(question_items):
        question = item["question"]
        gold_sql = item.get("SQL", "")
        
        print(f"\nQuestion {i+1}: {question}")
        
        # Generate SQL with the model
        prompt = f"""
        You are a SQL expert. Generate a valid PostgreSQL query to answer the following question.
        
        DATABASE SCHEMA:
        {schema}
        
        QUESTION: {question}
        
        Return ONLY the SQL query without explanation or markdown formatting.
        """
        
        try:
            # Generate SQL
            sql_query = chain.invoke(prompt)
            
            # Clean up any markdown formatting
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            print(f"\nGenerated SQL: {sql_query}")
            print(f"Gold SQL: {gold_sql}")
            
            # Execute the query
            try:
                results_df = execute_query(sql_query)
                query_success = True if results_df is not None else False
                
                # Try to execute the gold SQL for comparison
                gold_results = None
                try:
                    # Adapt gold SQL to PostgreSQL if needed
                    adapted_gold_sql = gold_sql.replace('`', '"')
                    gold_results = execute_query(adapted_gold_sql)
                except Exception as gold_error:
                    print(f"Error executing gold SQL: {gold_error}")
                
                # Display results (first few rows)
                if query_success and not results_df.empty:
                    print("\nResults (first 3 rows):")
                    print(results_df.head(3))
                else:
                    print("\nQuery returned no results or failed")
                
                results.append({
                    "db_id": db_name,
                    "question_id": item.get("question_id", i),
                    "question": question,
                    "sql_query": sql_query,
                    "gold_sql": gold_sql,
                    "success": query_success,
                    "results_match": False,  # We'll update this if gold SQL works
                })
                
                # If both queries worked, compare results
                if query_success and gold_results is not None:
                    results[-1]["results_match"] = results_df.equals(gold_results)
                
            except Exception as exec_error:
                print(f"Error executing query: {exec_error}")
                results.append({
                    "db_id": db_name,
                    "question_id": item.get("question_id", i),
                    "question": question,
                    "sql_query": sql_query,
                    "gold_sql": gold_sql,
                    "success": False,
                    "error": str(exec_error)
                })
                
        except Exception as gen_error:
            print(f"Error generating SQL: {gen_error}")
            results.append({
                "db_id": db_name,
                "question_id": item.get("question_id", i),
                "question": question,
                "gold_sql": gold_sql,
                "success": False,
                "error": str(gen_error)
            })
    
    # Calculate success rate
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    return results

def main():
    # Check if benchmark file exists
    benchmark_path = "dev_20240627/dev.json"
    if not os.path.exists(benchmark_path):
        print(f"Benchmark file not found: {benchmark_path}")
        return
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(benchmark_path)
    print(f"Loaded {len(benchmark_data)} benchmark examples")
    
    # Get list of available databases
    available_dbs = [
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
    
    # Filter benchmark to only questions from databases we have
    filtered_benchmark = [item for item in benchmark_data if item.get("db_id") in available_dbs]
    print(f"Filtered to {len(filtered_benchmark)} questions from available databases")
    
    # Group questions by database
    questions_by_db = {}
    for item in filtered_benchmark:
        db_id = item.get("db_id")
        if db_id not in questions_by_db:
            questions_by_db[db_id] = []
        questions_by_db[db_id].append(item)
    
    # How many questions to evaluate per database
    num_samples_per_db = 2
    
    # Evaluate questions from each database
    all_results = []
    for db_id, questions in questions_by_db.items():
        print(f"\n===== Evaluating questions for database: {db_id} =====")
        
        # Update DB_CONFIG for this database
        global DB_CONFIG
        DB_CONFIG["dbname"] = db_id
        
        # Evaluate a sample of questions from this database
        db_questions = questions[:num_samples_per_db]
        results = evaluate_questions(db_questions, len(db_questions))
        all_results.extend(results)
    
    # Save combined results
    output_file = "results_all_databases.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if isinstance(x, pd.DataFrame) else x)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 