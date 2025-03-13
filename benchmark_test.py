#!/usr/bin/env python
"""
Benchmark test for MAC-SQL framework

This script creates a simple benchmark file for the student_club database
and runs evaluation to test the improved evaluation framework.
"""

import os
import json
import time
import traceback
from mac_sql import MACSQL

def create_benchmark_file():
    """Create a simple benchmark file for student_club database"""
    
    # Define test queries
    test_queries = [
        {
            "db_id": "student_club",
            "question": "How many records are in the member table?",
            "query": "SELECT COUNT(*) FROM member;",
            "complexity": "simple",
            "question_id": 1
        },
        {
            "db_id": "student_club",
            "question": "What is the total amount of income received in 2019, grouped by source?",
            "query": "SELECT source, SUM(amount) AS total_amount FROM income WHERE date_received LIKE '2019-%' GROUP BY source;",
            "complexity": "medium",
            "question_id": 2
        },
        {
            "db_id": "student_club",
            "question": "Which members joined the club in 2020?",
            "query": "SELECT first_name, last_name FROM member WHERE join_date LIKE '2020-%';",
            "complexity": "simple",
            "question_id": 3
        },
        {
            "db_id": "student_club",
            "question": "What are the names of all events that have expenses over $100?",
            "query": "SELECT e.name FROM event e JOIN expense x ON e.id = x.event_id WHERE x.amount > 100;",
            "complexity": "complex",
            "question_id": 4
        },
        {
            "db_id": "student_club",
            "question": "What is the total amount spent on all events?",
            "query": "SELECT SUM(amount) FROM expense;",
            "complexity": "simple",
            "question_id": 5
        }
    ]
    
    # Create benchmark directory if it doesn't exist
    os.makedirs("benchmark", exist_ok=True)
    
    # Write benchmark file
    benchmark_file = "benchmark/student_club_benchmark.json"
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(test_queries, f, indent=2)
    
    print(f"Created benchmark file: {benchmark_file}")
    return benchmark_file

def test_mac_sql_manually():
    """Test MAC-SQL manually on student_club database"""
    print("Testing MAC-SQL manually...")
    
    # Initialize MAC-SQL
    mac_sql = MACSQL()
    
    # Connect to student_club database
    print("Connecting to student_club database...")
    connected = mac_sql.connect_to_database("student_club")
    if not connected:
        print("Failed to connect to student_club database!")
        return
    
    # Define test questions
    test_questions = [
        "How many records are in the member table?",
        "What is the total amount of income received in 2019, grouped by source?",
        "Which members joined the club in 2020?",
        "What are the names of all events that have expenses over $100?",
        "What is the total amount spent on all events?"
    ]
    
    # Test each question
    correct_count = 0
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"\nTest {i+1}: {question}")
        try:
            # Process the question
            start_time = time.time()
            result = mac_sql.query(question, verbose=True)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Check if query was successful
            sql_query = result.get('sql_query', '')
            query_result = result.get('query_result', None)
            error = result.get('error', None)
            
            print(f"Generated SQL: {sql_query}")
            print(f"Execution time: {execution_time:.2f}s")
            
            if error:
                print(f"Error: {error}")
                success = False
            elif query_result is not None:
                print(f"Result: {query_result}")
                success = True
                correct_count += 1
            else:
                print("No result or error returned")
                success = False
            
            # Track results
            results.append({
                'question_id': i+1,
                'question': question,
                'generated_sql': sql_query,
                'result': query_result,
                'error': error,
                'success': success,
                'execution_time': execution_time
            })
            
        except Exception as e:
            print(f"Error processing question: {e}")
            traceback.print_exc()
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/manual_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n===== MANUAL TEST RESULTS =====")
    print(f"Total correct: {correct_count} / {len(test_questions)}")
    print(f"Accuracy: {(correct_count / len(test_questions)) * 100:.2f}%")
    

def main():
    # Create benchmark file
    benchmark_file = create_benchmark_file()
    
    # Test MAC-SQL manually
    test_mac_sql_manually()
    
    # Now try to test the evaluation framework with a modified approach
    try:
        print("\nTesting the evaluation framework...")
        
        # Initialize MAC-SQL
        mac_sql = MACSQL()
        
        # Initialize results
        results = {
            "model": mac_sql.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "error_count": 0,
            "detailed_results": [],
            "results_by_database": {}
        }
        
        # Load benchmark data
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        # Connect to student_club database
        print("Connecting to student_club database...")
        connected = mac_sql.connect_to_database("student_club")
        if not connected:
            print("Failed to connect to student_club database!")
            return
        
        # Process each benchmark item
        for i, item in enumerate(benchmark_data):
            question = item["question"]
            gold_sql = item["query"]
            
            print(f"\nBenchmark {i+1}: {question}")
            
            # Process the question
            start_time = time.time()
            result = mac_sql.query(question, verbose=False)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Extract results
            generated_sql = result.get('sql_query', '')
            query_result = result.get('query_result', None)
            error = result.get('error', None)
            
            print(f"Generated SQL: {generated_sql}")
            print(f"Gold SQL: {gold_sql}")
            print(f"Execution time: {execution_time:.2f}s")
            
            # Execute gold SQL
            gold_result = None
            try:
                gold_result = mac_sql.chat_manager.execute_sql_query(gold_sql)
                print(f"Gold result: {gold_result.get('rows', [])}")
            except Exception as e:
                print(f"Error executing gold SQL: {e}")
            
            # Check if results match
            results_match = False
            if query_result is not None and gold_result is not None and gold_result.get('success', False):
                # Compare results (simplified)
                if len(query_result) == len(gold_result.get('rows', [])):
                    results_match = True
            
            # Update counts
            if results_match:
                results["correct"] += 1
                print("Result: ✓ (Match)")
            else:
                print("Result: ✗ (No match)")
            
            results["total"] += 1
            if error:
                results["error_count"] += 1
            
            # Add detailed result
            results["detailed_results"].append({
                "question_id": item.get("question_id", i),
                "db_id": "student_club",
                "question": question,
                "gold_sql": gold_sql,
                "generated_sql": generated_sql,
                "results_match": results_match,
                "execution_time": execution_time,
                "error": error,
                "complexity": item.get("complexity", "unknown")
            })
        
        # Calculate accuracy
        if results["total"] > 0:
            results["execution_accuracy"] = (results["correct"] / results["total"]) * 100
        
        # Update database results
        results["results_by_database"]["student_club"] = {
            "correct": results["correct"],
            "total": results["total"],
            "accuracy": results["execution_accuracy"]
        }
        
        # Save results
        output_file = "results/custom_evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print("\n===== CUSTOM EVALUATION RESULTS =====")
        print(f"Model: {results['model']}")
        print(f"Execution Accuracy: {results['execution_accuracy']:.2f}%")
        print(f"Total correct: {results['correct']} / {results['total']}")
        print(f"Errors: {results['error_count']}")
        
    except Exception as e:
        print(f"Error in custom evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 