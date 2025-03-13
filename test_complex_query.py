#!/usr/bin/env python
"""
Test script for MAC-SQL with a complex query
"""

from mac_sql import MACSQL

def main():
    # Initialize MAC-SQL
    print("Initializing MAC-SQL...")
    mac_sql = MACSQL(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    
    # Connect to student_club database
    print("Connecting to student_club database...")
    success = mac_sql.connect_to_database("student_club")
    if not success:
        print("Failed to connect to database")
        return
    
    # Run a complex query with join
    query = "What is the total amount of income received in 2019, grouped by source? The date_received field in the income table is a text field in 'YYYY-MM-DD' format that needs to be converted with TO_DATE() or substring operations."
    print(f"Running query: {query}")
    result = mac_sql.query(query, verbose=True)
    
    # Print results
    print("\nSQL Query:")
    print(result.get("sql_query", "No SQL generated"))
    
    print("\nQuery Result:")
    query_result = result.get("query_result", "No result")
    if isinstance(query_result, object) and hasattr(query_result, "head"):
        print(f"({len(query_result)} rows):")
        print(query_result)
    else:
        print(query_result)

if __name__ == "__main__":
    main() 