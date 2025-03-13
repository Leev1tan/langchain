#!/usr/bin/env python
"""
Simple test script for MAC-SQL
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
    
    # Run a query
    print("Running query: How many records are in the member table?")
    result = mac_sql.query("How many records are in the member table?", verbose=True)
    
    # Print results
    print("\nSQL Query:")
    print(result.get("sql_query", "No SQL generated"))
    
    print("\nQuery Result:")
    query_result = result.get("query_result", "No result")
    if isinstance(query_result, object) and hasattr(query_result, "head"):
        print(f"({len(query_result)} rows):")
        print(query_result.head())
    else:
        print(query_result)

if __name__ == "__main__":
    main() 