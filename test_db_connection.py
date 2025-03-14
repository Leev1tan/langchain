#!/usr/bin/env python
"""
Test script to verify database connections and table existence
"""

import os
import psycopg2
import json
from psycopg2 import sql

# Get PostgreSQL connection parameters from environment variables
DB_CONFIG = {
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}

def test_database_connection(db_name):
    """
    Test connection to a database and list all tables
    
    Args:
        db_name: Name of database to connect to
    """
    print(f"\nTesting connection to database: {db_name}")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=db_name,
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"]
        )
        
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        tables = cursor.fetchall()
        
        print(f"Successfully connected to {db_name}")
        print(f"Tables in {db_name}:")
        for table in tables:
            print(f"  - {table[0]}")
            
            # Get table schema
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table[0]}'
            """)
            
            columns = cursor.fetchall()
            print(f"    Columns:")
            for col in columns:
                print(f"      - {col[0]} ({col[1]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"    Row count: {count}")
            
            # Sample data (first 3 rows)
            cursor.execute(f"SELECT * FROM {table[0]} LIMIT 3")
            rows = cursor.fetchall()
            if rows:
                print(f"    Sample data:")
                for row in rows:
                    print(f"      {row}")
            
            print()
        
    except Exception as e:
        print(f"Error connecting to database {db_name}: {e}")
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()

def test_query(db_name, query):
    """
    Test a specific SQL query on a database
    
    Args:
        db_name: Name of database to connect to
        query: SQL query to execute
    """
    print(f"\nTesting query on database: {db_name}")
    print(f"Query: {query}")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=db_name,
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"]
        )
        
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query)
        
        # Fetch results
        results = cursor.fetchall()
        
        print(f"Query executed successfully")
        print(f"Results:")
        for row in results:
            print(f"  {row}")
        
    except Exception as e:
        print(f"Error executing query on database {db_name}: {e}")
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    print("Testing database connections...")
    
    # Test student_club database
    test_database_connection("student_club")
    
    # Test formula_1 database
    test_database_connection("formula_1")
    
    # Test some example queries
    test_query("student_club", "SELECT m.name, e.event_name FROM member m JOIN attendance a ON m.member_id = a.member_id JOIN event e ON a.event_id = e.event_id WHERE a.status = 'attended'")
    
    test_query("formula_1", "SELECT d.name, t.name, COUNT(r.result_id) as wins FROM driver d JOIN result r ON d.driver_id = r.driver_id JOIN team t ON r.team_id = t.team_id WHERE r.position = 1 GROUP BY d.name, t.name")
    
    print("\nDatabase connection tests complete!") 