#!/usr/bin/env python
"""
Test script for ChatManager database connection
"""

import os
import sys
import psycopg2
from core.chat_manager import ChatManager, DB_CONFIG

def get_actual_schema(db_name, db_config):
    """
    Get the actual schema from the database
    """
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config["port"]
        )
        
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_str = f"Database: {db_name}\n\n"
        
        for table in tables:
            schema_str += f"Table: {table}\n"
            
            # Get columns
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """)
            
            schema_str += "Columns:\n"
            for col in cursor.fetchall():
                nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                schema_str += f"  - {col[0]} ({col[1]}, {nullable})\n"
            
            # Get primary keys
            cursor.execute(f"""
                SELECT c.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                    AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = '{table}'
            """)
            
            pks = cursor.fetchall()
            if pks:
                schema_str += "Primary Keys:\n"
                for pk in pks:
                    schema_str += f"  - {pk[0]}\n"
            
            # Get foreign keys
            cursor.execute(f"""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table}'
            """)
            
            fks = cursor.fetchall()
            if fks:
                schema_str += "Foreign Keys:\n"
                for fk in fks:
                    schema_str += f"  - {fk[0]} -> {fk[1]}.{fk[2]}\n"
            
            schema_str += "\n"
        
        return schema_str
    
    except Exception as e:
        return f"Error getting schema: {str(e)}"
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()

def test_chat_manager_db_connection():
    """
    Test the ChatManager's ability to connect to our test databases
    """
    print("Testing ChatManager database connection...")
    
    # Initialize ChatManager
    chat_manager = ChatManager(verbose=True)
    
    # Test connection to student_club database
    print("\nTesting connection to student_club database:")
    success = chat_manager.connect_to_database("student_club")
    print(f"Connection successful: {success}")
    
    if success:
        # Get schema from ChatManager
        schema = chat_manager.get_schema()
        print("\nSchema from ChatManager for student_club database:")
        print(schema)
        
        # Get actual schema
        actual_schema = get_actual_schema("student_club", DB_CONFIG)
        print("\nActual schema for student_club database:")
        print(actual_schema)
        
        # Test a simple query
        query = "SELECT COUNT(*) FROM member"
        print(f"\nExecuting query: {query}")
        try:
            result = chat_manager.execute_sql_query(query)
            print(f"Query result: {result}")
        except Exception as e:
            print(f"Error executing query: {str(e)}")
    
    # Test connection to formula_1 database
    print("\nTesting connection to formula_1 database:")
    success = chat_manager.connect_to_database("formula_1")
    print(f"Connection successful: {success}")
    
    if success:
        # Get schema from ChatManager
        schema = chat_manager.get_schema()
        print("\nSchema from ChatManager for formula_1 database:")
        print(schema)
        
        # Get actual schema
        actual_schema = get_actual_schema("formula_1", DB_CONFIG)
        print("\nActual schema for formula_1 database:")
        print(actual_schema)
        
        # Test a simple query
        query = "SELECT COUNT(*) FROM driver"
        print(f"\nExecuting query: {query}")
        try:
            result = chat_manager.execute_sql_query(query)
            print(f"Query result: {result}")
        except Exception as e:
            print(f"Error executing query: {str(e)}")

if __name__ == "__main__":
    test_chat_manager_db_connection() 