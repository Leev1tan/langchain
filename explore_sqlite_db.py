#!/usr/bin/env python
"""
SQLite Database Explorer
========================

This script explores the structure and contents of a SQLite database.

Usage:
    python explore_sqlite_db.py <database_name>

Example:
    python explore_sqlite_db.py card_games
"""

import os
import sys
import sqlite3
import pandas as pd

def get_sqlite_db_path(db_name, base_dir="dev_20240627/dev_databases"):
    """Get the path to a SQLite database file based on the database name"""
    db_path = os.path.join(base_dir, db_name, f"{db_name}.sqlite")
    if os.path.exists(db_path):
        return db_path
    else:
        print(f"Database file not found: {db_path}")
        return None

def explore_database(db_path):
    """Explore the structure and contents of a SQLite database"""
    print(f"Exploring database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nTables in database ({len(tables)}):")
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table}")
    
    # Explore each table
    for table in tables:
        print(f"\n=== Table: {table} ===")
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            col_id, col_name, data_type, not_null, default_value, primary_key = col
            pk = " (PRIMARY KEY)" if primary_key else ""
            null = " NOT NULL" if not_null else ""
            default = f" DEFAULT {default_value}" if default_value is not None else ""
            print(f"  - {col_name} ({data_type}{null}{default}{pk})")
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            print("Foreign Keys:")
            for fk in foreign_keys:
                id, seq, ref_table, from_col, to_col, on_update, on_delete, match = fk
                print(f"  - {from_col} -> {ref_table}({to_col})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        row_count = cursor.fetchone()[0]
        print(f"Row count: {row_count}")
        
        # Show sample data (up to 5 rows)
        if row_count > 0:
            print("Sample data (up to 5 rows):")
            cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
            rows = cursor.fetchall()
            column_names = [col[1] for col in columns]
            df = pd.DataFrame(rows, columns=column_names)
            print(df)
    
    conn.close()

def main():
    if len(sys.argv) < 2:
        print("Please provide a database name.")
        print("Available databases:")
        base_dir = "dev_20240627/dev_databases"
        if os.path.exists(base_dir):
            for db_dir in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, db_dir)):
                    print(f"  - {db_dir}")
        print("\nUsage: python explore_sqlite_db.py <database_name>")
        return
    
    db_name = sys.argv[1]
    db_path = get_sqlite_db_path(db_name)
    
    if db_path:
        explore_database(db_path)
    else:
        print("Exiting.")

if __name__ == "__main__":
    main() 