import os
import zipfile
import glob
import subprocess
import sqlite3
import psycopg2
import pandas as pd
import sys

# PostgreSQL connection configuration
from mac_sql_agent import DB_CONFIG

def extract_zip_if_needed():
    """Extract the dev_databases.zip file if not already extracted"""
    zip_path = "dev_20240627/dev_databases.zip"
    target_dir = "dev_20240627/dev_databases"
    
    if os.path.exists(target_dir) and os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"Database directory {target_dir} already exists and is not empty")
    elif os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dev_20240627")
        print(f"Extracted to {target_dir}")
    else:
        print(f"Warning: Zip file {zip_path} not found")
    
    return target_dir

def get_database_files(db_dir):
    """Find database files in the specified directory"""
    if not os.path.exists(db_dir):
        print(f"Error: Directory {db_dir} not found")
        return []
    
    # Look for various database file types
    sqlite_files = glob.glob(os.path.join(db_dir, "**", "*.db"), recursive=True)
    sqlite_files += glob.glob(os.path.join(db_dir, "**", "*.sqlite"), recursive=True)
    sqlite_files += glob.glob(os.path.join(db_dir, "**", "*.sqlite3"), recursive=True)
    
    # Also check for SQL dump files
    sql_files = glob.glob(os.path.join(db_dir, "**", "*.sql"), recursive=True)
    
    if sqlite_files:
        print(f"Found {len(sqlite_files)} SQLite database files:")
        for f in sqlite_files:
            print(f" - {f}")
    
    if sql_files:
        print(f"Found {len(sql_files)} SQL files:")
        for f in sql_files:
            print(f" - {f}")
    
    if not sqlite_files and not sql_files:
        # Look for any file and try to identify database files
        all_files = glob.glob(os.path.join(db_dir, "**", "*"), recursive=True)
        print(f"Found {len(all_files)} files (not recognized as databases):")
        for f in all_files[:10]:  # Show just the first 10 to avoid overwhelming output
            print(f" - {f}")
        if len(all_files) > 10:
            print(f"   ... and {len(all_files) - 10} more")
    
    return sqlite_files + sql_files

def import_sqlite_to_postgres(sqlite_file, db_name=None):
    """Import a SQLite database file into PostgreSQL"""
    # Create a database name from the file name if not provided
    if db_name is None:
        db_name = os.path.splitext(os.path.basename(sqlite_file))[0].lower()
        # Remove any characters that aren't valid in PostgreSQL identifiers
        db_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in db_name)
    
    print(f"\nImporting {sqlite_file} to PostgreSQL database {db_name}...")
    
    try:
        # Connect to PostgreSQL
        pg_conn = psycopg2.connect(**DB_CONFIG)
        pg_conn.autocommit = True
        pg_cursor = pg_conn.cursor()
        
        # First check if the database already exists
        pg_cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = pg_cursor.fetchone()
        
        # If database doesn't exist, create it
        if not exists:
            print(f"Creating database {db_name}...")
            # Close existing connection to default database
            pg_conn.close()
            
            # Connect to default postgres database to create new database
            default_config = DB_CONFIG.copy()
            default_config["dbname"] = "postgres"
            pg_conn = psycopg2.connect(**default_config)
            pg_conn.autocommit = True
            pg_cursor = pg_conn.cursor()
            
            # Create the database
            pg_cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database {db_name} created successfully")
            
            # Close connection to default database
            pg_conn.close()
            
            # Connect to the new database
            db_config = DB_CONFIG.copy()
            db_config["dbname"] = db_name
            pg_conn = psycopg2.connect(**db_config)
            pg_conn.autocommit = True
            pg_cursor = pg_conn.cursor()
        
        # Connect to SQLite database
        sqlite_conn = sqlite3.connect(sqlite_file)
        sqlite_cursor = sqlite_conn.cursor()
        
        # Get list of tables
        sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = sqlite_cursor.fetchall()
        
        if not tables:
            print(f"No tables found in SQLite database: {sqlite_file}")
            return
        
        print(f"Found {len(tables)} tables in SQLite database")
        
        # Import each table
        for table_row in tables:
            table_name = table_row[0]
            print(f"Importing table: {table_name}")
            
            # Get table schema
            sqlite_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = sqlite_cursor.fetchall()
            
            # Create table in PostgreSQL
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
            column_defs = []
            
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                
                # Map SQLite types to PostgreSQL types
                pg_type = col_type.upper()
                if pg_type == "INTEGER":
                    pg_type = "INTEGER"
                elif pg_type == "REAL" or pg_type == "FLOAT":
                    pg_type = "REAL"
                elif pg_type.startswith("VARCHAR") or pg_type == "TEXT":
                    pg_type = "TEXT"
                elif pg_type == "BLOB":
                    pg_type = "BYTEA"
                elif pg_type == "BOOLEAN":
                    pg_type = "BOOLEAN"
                elif pg_type.startswith("DATETIME"):
                    pg_type = "TIMESTAMP"
                elif pg_type.startswith("DATE"):
                    pg_type = "DATE"
                elif pg_type.startswith("TIME"):
                    pg_type = "TIME"
                else:
                    # Default to TEXT for unknown types
                    pg_type = "TEXT"
                
                # Build column definition
                col_def = f'"{col_name}" {pg_type}'
                if not_null:
                    col_def += " NOT NULL"
                if is_pk:
                    col_def += " PRIMARY KEY"
                elif default_val is not None:
                    col_def += f" DEFAULT {default_val}"
                
                column_defs.append(col_def)
            
            create_table_sql += ",\n".join(column_defs)
            create_table_sql += "\n);"
            
            # Create table
            try:
                pg_cursor.execute(create_table_sql)
                print(f"Created table: {table_name}")
            except Exception as e:
                print(f"Error creating table {table_name}: {e}")
                continue
            
            # Transfer data
            sqlite_cursor.execute(f"SELECT * FROM {table_name}")
            rows = sqlite_cursor.fetchall()
            
            if rows:
                print(f"Transferring {len(rows)} rows for table {table_name}")
                
                # Prepare column names for insert
                col_names = [f'"{col[1]}"' for col in columns]
                placeholders = ','.join(['%s'] * len(columns))
                
                insert_sql = f'INSERT INTO {table_name} ({",".join(col_names)}) VALUES ({placeholders})'
                
                # Insert in batches
                batch_size = 1000
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i+batch_size]
                    try:
                        pg_cursor.executemany(insert_sql, batch)
                        print(f"Inserted batch {i//batch_size + 1}/{(len(rows)-1)//batch_size + 1}")
                    except Exception as e:
                        print(f"Error inserting data into {table_name}: {e}")
                        break
            
        # Close connections
        sqlite_conn.close()
        pg_conn.close()
        
        print(f"Successfully imported {sqlite_file} to PostgreSQL database {db_name}")
        return db_name
        
    except Exception as e:
        print(f"Error importing {sqlite_file}: {e}")
        return None

def update_db_config_for_evaluation(db_name):
    """Update DB_CONFIG in mac_sql_agent.py to point to the imported database"""
    agent_file = "mac_sql_agent.py"
    
    # Read the current file
    with open(agent_file, 'r') as f:
        content = f.read()
    
    # Find the DB_CONFIG section and update the dbname
    updated_content = content.replace(
        '"dbname": "postgres"',
        f'"dbname": "{db_name}"'
    )
    
    # Write back the updated file
    with open(agent_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated {agent_file} to use database: {db_name}")

def main():
    # Extract zip file if needed
    db_dir = extract_zip_if_needed()
    
    # Get database files
    db_files = get_database_files(db_dir)
    
    if not db_files:
        print("No database files found. Cannot proceed with import.")
        return
    
    # Import databases
    imported_dbs = []
    for db_file in db_files:
        if db_file.lower().endswith(('.db', '.sqlite', '.sqlite3')):
            db_name = import_sqlite_to_postgres(db_file)
            if db_name:
                imported_dbs.append(db_name)
    
    if imported_dbs:
        print(f"\nSuccessfully imported {len(imported_dbs)} databases:")
        for db in imported_dbs:
            print(f" - {db}")
        
        # Ask which database to use for evaluation
        if len(imported_dbs) > 1:
            print("\nWhich database would you like to use for evaluation?")
            for i, db in enumerate(imported_dbs):
                print(f"{i+1}. {db}")
            
            choice = input("Enter the number of your choice: ")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(imported_dbs):
                    selected_db = imported_dbs[idx]
                else:
                    print("Invalid choice. Using the first database.")
                    selected_db = imported_dbs[0]
            except ValueError:
                print("Invalid input. Using the first database.")
                selected_db = imported_dbs[0]
        else:
            selected_db = imported_dbs[0]
        
        # Update DB_CONFIG in mac_sql_agent.py
        update_db_config_for_evaluation(selected_db)
        
        print(f"\nYou can now run the evaluation on database '{selected_db}' with:")
        print(f"python evaluate_benchmark.py --benchmark dev_20240627/dev.json --model meta-llama/Llama-3.3-70B-Instruct-Turbo --num_samples 5")
    else:
        print("No databases were successfully imported.")

if __name__ == "__main__":
    main() 