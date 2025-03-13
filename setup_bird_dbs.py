#!/usr/bin/env python
"""
Setup Script for BIRD Benchmark PostgreSQL Databases
==================================================

This script sets up the required PostgreSQL databases for the BIRD benchmark.
It reads the database IDs from the benchmark file and creates the databases.
This script focuses exclusively on PostgreSQL databases and executes PostgreSQL SQL scripts.
It can also use an existing BIRD database as a source if available.

Usage:
    python setup_bird_dbs.py [--benchmark BENCHMARK_FILE] [--db DB_NAMES [DB_NAMES ...]]
"""

import os
import json
import argparse
import subprocess
import psycopg2
from tqdm import tqdm

# Import DB configuration
from core.config import DB_CONFIG

# Constants
DEFAULT_BENCHMARK_PATH = "minidev/MINIDEV/mini_dev_postgresql.json"
SQL_SCRIPT_PATH = "minidev/MINIDEV_postgresql/BIRD_dev.sql"
DEV_DATABASES_DIR = "minidev/MINIDEV_postgresql"
BIRD_DATABASE = "BIRD"  # Name of the main BIRD database

def load_benchmark_data(benchmark_path):
    """Load benchmark data from JSON file"""
    try:
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return []

def get_database_ids(benchmark_data):
    """Extract unique database IDs from benchmark data"""
    db_ids = set()
    for item in benchmark_data:
        db_id = item.get("db_id", "")
        if db_id:
            db_ids.add(db_id)
    return sorted(list(db_ids))

def filter_database_ids(db_ids, db_filter):
    """Filter database IDs based on user input"""
    if not db_filter:
        return db_ids
    return [db_id for db_id in db_ids if db_id.lower() in [db.lower() for db in db_filter]]

def connect_to_postgres(database="postgres"):
    """Connect to PostgreSQL server"""
    try:
        # Connect to the specified database
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=database
        )
        conn.autocommit = True  # Enable autocommit mode for creating databases
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL database {database}: {e}")
        return None

def database_exists(conn, db_name):
    """Check if a database exists"""
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        return cur.fetchone() is not None
    except Exception as e:
        print(f"Error checking if database exists: {e}")
        return False

def check_bird_database():
    """Check if the BIRD database exists and is properly set up"""
    conn = connect_to_postgres()
    if not conn:
        return False
    
    try:
        exists = database_exists(conn, BIRD_DATABASE)
        if exists:
            print(f"Found existing BIRD database: {BIRD_DATABASE}")
            # Try to connect to the BIRD database to verify it's accessible
            bird_conn = connect_to_postgres(BIRD_DATABASE)
            if bird_conn:
                bird_conn.close()
                return True
        return False
    except Exception as e:
        print(f"Error checking BIRD database: {e}")
        return False
    finally:
        conn.close()

def create_database(conn, db_name):
    """Create a PostgreSQL database"""
    try:
        cur = conn.cursor()
        
        # First drop the database if it exists
        cur.execute(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{db_name}'")
        cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
        
        # Create the database
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"Created database: {db_name}")
        return True
    except Exception as e:
        print(f"Error creating database {db_name}: {e}")
        return False

def find_database_script(db_name):
    """Find a PostgreSQL script for the database"""
    # Check for database SQL dumps in the default location
    script_path = os.path.join(DEV_DATABASES_DIR, f"{db_name}.sql")
    if os.path.exists(script_path):
        print(f"Found PostgreSQL script: {script_path}")
        return script_path
    
    # Check in the BIRD dev SQL file
    try:
        if os.path.exists(SQL_SCRIPT_PATH):
            with open(SQL_SCRIPT_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if the database is defined in the SQL file
            if f"-- Database: {db_name}" in content:
                return extract_database_script(content, db_name)
    except Exception as e:
        print(f"Error searching for database script: {e}")
    
    print(f"No PostgreSQL script found for {db_name}")
    return None

def extract_database_script(content, db_name):
    """Extract database script from the BIRD SQL file"""
    # Create a temporary file with the extracted script
    start_marker = f"-- Database: {db_name}"
    end_marker = "-- Database:"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print(f"Database {db_name} not found in SQL script")
        return None
    
    # Find the next database section
    next_db_idx = content.find(end_marker, start_idx + len(start_marker))
    if next_db_idx == -1:
        # If this is the last database, take all remaining content
        script = content[start_idx:]
    else:
        script = content[start_idx:next_db_idx]
    
    # Write to temp file
    temp_file = f"temp_{db_name}.sql"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(script)
    
    return temp_file

def setup_database_schema(db_name):
    """Set up the schema for a database using PostgreSQL scripts"""
    # Find the right SQL script for this database
    script_path = find_database_script(db_name)
    
    if not script_path:
        print(f"No script available for {db_name}")
        # Check if we can use the BIRD database as a source
        if check_bird_database():
            print(f"Using BIRD database as a source for {db_name}")
            return setup_from_bird_database(db_name)
        return False
    
    try:
        # Use psql to execute the script
        cmd = [
            "psql",
            "-h", DB_CONFIG["host"],
            "-p", DB_CONFIG["port"],
            "-U", DB_CONFIG["user"],
            "-d", db_name,
            "-f", script_path
        ]
        
        # Set PGPASSWORD environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = DB_CONFIG["password"]
        
        # Execute the command
        result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error executing SQL script: {result.stderr.decode()}")
            return False
        
        print(f"Successfully set up schema for database: {db_name}")
        
        # Clean up temp file if it was created
        if script_path.startswith("temp_") and os.path.exists(script_path):
            os.remove(script_path)
        
        return True
    except Exception as e:
        print(f"Error setting up schema for {db_name}: {e}")
        return False

def setup_from_bird_database(db_name):
    """Set up a database using the BIRD database as a source"""
    try:
        # Connect to the BIRD database
        bird_conn = connect_to_postgres(BIRD_DATABASE)
        if not bird_conn:
            return False
        
        # Get a list of tables in the BIRD database that belong to the target database
        bird_cur = bird_conn.cursor()
        bird_cur.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '{db_name}_%'")
        tables = [row[0] for row in bird_cur.fetchall()]
        
        if not tables:
            print(f"No tables found for {db_name} in the BIRD database")
            bird_conn.close()
            return False
        
        # Connect to the target database
        target_conn = connect_to_postgres(db_name)
        if not target_conn:
            bird_conn.close()
            return False
        
        target_cur = target_conn.cursor()
        
        # Copy each table to the target database
        for table in tables:
            # Get the table schema
            bird_cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'")
            columns = bird_cur.fetchall()
            
            # Create the table in the target database
            column_defs = ", ".join([f"{col[0]} {col[1]}" for col in columns])
            target_table = table.replace(f"{db_name}_", "", 1)  # Remove the database prefix
            target_cur.execute(f"CREATE TABLE IF NOT EXISTS {target_table} ({column_defs})")
            
            # Copy the data
            bird_cur.execute(f"SELECT * FROM {table}")
            rows = bird_cur.fetchall()
            
            if rows:
                placeholders = ", ".join(["%s"] * len(columns))
                column_names = ", ".join([col[0] for col in columns])
                for row in rows:
                    target_cur.execute(f"INSERT INTO {target_table} ({column_names}) VALUES ({placeholders})", row)
        
        target_conn.commit()
        print(f"Successfully set up {db_name} from BIRD database")
        
        # Close connections
        target_conn.close()
        bird_conn.close()
        return True
    except Exception as e:
        print(f"Error setting up {db_name} from BIRD database: {e}")
        return False

def setup_databases(benchmark_path, db_filter=None):
    """Set up databases based on benchmark data"""
    # Load benchmark data
    benchmark_data = load_benchmark_data(benchmark_path)
    if not benchmark_data:
        print("No benchmark data loaded, aborting setup")
        return False
    
    # Get database IDs
    db_ids = get_database_ids(benchmark_data)
    print(f"Found {len(db_ids)} databases in benchmark data")
    
    # Filter databases if specified
    if db_filter:
        db_ids = filter_database_ids(db_ids, db_filter)
        print(f"Filtered to {len(db_ids)} databases: {', '.join(db_ids)}")
    
    # Check if BIRD database exists
    has_bird = check_bird_database()
    if has_bird:
        print("Found BIRD database, will use it as a source if needed")
    
    # Connect to PostgreSQL
    conn = connect_to_postgres()
    if not conn:
        print("Failed to connect to PostgreSQL, aborting setup")
        return False
    
    # Process each database
    try:
        for db_id in tqdm(db_ids, desc="Setting up databases"):
            print(f"\nProcessing database: {db_id}")
            
            # Create database
            success = create_database(conn, db_id)
            if not success:
                print(f"Failed to create database: {db_id}, skipping...")
                continue
            
            # Set up schema using PostgreSQL script or BIRD database
            setup_database_schema(db_id)
        
        print("\nDatabase setup completed successfully")
        return True
    except Exception as e:
        print(f"Error during database setup: {e}")
        return False
    finally:
        # Close connection
        if conn:
            conn.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Set up BIRD benchmark PostgreSQL databases")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_PATH,
                      help="Path to benchmark file (JSON)")
    parser.add_argument("--db", nargs="+", default=None,
                      help="Filter to specific databases")
    args = parser.parse_args()
    
    # Run setup
    setup_databases(args.benchmark, args.db)

if __name__ == "__main__":
    main() 