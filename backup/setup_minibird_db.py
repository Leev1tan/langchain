import os
import glob
import subprocess
import psycopg2
from mac_sql_agent import DB_CONFIG

def list_minibird_files():
    """List all files in the mini-bird benchmark directory"""
    minibird_dir = "dev_20240627"
    if not os.path.exists(minibird_dir):
        print(f"Error: Directory {minibird_dir} not found")
        return []
    
    files = os.listdir(minibird_dir)
    print(f"Found {len(files)} files in {minibird_dir}:")
    for file in files:
        print(f" - {os.path.join(minibird_dir, file)}")
    
    return files

def setup_minibird_database():
    """Setup mini-bird databases in PostgreSQL"""
    minibird_dir = "dev_20240627"
    
    # Check for SQL files
    sql_files = glob.glob(os.path.join(minibird_dir, "*.sql"))
    if sql_files:
        print(f"Found {len(sql_files)} SQL files to import:")
        for sql_file in sql_files:
            print(f" - {sql_file}")
        
        # Connect to PostgreSQL
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Import each SQL file
            for sql_file in sql_files:
                print(f"\nImporting {sql_file}...")
                
                # Read SQL file content
                with open(sql_file, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                
                # Execute SQL
                try:
                    cursor.execute(sql_content)
                    print(f"Successfully imported {sql_file}")
                except Exception as e:
                    print(f"Error importing {sql_file}: {e}")
            
            # Verify imported databases and tables
            cursor.execute("""
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name;
            """)
            tables = cursor.fetchall()
            
            print("\nImported tables:")
            for schema, table in tables:
                print(f" - {schema}.{table}")
            
            conn.close()
            
        except Exception as e:
            print(f"Database connection error: {e}")
            
    else:
        # Check for dump files
        dump_files = glob.glob(os.path.join(minibird_dir, "*.dump"))
        if dump_files:
            print(f"Found {len(dump_files)} PostgreSQL dump files:")
            for dump_file in dump_files:
                print(f" - {dump_file}")
            
            # Import dump files using pg_restore
            for dump_file in dump_files:
                print(f"\nImporting {dump_file}...")
                
                # Use pg_restore command
                cmd = [
                    "pg_restore",
                    "-h", DB_CONFIG["host"],
                    "-p", DB_CONFIG["port"],
                    "-U", DB_CONFIG["user"],
                    "-d", DB_CONFIG["dbname"],
                    "-v",
                    dump_file
                ]
                
                try:
                    # Set PGPASSWORD environment variable
                    env = os.environ.copy()
                    env["PGPASSWORD"] = DB_CONFIG["password"]
                    
                    # Run pg_restore
                    process = subprocess.run(cmd, env=env, check=True, 
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Successfully imported {dump_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Error importing {dump_file}:")
                    print(f"stdout: {e.stdout.decode('utf-8')}")
                    print(f"stderr: {e.stderr.decode('utf-8')}")
                    
        else:
            # Check for other database files
            print("No SQL or dump files found. Checking for other database files...")
            
            db_files = glob.glob(os.path.join(minibird_dir, "*.db")) + \
                       glob.glob(os.path.join(minibird_dir, "*.sqlite")) + \
                       glob.glob(os.path.join(minibird_dir, "*.sqlite3"))
            
            if db_files:
                print(f"Found {len(db_files)} SQLite database files:")
                for db_file in db_files:
                    print(f" - {db_file}")
                print("\nNote: SQLite files need to be converted to PostgreSQL format.")
                print("Please use a tool like sqlite3-to-postgresql to convert them.")
            else:
                print("No database files found in the mini-bird directory.")
                print("Check if files need to be extracted from an archive.")

if __name__ == "__main__":
    list_minibird_files()
    setup_minibird_database() 