import os
import sqlite3
import pandas as pd
from langchain_community.utilities import SQLDatabase

class SQLiteAdapter:
    """
    Adapter for SQLite databases to be used with the MAC-SQL agent.
    This provides a consistent interface regardless of the database backend.
    """
    
    def __init__(self, db_path):
        """Initialize the SQLite adapter with the path to the SQLite database"""
        self.db_path = db_path
        self.connection = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Connect to the SQLite database"""
        try:
            # Create SQLDatabase instance for LangChain
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            
            # Create direct connection for custom queries
            self.connection = sqlite3.connect(self.db_path)
            print(f"Successfully connected to SQLite database: {self.db_path}")
            return True
        except Exception as e:
            print(f"Error connecting to SQLite database: {e}")
            return False
    
    def get_tables(self):
        """Get all tables in the database"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    
    def get_table_schema(self, table_name):
        """Get schema information for a specific table"""
        cursor = self.connection.cursor()
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        # Format column information
        col_info = []
        for col in columns:
            col_id, col_name, data_type, not_null, default_value, pk = col
            nullable = "NOT NULL" if not_null else "NULL"
            default = f"DEFAULT {default_value}" if default_value is not None else "NO DEFAULT"
            is_pk = "PRIMARY KEY" if pk else ""
            col_info.append(f"{col_name} ({data_type}, {nullable}, {default}, {is_pk})")
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        fk_info = []
        for fk in cursor.fetchall():
            id, seq, ref_table, from_col, to_col, on_update, on_delete, match = fk
            fk_info.append(f"FOREIGN KEY {from_col} REFERENCES {ref_table}({to_col})")
        
        return {
            "columns": col_info,
            "foreign_keys": fk_info
        }
    
    def execute_query(self, query):
        """Execute a SQL query and return the results as a pandas DataFrame"""
        try:
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def get_full_schema(self):
        """Get the full schema for all tables in the database"""
        tables = self.get_tables()
        schema_info = []
        
        for table in tables:
            schema = self.get_table_schema(table)
            table_info = f"Table: {table}\n"
            table_info += "Columns:\n"
            for col in schema["columns"]:
                table_info += f"  - {col}\n"
            
            if schema["foreign_keys"]:
                table_info += "Foreign Keys:\n"
                for fk in schema["foreign_keys"]:
                    table_info += f"  - {fk}\n"
            
            schema_info.append(table_info)
        
        return "\n".join(schema_info)
    
    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")

def get_sqlite_db_path(db_name, base_dir="dev_20240627/dev_databases"):
    """Get the path to a SQLite database file based on the database name"""
    db_path = os.path.join(base_dir, db_name, f"{db_name}.sqlite")
    if os.path.exists(db_path):
        return db_path
    else:
        print(f"Database file not found: {db_path}")
        return None 