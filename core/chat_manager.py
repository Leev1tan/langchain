"""
Chat Manager for MAC-SQL
"""

import logging
import psycopg2
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_together import Together
import os
import json
import time
import sqlite3
import dotenv

# Load environment variables from .env file if present
dotenv.load_dotenv()

from core.agents import SelectorAgent, DecomposerAgent, RefinerAgent
from core.config import DB_CONFIG

# PostgreSQL connection configuration 
DB_CONFIG = {
    "dbname": os.environ.get("POSTGRES_DB", "postgres"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}

logger = logging.getLogger(__name__)

class ChatManager:
    """
    Manages the communication between the three MAC-SQL agents.
    
    Responsibilities:
    1. Initialize and manage the three agents
    2. Maintain conversation memory
    3. Coordinate the workflow between agents
    4. Handle database interactions
    """
    
    def __init__(
            self,
            model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key: Optional[str] = None,
            db_name: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: int = 1024,
            **kwargs
        ):
        """Initialize the ChatManager with model name and API key."""
        self.model_name = model_name
        self.api_key = api_key
        self.db_name = db_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.connection = None
        self.cur = None
        self._schema_knowledge = None
        self.schema_knowledge = ""
        self.db_config = DB_CONFIG.copy()
        self.query_examples = []  # Will store successful query examples
        
        # Connect to database if db_name is provided
        if self.db_name:
            self.connect_to_database(self.db_name)
    
    @property
    def connected(self):
        """Check if the database connection is active."""
        return self.connection is not None

    def connect_to_database(self, db_name):
        """Connect to the specified database."""
        try:
            # Close existing connection if any
            if self.connection:
                self.connection.close()
                self.connection = None
                self.cur = None
            
            # Check if this is a SQLite database (file with .db, .sqlite, .sqlite3 extension)
            is_sqlite = db_name.endswith(('.db', '.sqlite', '.sqlite3')) or os.path.isfile(db_name)
            
            if is_sqlite:
                # Connect to SQLite database
                self.connection = sqlite3.connect(db_name)
                self.db_name = db_name
                self.cur = self.connection.cursor()
                # Initialize schema for SQLite
                self._initialize_sqlite_schema()
            else:
                # Connect to PostgreSQL database
                # Update database name in config
                self.db_config["dbname"] = db_name
                
                # Connect to PostgreSQL
                self.connection = psycopg2.connect(**self.db_config)
                self.db_name = db_name
                self.cur = self.connection.cursor()
                # Initialize schema for PostgreSQL
                self._initialize_schema_knowledge()
            
            print(f"Successfully connected to database: {db_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            print(f"Error connecting to database: {e}")
            return False

    def _initialize_sqlite_schema(self):
        """Initialize schema knowledge for SQLite database."""
        import time
        
        if not self.connection:
            logger.error("Cannot initialize schema: Not connected to database")
            return False
        
        try:
            # Get list of tables
            self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [table[0] for table in self.cur.fetchall()]
            
            if not tables:
                # Try again with a different query
                self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [table[0] for table in self.cur.fetchall() if not table[0].startswith('sqlite_')]
            
            if not tables:
                logger.warning("No tables found in database")
                # Create a dummy schema to allow querying to continue
                self.schema_knowledge = "This database appears to be empty or its schema cannot be accessed."
                return False
            
            # Build schema information
            schema_info = []
            
            for table in tables:
                try:
                    # Get table info
                    self.cur.execute(f"PRAGMA table_info({table});")
                    columns = self.cur.fetchall()
                    
                    if not columns:
                        logger.warning(f"No columns found for table {table}")
                        continue
                    
                    table_info = f"Table: {table}\nColumns:\n"
                    primary_keys = []
                    
                    for col in columns:
                        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
                        col_id, col_name, col_type, not_null, default_val, is_pk = col
                        
                        # Build column description
                        nullable = "" if not_null else " NULL"
                        default = f" DEFAULT {default_val}" if default_val else ""
                        pk_indicator = " (PRIMARY KEY)" if is_pk else ""
                        
                        if is_pk:
                            primary_keys.append(col_name)
                        
                        table_info += f"  - {col_name}: {col_type}{pk_indicator}{nullable}{default}\n"
                    
                    # Get foreign keys
                    self.cur.execute(f"PRAGMA foreign_key_list({table});")
                    foreign_keys = self.cur.fetchall()
                    
                    if foreign_keys:
                        table_info += "Foreign Keys:\n"
                        for fk in foreign_keys:
                            # PRAGMA foreign_key_list returns: id, seq, table, from, to, on_update, on_delete, match
                            _, _, target_table, from_col, to_col, _, _, _ = fk
                            table_info += f"  - {from_col} -> {target_table}.{to_col}\n"
                    
                    # Get sample data
                    try:
                        # Use LIMIT with a reasonable value
                        self.cur.execute(f"SELECT * FROM [{table}] LIMIT 3;")
                        sample_data = self.cur.fetchall()
                        
                        if sample_data:
                            # Get column names
                            col_names = [description[0] for description in self.cur.description]
                            
                            table_info += "Sample Data (up to 3 rows):\n"
                            for row in sample_data:
                                row_values = []
                                for i, value in enumerate(row):
                                    if value is None:
                                        row_values.append(f"{col_names[i]}=NULL")
                                    else:
                                        row_values.append(f"{col_names[i]}={value}")
                                
                                table_info += f"  - {', '.join(row_values)}\n"
                    except Exception as e:
                        logger.warning(f"Error getting sample data for table {table}: {e}")
                    
                    schema_info.append(table_info)
                except Exception as table_error:
                    logger.warning(f"Error processing table {table}: {table_error}")
            
            # Join all schema information
            if schema_info:
                self.schema_knowledge = "\n".join(schema_info)
                logger.info(f"Successfully initialized schema with {len(tables)} tables and {len(schema_info)} table descriptions")
                print(f"Schema initialized with {len(tables)} tables")
                return True
            else:
                # No schema info available
                self.schema_knowledge = "Schema information could not be retrieved."
                logger.warning("No schema information could be retrieved")
                return False
        
        except Exception as e:
            logger.error(f"Error initializing SQLite schema: {e}")
            self.schema_knowledge = f"Error initializing schema: {str(e)}"
            print(f"Error initializing schema: {str(e)}")
            return False
    
    def _initialize_schema_knowledge(self):
        """Initialize schema knowledge by extracting database schema information from PostgreSQL."""
        if not self.connected:
            logger.error("Cannot initialize schema knowledge: Not connected to database")
            return False
        
        try:
            # Get all tables with retry mechanism
            max_retries = 3
            retry_count = 0
            tables = []
            
            while retry_count < max_retries and not tables:
                try:
                    self.cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """)
                    tables = [table[0] for table in self.cur.fetchall()]
                    
                    if not tables and retry_count < max_retries - 1:
                        logger.warning(f"No tables found, retrying... (attempt {retry_count+1})")
                        time.sleep(1)  # Wait a bit before retrying
                except Exception as e:
                    logger.error(f"Error fetching tables (attempt {retry_count+1}): {e}")
                    if retry_count < max_retries - 1:
                        time.sleep(1)  # Wait a bit before retrying
                
                retry_count += 1
            
            if not tables:
                logger.error("Failed to retrieve any tables after retries")
                return False
            
            # Get column information for each table
            schema_info = []
            for table in tables:
                try:
                    self.cur.execute(f"""
                        SELECT 
                            column_name, 
                            data_type, 
                            is_nullable,
                            column_default
                        FROM 
                            information_schema.columns 
                        WHERE 
                            table_schema = 'public' AND 
                            table_name = '{table}'
                    """)
                    
                    columns = []
                    for col in self.cur.fetchall():
                        column_name, data_type, is_nullable, default = col
                        columns.append({
                            "name": column_name,
                            "type": data_type,
                            "nullable": is_nullable == "YES",
                            "default": default
                        })
                    
                    # Get primary keys
                    self.cur.execute(f"""
                        SELECT 
                            kcu.column_name
                        FROM 
                            information_schema.table_constraints tc
                        JOIN 
                            information_schema.key_column_usage kcu
                        ON 
                            tc.constraint_name = kcu.constraint_name
                        WHERE 
                            tc.constraint_type = 'PRIMARY KEY' AND
                            tc.table_schema = 'public' AND
                            tc.table_name = '{table}'
                    """)
                    
                    primary_keys = [pk[0] for pk in self.cur.fetchall()]
                    
                    # Get foreign keys
                    self.cur.execute(f"""
                        SELECT
                            kcu.column_name,
                            ccu.table_name AS foreign_table_name,
                            ccu.column_name AS foreign_column_name
                        FROM
                            information_schema.table_constraints AS tc
                        JOIN
                            information_schema.key_column_usage AS kcu
                        ON
                            tc.constraint_name = kcu.constraint_name
                        JOIN
                            information_schema.constraint_column_usage AS ccu
                        ON
                            ccu.constraint_name = tc.constraint_name
                        WHERE
                            tc.constraint_type = 'FOREIGN KEY' AND
                            tc.table_schema = 'public' AND
                            tc.table_name = '{table}'
                    """)
                    
                    foreign_keys = []
                    for fk in self.cur.fetchall():
                        column_name, foreign_table, foreign_column = fk
                        foreign_keys.append({
                            "column": column_name,
                            "foreign_table": foreign_table,
                            "foreign_column": foreign_column
                        })
                    
                    # Get sample data with error handling
                    sample_rows = []
                    try:
                        # Use a 2-second timeout for sample data queries
                        self.cur.execute(f"SET statement_timeout = 2000")
                        self.cur.execute(f'SELECT * FROM "{table}" LIMIT 5')
                        sample_data = self.cur.fetchall()
                        col_names = [desc[0] for desc in self.cur.description]
                        
                        for row in sample_data:
                            sample_row = {}
                            for i, value in enumerate(row):
                                if value is None:
                                    sample_row[col_names[i]] = "NULL"
                                elif isinstance(value, (str, int, float, bool)):
                                    sample_row[col_names[i]] = str(value)
                                else:
                                    # Handle other types safely
                                    try:
                                        sample_row[col_names[i]] = str(value)
                                    except:
                                        sample_row[col_names[i]] = f"<{type(value).__name__}>"
                            sample_rows.append(sample_row)
                        
                        # Reset timeout
                        self.cur.execute("SET statement_timeout = 0")
                    except Exception as e:
                        logger.warning(f"Could not get sample data for table {table}: {e}")
                        # Reset timeout
                        try:
                            self.cur.execute("SET statement_timeout = 0")
                        except:
                            pass
                    
                    schema_info.append({
                        "table_name": table,
                        "columns": columns,
                        "primary_keys": primary_keys,
                        "foreign_keys": foreign_keys,
                        "sample_data": sample_rows
                    })
                except Exception as e:
                    logger.error(f"Error processing schema for table {table}: {e}")
            
            # Format schema info for LLM
            formatted_schema = []
            for table_info in schema_info:
                table_desc = f"Table: {table_info['table_name']}\n"
                
                # Columns
                table_desc += "Columns:\n"
                for col in table_info['columns']:
                    pk_indicator = " (PRIMARY KEY)" if col['name'] in table_info['primary_keys'] else ""
                    nullable = "" if col['nullable'] else " NOT NULL"
                    default = f" DEFAULT {col['default']}" if col['default'] else ""
                    table_desc += f"  - {col['name']}: {col['type']}{pk_indicator}{nullable}{default}\n"
                
                # Foreign Keys
                if table_info['foreign_keys']:
                    table_desc += "Foreign Keys:\n"
                    for fk in table_info['foreign_keys']:
                        table_desc += f"  - {fk['column']} -> {fk['foreign_table']}.{fk['foreign_column']}\n"
                
                # Sample Data
                if table_info['sample_data']:
                    table_desc += "Sample Data (up to 5 rows):\n"
                    for row in table_info['sample_data']:
                        row_values = [f"{k}={v}" for k, v in row.items()]
                        table_desc += f"  - {', '.join(row_values)}\n"
                
                formatted_schema.append(table_desc)
            
            final_schema = "\n".join(formatted_schema)
            self.schema_knowledge = final_schema
            print(f"Successfully initialized PostgreSQL schema with {len(tables)} tables")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing schema knowledge: {e}")
            return False
    
    def execute_sql_query(self, query):
        """Execute a SQL query on the connected database."""
        if not self.connection:
            logger.error("Cannot execute query: Not connected to database")
            return {'success': False, 'error': 'Not connected to database'}
        
        try:
            # Create a cursor for this query
            cursor = self.connection.cursor()
            
            # Execute the query
            cursor.execute(query)
            
            # Check if the query returns results
            if cursor.description is not None:
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Fetch results
                rows = cursor.fetchall()
                
                # Commit any changes (this is safe even for SELECT queries)
                self.connection.commit()
                
                return {
                    'success': True,
                    'columns': columns,
                    'rows': rows,
                    'row_count': len(rows)
                }
            else:
                # No results (likely an INSERT, UPDATE, DELETE)
                # Commit the changes
                self.connection.commit()
                
                return {
                    'success': True,
                    'columns': [],
                    'rows': [],
                    'row_count': 0
                }
        
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            
            # Rollback on error
            try:
                self.connection.rollback()
                print(f"Transaction rolled back due to error: {e}")
            except Exception as rollback_error:
                print(f"Error during rollback: {rollback_error}")
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_llm_response(self, prompt: str) -> str:
        """
        Generate a response from the language model.
        
        Args:
            prompt: The prompt to send to the language model
            
        Returns:
            Response from the language model
        """
        try:
            # Initialize the LLM if not already done
            if not hasattr(self, 'llm'):
                from langchain_together import Together
                from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
                
                # Get API key with proper priority order:
                # 1. Constructor-provided api_key
                # 2. Environment variable TOGETHER_API_KEY
                # 3. Fall back to None (will cause an error if not provided elsewhere)
                api_key = self.api_key
                if api_key is None:
                    api_key = os.environ.get("TOGETHER_API_KEY")
                    if api_key is None:
                        raise ValueError(
                            "No API key provided. Please provide an API key via the constructor "
                            "or set the TOGETHER_API_KEY environment variable."
                        )
                
                # Initialize the LLM
                self.llm = Together(
                    model=self.model_name,
                    temperature=self.temperature,
                    together_api_key=api_key,
                    max_tokens=self.max_tokens,
                    **self.kwargs
                )
            
            # Generate response
            from langchain_core.messages import HumanMessage
            
            # Handle the prompt as either a string or a message
            if isinstance(prompt, str):
                response = self.llm.invoke(prompt)
            else:
                response = self.llm.invoke(prompt)
            
            return response
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Add retry mechanism for common errors like rate limiting
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                import time
                logger.info("Rate limit hit, waiting 5 seconds and retrying...")
                time.sleep(5)
                return self.generate_llm_response(prompt)
            raise 