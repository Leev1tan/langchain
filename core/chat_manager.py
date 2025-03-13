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

from core.agents import SelectorAgent, DecomposerAgent, RefinerAgent
from core.config import DB_CONFIG

# PostgreSQL connection configuration (from int.py)
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "superuser",
    "host": "localhost",
    "port": "5432"
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
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """Initialize the chat manager."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set default API key if none provided
        if api_key is None:
            api_key = "6e4593b7c0e0279476b65f144273d1ee972a47e3eb543c9649b36aaf6c114a82"
        
        # Initialize the LLM
        self.llm = Together(
            model=model_name,
            temperature=temperature,
            together_api_key=api_key,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        # Setup agents with basic parameters
        self.selector_agent = SelectorAgent(model_name, api_key)
        self.decomposer_agent = DecomposerAgent(model_name, api_key)
        self.refiner_agent = RefinerAgent(model_name, api_key)
        
        # Setup memory (using message list instead of ConversationBufferMemory)
        self.messages = []
        
        # Setup database configuration
        self.db_config = DB_CONFIG.copy()
        self.db_name = None
        self.connection = None
        self.cur = None
        self.connected = False
        
        # Initialize schema knowledge
        schema_info = self._initialize_schema_knowledge()
        if isinstance(schema_info, bool) and not schema_info:
            # If initialization failed, set empty schema
            self.schema_knowledge = ""
            logger.warning("Failed to initialize schema knowledge, using empty schema")
        else:
            # Otherwise use the returned schema
            self.schema_knowledge = schema_info
        
        self.query_examples = []  # Will store successful query examples
        
        # Connect to database if db_name is provided
        if self.db_name:
            self._connect_to_database()
    
    def update_database(self, db_name):
        """
        Update the database connection to a new database
        
        Args:
            db_name: Name of the database to connect to
            
        Returns:
            Boolean indicating success
        """
        # Close existing connection if any
        if self.connection:
            try:
                self.connection.close()
                print(f"Closed connection to database: {self.db_name}")
            except Exception as e:
                print(f"Error closing connection: {e}")
        
        # Update database name
        self.db_name = db_name
        self.db_config["dbname"] = db_name
        
        # Check if we already have cached schema for this database on disk
        schema_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'schema_cache')
        schema_cache_file = os.path.join(schema_cache_dir, f"{db_name}_schema.json")
        
        # Try to load from disk cache first
        schema_info = None
        
        if os.path.exists(schema_cache_file):
            try:
                with open(schema_cache_file, 'r') as f:
                    schema_data = json.load(f)
                    if 'schema' in schema_data and 'timestamp' in schema_data:
                        # Check if cache is valid (less than 24 hours old)
                        cache_age = time.time() - schema_data['timestamp']
                        if cache_age < 86400:  # 24 hours
                            schema_info = schema_data['schema']
                            logger.info(f"Loaded schema for {db_name} from disk cache (age: {cache_age/3600:.1f} hours)")
                            print(f"Using cached schema for database: {db_name}")
            except Exception as e:
                logger.warning(f"Failed to load schema cache from disk: {e}")
        
        # Connect to the database
        if self._connect_to_database():
            # If we don't have valid cache, initialize schema knowledge
            if not schema_info:
                logger.info(f"Initializing schema knowledge for {db_name}")
                print(f"Extracting schema for database: {db_name}")
                
                # Try full schema extraction first
                full_schema = self._initialize_schema_knowledge()
                
                if full_schema and not isinstance(full_schema, bool):
                    schema_info = full_schema
                    self._save_schema_to_cache(schema_info)
                    logger.info(f"Successfully initialized full schema for {db_name}")
                else:
                    # If full extraction failed, try the simple approach
                    logger.warning(f"Full schema extraction failed for {db_name}, trying simple extraction")
                    try:
                        simple_schema = self._get_simple_schema_info()
                        if simple_schema and isinstance(simple_schema, str):
                            schema_info = simple_schema
                            self._save_schema_to_cache(schema_info)
                            logger.info(f"Used simple schema extraction for {db_name}")
                        else:
                            schema_info = ""
                            logger.warning(f"Even simple schema extraction failed for {db_name}")
                    except Exception as e:
                        schema_info = ""
                        logger.error(f"Error in simple schema extraction: {e}")
            
            # Set the schema knowledge
            self.schema_knowledge = schema_info if schema_info else ""
            print(f"Successfully connected to database: {db_name}")
            return True
        else:
            self.schema_knowledge = ""
            print(f"Failed to connect to database: {db_name}")
            return False
    
    def _connect_to_database(self):
        """Connect to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                dbname=self.db_config.get("dbname", "postgres"),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", "postgres"),
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", "5432")
            )
            self.cur = self.connection.cursor()
            self.connected = True
            logger.info("Successfully connected to PostgreSQL database.")
            return True
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            self.connected = False
            return False
    
    def _initialize_schema_knowledge(self):
        """Initialize schema knowledge by extracting database schema information."""
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
                        self.cur.execute(f"SELECT * FROM {table} LIMIT 5")
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
            return final_schema
        
        except Exception as e:
            logger.error(f"Error initializing schema knowledge: {e}")
            return False
    
    def _get_formatted_history(self) -> str:
        """
        Get formatted conversation history
        
        Returns:
            String representation of the conversation history
        """
        if not self.messages:
            return ""
        
        formatted_history = ""
        for message in self.messages:
            if isinstance(message, HumanMessage):
                formatted_history += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"AI: {message.content}\n"
            elif isinstance(message, SystemMessage):
                formatted_history += f"System: {message.content}\n"
        
        return formatted_history
    
    def execute_sql_query(self, query):
        """Execute a SQL query on the connected database."""
        if not self.connected:
            logger.error("Cannot execute query: Not connected to database")
            return {'error': 'Not connected to database'}
        
        try:
            # Check if we need to ROLLBACK first (if query doesn't already start with ROLLBACK)
            if not query.strip().upper().startswith("ROLLBACK"):
                try:
                    # Check transaction status - a simple query that will fail if transaction is aborted
                    self.cur.execute("SELECT 1")
                except Exception as e:
                    if "current transaction is aborted" in str(e).lower():
                        print("Transaction is aborted. Executing ROLLBACK before continuing.")
                        # Get a new connection to reset the transaction
                        self.connection.rollback()
            
            # Now execute the actual query
            self.cur.execute(query)
            
            # Get column names
            columns = [desc[0] for desc in self.cur.description] if self.cur.description else []
            
            # Fetch results
            results = self.cur.fetchall()
            
            # Commit the transaction if it was a successful query
            self.connection.commit()
            
            # Format results as dictionary
            formatted_results = []
            for row in results:
                formatted_row = {}
                for i, value in enumerate(row):
                    formatted_row[columns[i]] = value
                formatted_results.append(formatted_row)
            
            return {
                'success': True,
                'column_names': columns,
                'rows': formatted_results,
                'row_count': len(formatted_results)
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
    
    def process_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a SQL query directly and return the results.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Dictionary with query results
        """
        import pandas as pd
        
        if not self.connection:
            return {"success": False, "error": "Not connected to database"}
        
        try:
            # Execute the query
            result = pd.read_sql_query(sql_query, self.connection)
            return {"success": True, "result": result}
        except Exception as e:
            error_message = str(e)
            print(f"Error executing query: {error_message}")
            return {"success": False, "error": error_message}
    
    def _get_simple_schema_info(self):
        """Get simplified schema information when full extraction fails"""
        if not self.connected:
            return ""
        
        try:
            # Get all tables
            self.cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [table[0] for table in self.cur.fetchall()]
            
            if not tables:
                return ""
            
            # Build simple schema information
            schema_info = []
            for table in tables:
                # Get columns
                self.cur.execute(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = '{table}'
                """)
                columns = self.cur.fetchall()
                
                table_info = f"Table: {table}\nColumns:\n"
                for col_name, col_type in columns:
                    table_info += f"  - {col_name}: {col_type}\n"
                
                # Get sample row if available
                try:
                    self.cur.execute(f"SELECT * FROM {table} LIMIT 1")
                    sample = self.cur.fetchone()
                    if sample:
                        col_names = [desc[0] for desc in self.cur.description]
                        table_info += "Sample Data (1 row):\n  - "
                        sample_values = []
                        for i, val in enumerate(sample):
                            sample_values.append(f"{col_names[i]}={val}" if val is not None else f"{col_names[i]}=NULL")
                        table_info += ", ".join(sample_values) + "\n"
                except:
                    pass
                
                schema_info.append(table_info)
            
            return "\n".join(schema_info)
        except Exception as e:
            logger.error(f"Error in simple schema extraction: {e}")
            return ""
    
    def _save_schema_to_cache(self, schema):
        """Save schema information to disk cache"""
        if not self.db_name:
            return
        
        try:
            schema_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'schema_cache')
            os.makedirs(schema_cache_dir, exist_ok=True)
            
            schema_cache_file = os.path.join(schema_cache_dir, f"{self.db_name}_schema.json")
            
            cache_data = {
                'schema': schema,
                'timestamp': time.time(),
                'db_name': self.db_name
            }
            
            with open(schema_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved schema for {self.db_name} to disk cache")
        except Exception as e:
            logger.warning(f"Failed to save schema to disk cache: {e}") 