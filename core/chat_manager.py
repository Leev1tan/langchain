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
import traceback

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

# Constants for agent communication
SYSTEM_NAME = "System"
SELECTOR_NAME = "SelectorAgent"
DECOMPOSER_NAME = "DecomposerAgent"
REFINER_NAME = "RefinerAgent"
MAX_ROUNDS = 3

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
            verbose: bool = False
    ):
        """
        Initialize the chat manager
        
        Args:
            model_name: Name of the LLM to use
            api_key: API key for the LLM
            db_name: Name of the database to connect to
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
            verbose: Whether to print verbose output
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.db_name = db_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = Together(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            together_api_key=self.api_key
        )
        
        # Initialize database connection
        self.db_conn = None
        if db_name:
            self.connect_to_database(db_name)
            
        # Initialize agents
        self.selector_agent = SelectorAgent(self)
        self.decomposer_agent = DecomposerAgent(self)
        self.refiner_agent = RefinerAgent(self)
        
        # Schema cache
        self.schema_cache = {}
        
        # Message history
        self.message_history = []

    def process_query(self, user_query, db_id=None, evidence="") -> Dict[str, Any]:
        """
        Process a natural language query using the MAC-SQL agent workflow
        
        This method implements the agent workflow similar to the original MAC-SQL:
        1. Selector agent identifies relevant schema elements
        2. Decomposer agent creates the query plan
        3. Refiner agent generates and refines SQL
        
        Args:
            user_query: The natural language query to process
            db_id: Optional database ID to use (defaults to current db_name)
            evidence: Optional additional context information
            
        Returns:
            Dictionary with results including the generated SQL, execution results, etc.
        """
        # Handle database connection
        if db_id and db_id != self.db_name:
            logger.info(f"Connecting to database: {db_id}")
            if not self.connect_to_database(db_id):
                logger.warning(f"Failed to connect to database: {db_id}, will try to use hardcoded schema")
                # Even if connection fails, we'll continue with hardcoded schema if available
        
        # Get schema information (will use hardcoded schema if database connection fails)
        schema_info = self.get_schema()
        
        if not schema_info:
            logger.error(f"No schema available for database: {self.db_name}")
            return {
                'success': False,
                'error': f"No schema available for database: {self.db_name}",
                'query': user_query,
                'sql': '',
                'understanding': '',
                'plan': '',
                'db_id': self.db_name
            }
        
        start_time = time.time()
        print(f"\n[ChatManager] Processing query: {user_query}")
        
        # Check if evidence is provided (common in benchmark evaluations)
        if evidence:
            print(f"[ChatManager] Evidence provided: {evidence}")
            # Incorporate evidence into the user query if it contains useful information
            enhanced_query = user_query
            if evidence and len(evidence) > 5 and evidence.lower() not in user_query.lower():
                enhanced_query = f"{user_query}\nContext: {evidence}"
                print(f"[ChatManager] Enhanced query with evidence: {enhanced_query}")
            else:
                enhanced_query = user_query
        else:
            enhanced_query = user_query
        
        # Initialize the message object for agent communication
        message = {
            'db_id': self.db_name,
            'query': enhanced_query,
            'evidence': evidence,
            'desc_str': schema_info,
            'fk_str': '',  # We'll populate this later if needed
            'extracted_schema': {},
            'send_to': "SelectorAgent",
            'try_times': 0
        }
        
        # First, use the Selector agent to identify relevant schema elements
        try:
            print(f"[SelectorAgent] Analyzing schema for query: {enhanced_query}")
            selected_schema = self.selector_agent.select_schema(enhanced_query, schema_info)
            if not selected_schema:
                logger.warning("Selector agent returned empty schema - using full schema")
                # If no schema was selected, use a simplified approach
                selected_schema = self._create_default_schema_selection(schema_info)
            
            message['extracted_schema'] = selected_schema
            print(f"[SelectorAgent] Selected tables: {', '.join(selected_schema.keys())}")
        except Exception as e:
            logger.error(f"Error in schema selection: {e}")
            traceback.print_exc()
            # Create a default schema selection
            message['extracted_schema'] = self._create_default_schema_selection(schema_info)
            print(f"[SelectorAgent] Error in selection, using default tables: {', '.join(message['extracted_schema'].keys())}")
        
        # Now use the Decomposer agent to understand the query and create a plan
        try:
            print(f"[DecomposerAgent] Processing: {enhanced_query}")
            understanding_and_plan = self.decomposer_agent.understand_and_plan(enhanced_query, schema_info)
            message['understanding'] = understanding_and_plan.get('understanding', '')
            message['plan'] = understanding_and_plan.get('plan', '')
            print(f"[DecomposerAgent] Understanding: {message['understanding'][:100]}...")
            print(f"[DecomposerAgent] Plan: {message['plan'][:100]}...")
        except Exception as e:
            logger.error(f"Error in understanding and planning: {e}")
            traceback.print_exc()
            message['understanding'] = f"The query asks about {enhanced_query}"
            message['plan'] = f"Search for information related to the query in the database"
        
        # Generate the initial SQL using the Refiner agent (which has SQL generation capabilities)
        try:
            sql_result = self.refiner_agent.generate_sql(
                enhanced_query, 
                schema_info, 
                message.get('understanding', ''),
                message.get('plan', '')
            )
            message['final_sql'] = sql_result
            print(f"[RefinerAgent] Generated SQL: {message['final_sql']}")
        except Exception as e:
            logger.error(f"Error in SQL generation: {e}")
            traceback.print_exc()
            # Create a simple SELECT query for fallback
            message['final_sql'] = self._generate_fallback_sql(schema_info)
            print(f"[RefinerAgent] Fallback SQL: {message['final_sql']}")
        
        # Get sample data from tables to help with validation
        sample_data = self._get_table_samples(message['extracted_schema'])
        
        # Execute the SQL and handle any errors
        for attempt in range(MAX_ROUNDS):
            print(f"[ChatManager] SQL execution attempt {attempt+1}/{MAX_ROUNDS}")
            
            # Execute the SQL
            result = self.execute_sql_query(message['final_sql'])
            
            if result.get('success', False):
                # Check if results are empty but shouldn't be
                expected_empty = self._query_expects_empty_result(enhanced_query)
                if not expected_empty and len(result.get('rows', [])) == 0:
                    print(f"[ChatManager] SQL executed successfully but returned no results. Attempting refinement.")
                    # Pass the result to the refiner agent for fixing
                    try:
                        refined_sql = self.refiner_agent.refine_sql(
                            query=enhanced_query,
                            schema=schema_info,
                            understanding=message.get('understanding', ''),
                            original_sql=message['final_sql'],
                            error_message=result  # Pass the successful but empty result
                        )
                        if refined_sql != message['final_sql']:
                            print(f"[ChatManager] Refined SQL to address empty results: {refined_sql}")
                            message['final_sql'] = refined_sql
                            continue  # Retry with the refined SQL
                        else:
                            # If refinement didn't change the SQL, we'll accept the empty result
                            print("[ChatManager] Refinement didn't change the SQL. Accepting empty result.")
                    except Exception as e:
                        logger.error(f"Error refining SQL for empty results: {e}")
                
                # Successful execution with results (or accepted empty results)
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Format results for better readability if needed
                formatted_results = self._format_results(result, enhanced_query)
                
                return {
                    'success': True,
                    'query': user_query,
                    'sql': message['final_sql'],
                    'results': result,
                    'formatted_results': formatted_results,
                    'understanding': message.get('understanding', ''),
                    'plan': message.get('plan', ''),
                    'execution_time': execution_time,
                    'attempts': attempt + 1
                }
            else:
                # Failed execution, try to refine the SQL
                error_message = result.get('error', 'Unknown error')
                print(f"[ChatManager] SQL execution error: {error_message}")
                
                # Enhance error information with sample data if available
                if sample_data:
                    if 'error_info' not in result:
                        result['error_info'] = {}
                    result['error_info']['sample_data'] = sample_data
                
                # Use the Refiner agent to fix the SQL
                try:
                    # Store the original SQL before refinement
                    original_sql_before_refinement = message['final_sql']
                    
                    # Get refined SQL
                    refined_sql = self.refiner_agent.refine_sql(
                        query=enhanced_query, 
                        schema=schema_info, 
                        understanding=message.get('understanding', ''), 
                        original_sql=message['final_sql'],
                        error_message=result  # Pass the entire result object to include error_info
                    )
                    
                    # Check if we got a dictionary or string response
                    if isinstance(refined_sql, dict):
                        message['final_sql'] = refined_sql.get('sql', message['final_sql'])
                    else:
                        message['final_sql'] = refined_sql
                    
                    # If the refiner couldn't fix it, stop trying
                    if message['final_sql'] == original_sql_before_refinement:
                        print("[ChatManager] Refiner couldn't improve the SQL further, stopping refinement.")
                        break
                    
                    # We need to ensure message['final_sql'] is a string for execute_sql_query
                    if isinstance(message['final_sql'], dict):
                        message['final_sql'] = message['final_sql'].get('sql', '')
                except Exception as e:
                    logger.error(f"Error in SQL refinement: {e}")
                    traceback.print_exc()
                    # If refinement fails, try one more time with a simpler approach
                    if attempt == MAX_ROUNDS - 1:
                        simplified_sql = self._generate_simplified_sql(enhanced_query, schema_info, error_message)
                        if simplified_sql != message['final_sql']:
                            message['final_sql'] = simplified_sql
                        else:
                            break
        
        # If we get here, all attempts failed
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            'success': False,
            'query': user_query,
            'sql': message['final_sql'],
            'error': "Failed to execute SQL after multiple refinement attempts",
            'understanding': message.get('understanding', ''),
            'plan': message.get('plan', ''),
            'execution_time': execution_time,
            'attempts': MAX_ROUNDS
        }
    
    def _query_expects_empty_result(self, query):
        """
        Determine if a query is expected to possibly return no results
        """
        # Queries asking about "any", "exists", etc. might legitimately return no results
        negative_indicators = [
            "any", "are there", "exists", "does", "can", "could", "has", "have", "had",
            "is there", "were there", "not"
        ]
        
        query_lower = query.lower()
        for indicator in negative_indicators:
            if indicator in query_lower:
                return True
                
        return False
    
    def _format_results(self, result, query):
        """
        Format SQL results for better readability
        """
        if not result.get('success', False) or not result.get('rows'):
            return None
            
        rows = result.get('rows', [])
        columns = result.get('columns', [])
        
        # For count queries, extract and return the number
        if len(columns) == 1 and columns[0].lower() == 'count' and len(rows) == 1:
            return {'count': rows[0][0]}
            
        # For queries asking "how many", extract the count
        if 'how many' in query.lower() and len(rows) == 1 and len(columns) == 1:
            return {'count': rows[0][0]}
            
        # Format regular result sets
        formatted = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                row_dict[col] = row[i]
            formatted.append(row_dict)
            
        return formatted
    
    def _get_table_samples(self, selected_tables):
        """
        Get sample data from selected tables to help with SQL generation and refinement
        """
        if not self.db_conn:
            return None
            
        sample_data = {}
        
        for table_name in selected_tables.keys():
            try:
                cursor = self.db_conn.cursor()
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                if rows:
                    sample_data[table_name] = {
                        'columns': columns,
                        'rows': rows
                    }
            except Exception as e:
                logger.error(f"Error getting sample data for {table_name}: {e}")
                
        return sample_data

    def _create_default_schema_selection(self, schema_info):
        """Create a default schema selection when the selector agent fails"""
        # Extract tables from schema
        import re
        tables = []
        table_pattern = r"Table:\s+([a-zA-Z0-9_]+)"
        matches = re.finditer(table_pattern, schema_info)
        
        for match in matches:
            table_name = match.group(1)
            if table_name and table_name not in tables:
                tables.append(table_name)
        
        # Create a simple schema dict with all tables
        return {table: 'keep_all' for table in tables}
        
    def _generate_fallback_sql(self, schema_info):
        """Generate a simple fallback SQL query when generation fails"""
        # Extract the first table from schema
        import re
        table_match = re.search(r"Table:\s+([a-zA-Z0-9_]+)", schema_info)
        
        if table_match:
            table_name = table_match.group(1)
            return f"SELECT * FROM {table_name} LIMIT 5;"
        else:
            # Ultimate fallback
            return "SELECT version();"
        
    def _generate_simplified_sql(self, query, schema_info, error_message):
        """Generate a simplified SQL as a last resort"""
        # Create a prompt for simplified SQL generation
        prompt = f"""
        Generate a very simple SQL query for this question:
        "{query}"
        
        Previous error: "{error_message}"
        
        Schema information:
        {schema_info[:500]}...
        
        Return only a simple SQL query that is likely to execute successfully, even if it doesn't fully answer the question.
        Focus on using only tables and columns that definitely exist, and use simple SELECT statements.
        """
        
        try:
            # Generate a simplified SQL query
            response = self.generate_llm_response(prompt)
            
            # Extract SQL from response
            import re
            sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
            if sql_match:
                sql = sql_match.group(1).strip()
            else:
                # Try to extract without code blocks
                sql_match = re.search(r'SELECT\s+.*?;', response, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    sql = sql_match.group(0).strip()
                else:
                    # Just use the response as is
                    sql = response.strip()
            
            # Ensure SQL ends with semicolon
            if not sql.strip().endswith(';'):
                sql = sql.strip() + ';'
                
            return sql
        except Exception as e:
            logger.error(f"Error generating simplified SQL: {e}")
            # Ultimate fallback - a SQL query that should always work
            return "SELECT 'Could not generate valid SQL' AS message;"
        
    def connect_to_database(self, db_name):
        """
        Connect to a PostgreSQL database
        
        Args:
            db_name: Name of the database to connect to
            
        Returns:
            True if connection successful, False otherwise
        """
        self.db_name = db_name
        
        # Check if this is a BIRD benchmark database
        is_bird_db = self._is_bird_benchmark_db(db_name)
        
        try:
            # Close existing connection if any
            if self.db_conn:
                self.db_conn.close()
                
            # Connect to PostgreSQL using the specified db_name
            self.db_conn = psycopg2.connect(
                dbname=db_name,  # Use the provided db_name instead of DB_CONFIG["dbname"]
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                host=DB_CONFIG["host"],
                port=DB_CONFIG["port"]
            )
            
            if self.verbose:
                print(f"Connected to database: {db_name}")
                
            return True
        
        except Exception as e:
            logger.warning(f"Error connecting to database {db_name}: {str(e)}")
            print(f"Error connecting to database {db_name}: {str(e)}")
            self.db_conn = None
            
            # For BIRD benchmark databases, we can continue with hardcoded schemas
            if is_bird_db:
                logger.info(f"Database {db_name} is a BIRD benchmark database. Will use hardcoded schema.")
                print(f"Database {db_name} is a BIRD benchmark database. Will use hardcoded schema.")
                # We return False but the calling code will still try to use hardcoded schema
            
            return False
    
    def get_db_connection(self):
        """
        Get the current database connection
        
        Returns:
            Database connection object or None if not connected
        """
        if not self.db_conn:
            if self.db_name:
                self.connect_to_database(self.db_name)
        
        return self.db_conn
        
    def get_schema(self):
        """
        Get the database schema as a string
        
        Returns:
            String representation of the database schema
        """
        # Check if we have a cached schema
        if self.db_name in self.schema_cache:
            return self.schema_cache[self.db_name]
            
        # Check if this is a BIRD benchmark database
        is_bird_db = self._is_bird_benchmark_db(self.db_name)
        
        # If no database connection, try to connect
        if not self.db_conn:
            if not self.connect_to_database(self.db_name):
                # If connection fails, check if we have a hardcoded schema
                hardcoded_schema = self.selector_agent.get_detailed_schema(self.db_name)
                if hardcoded_schema:
                    logger.info(f"Using hardcoded schema after connection failure: {self.db_name}")
                    self.schema_cache[self.db_name] = hardcoded_schema
                    return hardcoded_schema
                    
                # Try fallback schema for BIRD benchmark databases
                if is_bird_db:
                    bird_schema = self._get_bird_schema_fallback(self.db_name)
                    if bird_schema:
                        logger.info(f"Using fallback schema after connection failure: {self.db_name}")
                        self.schema_cache[self.db_name] = bird_schema
                        return bird_schema
                        
                logger.error(f"No schema available for database: {self.db_name}")
                return ""
                
        try:
            # Query PostgreSQL for schema information
            cursor = self.db_conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                logger.warning(f"No tables found in database {self.db_name}, checking hardcoded schema")
                # If no tables found, check if we have a hardcoded schema
                hardcoded_schema = self.selector_agent.get_detailed_schema(self.db_name)
                if hardcoded_schema:
                    self.schema_cache[self.db_name] = hardcoded_schema
                    return hardcoded_schema
                
                # If this is a BIRD benchmark database, try to use known schema
                if is_bird_db:
                    bird_schema = self._get_bird_schema_fallback(self.db_name)
                    if bird_schema:
                        self.schema_cache[self.db_name] = bird_schema
                        return bird_schema
                
                logger.error(f"No tables found in database {self.db_name} and no hardcoded schema available")
                return ""
            
            # Build schema string
            schema_str = ""
            
            for table in tables:
                schema_str += f"\nTable: {table}\n"
                
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
            
            # Cache and return the schema
            self.schema_cache[self.db_name] = schema_str
            return schema_str
            
        except Exception as e:
            logger.error(f"Error getting schema for database {self.db_name}: {str(e)}")
            
            # If this is a BIRD benchmark database, try to use hardcoded schema
            if is_bird_db:
                hardcoded_schema = self.selector_agent.get_detailed_schema(self.db_name)
                if hardcoded_schema:
                    logger.info(f"Using hardcoded schema after error: {self.db_name}")
                    self.schema_cache[self.db_name] = hardcoded_schema
                    return hardcoded_schema
                    
                # Try fallback schema
                bird_schema = self._get_bird_schema_fallback(self.db_name)
                if bird_schema:
                    logger.info(f"Using fallback schema after error: {self.db_name}")
                    self.schema_cache[self.db_name] = bird_schema
                    return bird_schema
            
            return ""
            
    def _is_bird_benchmark_db(self, db_name: str) -> bool:
        """
        Check if the given database name is part of the BIRD benchmark
        
        Args:
            db_name: Database name to check
            
        Returns:
            True if this is a BIRD benchmark database
        """
        bird_dbs = [
            'student_club', 'formula_1', 'california_schools', 'european_football_2',
            'debit_card_specializing', 'thrombosis_prediction', 'world_1', 'chinook_1',
            'bike_1', 'insurance_fnol', 'medicine_cardiovascular_disease',
            'academic', 'activity', 'aircraft', 'allergy', 'apartment_rentals',
            'architecture', 'assets_maintenance', 'baseball', 'basketball',
            'battle_death', 'behavior_monitoring', 'bike_sharing', 'body_builder',
            'book_club', 'browser_web', 'candidate_poll', 'car_1', 'chinook',
            'cinema', 'city_record', 'climbing', 'club', 'coffee_shop', 'college',
            'company_1', 'company_employee', 'company_office', 'concert_singer',
            'county_public_safety', 'course_teach', 'cre_Doc_Control_Systems',
            'cre_Doc_Template_Mgt', 'cre_Theme_park', 'credit_card', 'customer_deliveries',
            'customer_complaints', 'customers_and_addresses', 'customers_and_invoices',
            'customers_and_products', 'customers_campaigns_ecommerce', 'customers_card_transactions',
            'debate', 'decoration_competition', 'department_management', 'department_store',
            'device', 'document_management', 'dog_kennels', 'dorm_1', 'driving_school',
            'e_learning', 'election', 'election_representative', 'entertainment_awards',
            'entrepreneur', 'epinions', 'estate_agent', 'farm', 'film_rank', 'flight_1',
            'flight_company', 'flight_schedule', 'formula_1', 'game_injury', 'gas_company',
            'geo', 'gymnast', 'hospital_1', 'hr_1', 'icfp', 'imdb', 'inn', 'insurance_and_accident',
            'insurance_fnol', 'insurance_policies', 'journal_committee', 'loan_1', 'local_govt_and_lot',
            'local_govt_in_alabama', 'local_govt_mdm', 'machine_repair', 'manufactory_1',
            'manufacturer', 'match_season', 'medicine_enzyme_interaction', 'mountain_photos',
            'movie_1', 'music_1', 'music_2', 'music_4', 'musical', 'network_1', 'network_2',
            'news_report', 'orchestra', 'party_people', 'performance_attendance', 'perpetrator',
            'personnel_database', 'phone_1', 'phone_market', 'pilot_record', 'plant_1',
            'plant_catalog', 'poker_player', 'product_catalog', 'products_for_hire',
            'products_gen_characteristics', 'program_share', 'protein_institute', 'race_track',
            'railway', 'real_estate_properties', 'restaurant', 'restaurant_1', 'riding_club',
            'roller_coaster', 'school_finance', 'school_player', 'scientist_1', 'ship',
            'ship_1', 'ship_mission', 'shop_membership', 'singer', 'small_bank_1',
            'soccer_1', 'soccer_2', 'solvency_ii', 'sports_competition', 'station_weather',
            'store_1', 'store_product', 'student_1', 'student_assessment', 'student_club',
            'student_transcripts_tracking', 'swimming', 'swimming_record', 'theme_gallery',
            'tracking_grants_for_research', 'tracking_software_problems', 'train_station',
            'tvshow', 'twitter_1', 'university_basketball', 'voter_1', 'wedding', 'wine_1',
            'workshop_paper', 'world_1', 'wrestler'
        ]
        
        if not db_name:
            return False
            
        db_name_lower = db_name.lower()
        
        # Direct match
        if db_name_lower in bird_dbs:
            return True
            
        # Partial match (if the db name contains a known BIRD db name)
        for bird_db in bird_dbs:
            if bird_db in db_name_lower:
                return True
                
        return False
        
    def _get_bird_schema_fallback(self, db_name: str) -> str:
        """
        Get a fallback schema string for BIRD benchmark databases
        
        Args:
            db_name: Database name
            
        Returns:
            Schema string or empty string if no fallback available
        """
        # Try to match with a known database name
        db_name_lower = db_name.lower()
        
        # Schema templates for common BIRD databases
        if 'student_club' in db_name_lower:
            return """
Table: member
Columns:
  - member_id (int, NOT NULL, PRIMARY KEY)
  - first_name (varchar, NULL)
  - last_name (varchar, NULL)
  - position (varchar, NULL)
  - email (varchar, NULL)
  - phone (varchar, NULL)

Table: event
Columns:
  - event_id (int, NOT NULL, PRIMARY KEY)
  - event_name (varchar, NULL)
  - type (varchar, NULL)
  - date (date, NULL)
  - location (varchar, NULL)

Table: attendance
Columns:
  - attendance_id (int, NOT NULL, PRIMARY KEY)
  - link_to_member (int, NULL)
  - link_to_event (int, NULL)
Foreign Keys:
  - link_to_member -> member.member_id
  - link_to_event -> event.event_id

Table: income
Columns:
  - income_id (int, NOT NULL, PRIMARY KEY)
  - amount (decimal, NULL)
  - date_received (date, NULL)
  - link_to_member (int, NULL)
Foreign Keys:
  - link_to_member -> member.member_id

Table: budget
Columns:
  - budget_id (int, NOT NULL, PRIMARY KEY)
  - category (varchar, NULL)
  - amount (decimal, NULL)
  - link_to_event (int, NULL)
Foreign Keys:
  - link_to_event -> event.event_id
"""

        elif 'formula_1' in db_name_lower:
            return """
Table: drivers
Columns:
  - driverId (int, NOT NULL, PRIMARY KEY)
  - driverRef (varchar, NULL)
  - number (int, NULL)
  - code (varchar, NULL)
  - forename (varchar, NULL)
  - surname (varchar, NULL)
  - dob (date, NULL)
  - nationality (varchar, NULL)
  - url (varchar, NULL)

Table: races
Columns:
  - raceId (int, NOT NULL, PRIMARY KEY)
  - year (int, NULL)
  - round (int, NULL)
  - circuitId (int, NULL)
  - name (varchar, NULL)
  - date (date, NULL)
  - time (time, NULL)
  - url (varchar, NULL)
Foreign Keys:
  - circuitId -> circuits.circuitId

Table: constructors
Columns:
  - constructorId (int, NOT NULL, PRIMARY KEY)
  - constructorRef (varchar, NULL)
  - name (varchar, NULL)
  - nationality (varchar, NULL)
  - url (varchar, NULL)

Table: results
Columns:
  - resultId (int, NOT NULL, PRIMARY KEY)
  - raceId (int, NULL)
  - driverId (int, NULL)
  - constructorId (int, NULL)
  - number (int, NULL)
  - grid (int, NULL)
  - position (int, NULL)
  - positionText (varchar, NULL)
  - positionOrder (int, NULL)
  - points (float, NULL)
  - laps (int, NULL)
  - time (varchar, NULL)
  - milliseconds (int, NULL)
  - fastestLap (int, NULL)
  - rank (int, NULL)
  - fastestLapTime (varchar, NULL)
  - fastestLapSpeed (varchar, NULL)
  - statusId (int, NULL)
Foreign Keys:
  - raceId -> races.raceId
  - driverId -> drivers.driverId
  - constructorId -> constructors.constructorId
  - statusId -> status.statusId

Table: circuits
Columns:
  - circuitId (int, NOT NULL, PRIMARY KEY)
  - circuitRef (varchar, NULL)
  - name (varchar, NULL)
  - location (varchar, NULL)
  - country (varchar, NULL)
  - lat (float, NULL)
  - lng (float, NULL)
  - alt (int, NULL)
  - url (varchar, NULL)

Table: status
Columns:
  - statusId (int, NOT NULL, PRIMARY KEY)
  - status (varchar, NULL)
"""

        elif 'european_football_2' in db_name_lower:
            return """
Table: player
Columns:
  - player_id (int, NOT NULL, PRIMARY KEY)
  - player_name (varchar, NULL)
  - team_id (int, NULL)
  - birthday (date, NULL)
  - height (int, NULL)
  - weight (int, NULL)
Foreign Keys:
  - team_id -> team.team_id

Table: team
Columns:
  - team_id (int, NOT NULL, PRIMARY KEY)
  - team_name (varchar, NULL)
  - league (varchar, NULL)

Table: match
Columns:
  - match_id (int, NOT NULL, PRIMARY KEY)
  - season (varchar, NULL)
  - date (date, NULL)
  - home_team_id (int, NULL)
  - away_team_id (int, NULL)
  - home_goals (int, NULL)
  - away_goals (int, NULL)
Foreign Keys:
  - home_team_id -> team.team_id
  - away_team_id -> team.team_id
"""

        elif 'debit_card_specializing' in db_name_lower:
            return """
Table: transactions_1k
Columns:
  - Transaction_ID (int, NOT NULL, PRIMARY KEY)
  - Account_ID (int, NULL)
  - Card_ID (int, NULL)
  - Date (date, NULL)
  - GasStationID (int, NULL)
  - Price (decimal, NULL)
  - Volume (decimal, NULL)
Foreign Keys:
  - GasStationID -> gasstations.GasStationID

Table: gasstations
Columns:
  - GasStationID (int, NOT NULL, PRIMARY KEY)
  - GasStationName (varchar, NULL)
  - Address (varchar, NULL)
  - City (varchar, NULL)
  - Country (varchar, NULL)
  - Latitude (decimal, NULL)
  - Longitude (decimal, NULL)
"""

        elif 'thrombosis_prediction' in db_name_lower:
            return """
Table: patient
Columns:
  - ID (int, NOT NULL, PRIMARY KEY)
  - SEX (char, NULL)
  - Birthday (date, NULL)
  - Height (decimal, NULL)
  - Weight (decimal, NULL)

Table: laboratory
Columns:
  - Lab_ID (int, NOT NULL, PRIMARY KEY)
  - ID (int, NULL)
  - Date (date, NULL)
  - CPK (int, NULL)
  - "T-CHO" (int, NULL)
  - HDL (int, NULL)
  - LDL (int, NULL)
  - TG (int, NULL)
  - UA (decimal, NULL)
  - GLU (int, NULL)
  - HbA1c (decimal, NULL)
  - BUN (decimal, NULL)
  - CRE (decimal, NULL)
  - Urine_PRO (varchar, NULL)
  - Urine_GLU (varchar, NULL)
Foreign Keys:
  - ID -> patient.ID
"""
            
        # Add more schemas as needed for other BIRD databases
        
        return ""
    
    def execute_sql_query(self, query, attempts=3):
        """
        Execute a SQL query against the connected database.
        
        Args:
            query: The SQL query to execute
            attempts: Number of attempts to make
            
        Returns:
            Dictionary with execution results
        """
        # Make sure we have a database connection
        if not self.db_conn:
            return {
                'success': False, 
                'error': "Not connected to database"
            }
        
        # Handle dictionary input (in case we get a dict with 'sql' field)
        if isinstance(query, dict) and 'sql' in query:
            query = query['sql']
        
        if not query or (isinstance(query, str) and not query.strip()) or (isinstance(query, dict)):
            return {
                'success': False, 
                'error': "Empty query string or invalid query format"
            }
        
        try:
            # Clone the connection to execute the query in a transaction
            # This allows us to roll back if an error occurs
            cursor = self.db_conn.cursor()
            
            # Start a transaction
            cursor.execute("BEGIN;")
            
            # Execute the query
            cursor.execute(query)
            
            # Check if the query returned results
            try:
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                # Commit the transaction
                cursor.execute("COMMIT;")
                
                return {
                    'success': True,
                    'rows': rows,
                    'columns': columns
                }
            except Exception as e:
                # Query executed successfully but didn't return results
                # This is common for INSERT, UPDATE, DELETE queries
                
                # Commit the transaction for non-SELECT queries
                cursor.execute("COMMIT;")
                
                # For non-SELECT queries, we'll return a success message
                return {
                    'success': True,
                    'rows': [],
                    'columns': [],
                    'affected_rows': cursor.rowcount if hasattr(cursor, 'rowcount') else 0
                }
        
        except Exception as e:
            # Roll back the transaction
            try:
                cursor.execute("ROLLBACK;")
                print(f"Transaction rolled back due to error: {str(e)}")
            except:
                pass
            
            # Capture common database errors and provide helpful messages
            error_message = str(e)
            error_info = {'error_details': {}}
            
            # Check for relation does not exist errors
            if "relation" in error_message and "does not exist" in error_message:
                # Try to extract the table name
                import re
                match = re.search(r'relation "([^"]+)" does not exist', error_message)
                if match:
                    table_name = match.group(1)
                    error_info['error_details']['missing_table'] = table_name
                    
                    # Get a list of available tables to suggest alternatives
                    try:
                        cursor = self.db_conn.cursor()
                        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
                        available_tables = [row[0] for row in cursor.fetchall()]
                        error_info['error_details']['available_tables'] = available_tables
                        
                        # Find similar table names
                        import difflib
                        similar_tables = difflib.get_close_matches(table_name, available_tables, n=3, cutoff=0.5)
                        if similar_tables:
                            error_info['error_details']['similar_tables'] = similar_tables
                    except:
                        pass
            
            # Column does not exist errors
            elif "column" in error_message and "does not exist" in error_message:
                import re
                match = re.search(r'column "([^"]+)" does not exist', error_message)
                if match:
                    column_name = match.group(1)
                    error_info['error_details']['missing_column'] = column_name
            
            # Syntax errors
            elif "syntax error" in error_message:
                error_info['error_details']['syntax_error'] = True
            
            return {
                'success': False,
                'error': error_message,
                'error_info': error_info
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
            # Add automatic rate limiting with a small delay
            # This helps avoid hitting API rate limits
            import time
            time.sleep(0.5)  # Half second delay between API calls
            
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