import psycopg2
import pandas as pd
from sql_dialect_adapter import adapt_sql_dialect, results_are_equivalent

def update_mac_sql_agent_evaluation():
    """
    Update the evaluation method in the MAC-SQL agent to handle SQL dialect differences
    """
    # Import our dialect adapter functions
    from sql_dialect_adapter import adapt_sql_dialect, results_are_equivalent
    
    # Update the execute_sql_query method to handle dialect differences
    def patched_execute_sql_query(self, sql_query, refinement_attempts=0):
        """Execute the generated SQL query and handle errors if they arise"""
        MAX_REFINEMENT_ATTEMPTS = 2
        try:
            # First adapt the SQL query to PostgreSQL dialect if needed
            if '`' in sql_query:  # Simple check for MySQL/SQLite dialect
                sql_query = adapt_sql_dialect(sql_query, "mysql", "postgresql")
            
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            return pd.DataFrame(rows, columns=columns)
        except psycopg2.Error as e:
            if refinement_attempts < MAX_REFINEMENT_ATTEMPTS:
                refined_query = self.refine_sql_query(sql_query, str(e))
                return self.execute_sql_query(refined_query, refinement_attempts + 1)
            else:
                return f"Error executing query: {e}"
    
    # Replace the MACSQLAgent.execute_sql_query method
    MACSQLAgent.execute_sql_query = patched_execute_sql_query
    
    print("Updated MAC-SQL agent evaluation to handle SQL dialect differences") 