def adapt_sql_dialect(sql_query, source_dialect="mysql", target_dialect="postgresql"):
    """
    Convert SQL query from one dialect to another
    
    Args:
        sql_query: Original SQL query
        source_dialect: Source dialect (mysql, sqlite)
        target_dialect: Target dialect (postgresql)
    
    Returns:
        Converted SQL query
    """
    import re
    
    if source_dialect.lower() in ["mysql", "sqlite"] and target_dialect.lower() == "postgresql":
        # 1. Replace backticks with double quotes for identifiers
        # Match backticked identifiers and replace with double quotes
        sql_query = re.sub(r'`([^`]+)`', r'"\1"', sql_query)
        
        # 2. Fix LIMIT/OFFSET syntax differences
        # MySQL: LIMIT x OFFSET y | PostgreSQL: LIMIT x OFFSET y (same, no change needed)
        
        # 3. Handle IFNULL vs COALESCE
        sql_query = re.sub(r'IFNULL\s*\(', 'COALESCE(', sql_query, flags=re.IGNORECASE)
        
        # 4. Handle string concatenation
        # MySQL: CONCAT(a, b) | PostgreSQL: a || b
        # This is more complex and might require custom parsing
        
        # 5. Handle date functions
        # Various differences in date functions between dialects
        
        # 6. Handle boolean literals
        sql_query = re.sub(r'\bTRUE\b', 'true', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bFALSE\b', 'false', sql_query, flags=re.IGNORECASE)
        
        return sql_query
    
    # If dialects are the same or unsupported combination, return original
    return sql_query

def results_are_equivalent(results1, results2):
    """
    Check if two SQL query results are equivalent, allowing for
    minor differences in formatting, order, etc.
    
    Args:
        results1: First result set (list of tuples)
        results2: Second result set (list of tuples)
    
    Returns:
        True if results are equivalent, False otherwise
    """
    import pandas as pd
    
    # Convert to pandas DataFrames for easier comparison
    if not isinstance(results1, pd.DataFrame):
        df1 = pd.DataFrame(results1)
    else:
        df1 = results1
        
    if not isinstance(results2, pd.DataFrame):
        df2 = pd.DataFrame(results2)
    else:
        df2 = results2
    
    # If DataFrames have different shapes, they're not equivalent
    if df1.shape != df2.shape:
        return False
    
    # Check if DataFrames have the same values (ignoring column names)
    # Sort both DataFrames if possible to handle order differences
    try:
        df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
        df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)
        return df1_sorted.equals(df2_sorted)
    except:
        # If sorting fails (e.g., due to mixed types), compare as is
        return df1.equals(df2) 