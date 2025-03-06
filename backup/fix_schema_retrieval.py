def modify_mac_sql_agent():
    """
    Fix the schema retrieval in the MAC-SQL agent to handle case sensitivity
    """
    with open('mac_sql_agent.py', 'r') as f:
        content = f.read()
    
    # Fix the sample data query to use double quotes for identifiers
    # Look for: cursor.execute(f"SELECT * FROM {table} LIMIT 3")
    # Replace with: cursor.execute(f'SELECT * FROM "{table}" LIMIT 3')
    modified = content.replace(
        'cursor.execute(f"SELECT * FROM {table} LIMIT 3")',
        'cursor.execute(f\'SELECT * FROM "{table}" LIMIT 3\')'
    )
    
    # Also fix any specific column reference that's causing errors
    modified = modified.replace(
        'SELECT asciiName FROM cards LIMIT 3',
        'SELECT "asciiName" FROM "cards" LIMIT 3'
    )
    
    with open('mac_sql_agent.py', 'w') as f:
        f.write(modified)
    
    print("Fixed schema retrieval in MAC-SQL agent") 