"""
Configuration settings for the MAC-SQL framework
"""

# PostgreSQL connection configuration
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "superuser",
    "dbname": None  # Will be set when connecting to a specific database
} 