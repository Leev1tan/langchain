#!/usr/bin/env python
"""
Setup script to create and populate test databases in PostgreSQL for MAC-SQL benchmarking.
This script creates test databases based on the schemas in the BIRD benchmark.
"""

import os
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Get PostgreSQL connection parameters from environment variables
DB_CONFIG = {
    "dbname": os.environ.get("POSTGRES_DB", "postgres"),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5432")
}

# Schema definitions for test databases
SCHEMAS = {
    "student_club": """
    -- Create tables for student_club database
    CREATE TABLE member (
        member_id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        join_date DATE NOT NULL
    );

    CREATE TABLE event (
        event_id SERIAL PRIMARY KEY,
        event_name VARCHAR(100) NOT NULL,
        type VARCHAR(50) NOT NULL,
        date DATE NOT NULL,
        location VARCHAR(100) NOT NULL
    );

    CREATE TABLE attendance (
        attendance_id SERIAL PRIMARY KEY,
        member_id INTEGER REFERENCES member(member_id),
        event_id INTEGER REFERENCES event(event_id),
        status VARCHAR(20) CHECK (status IN ('registered', 'attended', 'cancelled'))
    );

    -- Insert sample data
    INSERT INTO member (name, email, join_date) VALUES
        ('John Smith', 'john.smith@email.com', '2022-01-15'),
        ('Emily Johnson', 'emily.j@email.com', '2022-02-20'),
        ('Michael Brown', 'michael.b@email.com', '2022-03-10'),
        ('Sarah Davis', 'sarah.d@email.com', '2022-04-05'),
        ('David Wilson', 'david.w@email.com', '2022-05-12');

    INSERT INTO event (event_name, type, date, location) VALUES
        ('Annual Meeting', 'General', '2023-01-20', 'Main Hall'),
        ('Summer Picnic', 'Social', '2023-06-15', 'City Park'),
        ('Workshop', 'Educational', '2023-03-25', 'Room 101'),
        ('Charity Run', 'Community', '2023-05-10', 'Downtown'),
        ('Holiday Party', 'Social', '2023-12-15', 'Community Center');

    INSERT INTO attendance (member_id, event_id, status) VALUES
        (1, 1, 'attended'),
        (1, 2, 'registered'),
        (2, 1, 'attended'),
        (2, 3, 'attended'),
        (3, 2, 'cancelled'),
        (3, 4, 'registered'),
        (4, 1, 'attended'),
        (4, 5, 'registered'),
        (5, 3, 'attended'),
        (5, 5, 'attended');
    """,

    "formula_1": """
    -- Create tables for formula_1 database
    CREATE TABLE driver (
        driver_id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        nationality VARCHAR(50) NOT NULL,
        birth_date DATE NOT NULL
    );

    CREATE TABLE team (
        team_id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        country VARCHAR(50) NOT NULL,
        founded_year INTEGER NOT NULL
    );

    CREATE TABLE race (
        race_id SERIAL PRIMARY KEY,
        race_name VARCHAR(100) NOT NULL,
        circuit VARCHAR(100) NOT NULL,
        country VARCHAR(50) NOT NULL,
        date DATE NOT NULL
    );

    CREATE TABLE result (
        result_id SERIAL PRIMARY KEY,
        race_id INTEGER REFERENCES race(race_id),
        driver_id INTEGER REFERENCES driver(driver_id),
        team_id INTEGER REFERENCES team(team_id),
        position INTEGER,
        points INTEGER,
        DNF BOOLEAN DEFAULT FALSE
    );

    -- Insert sample data
    INSERT INTO driver (name, nationality, birth_date) VALUES
        ('Lewis Hamilton', 'British', '1985-01-07'),
        ('Max Verstappen', 'Dutch', '1997-09-30'),
        ('Charles Leclerc', 'Monegasque', '1997-10-16'),
        ('Sergio Perez', 'Mexican', '1990-01-26'),
        ('Fernando Alonso', 'Spanish', '1981-07-29');

    INSERT INTO team (name, country, founded_year) VALUES
        ('Mercedes', 'Germany', 1954),
        ('Red Bull Racing', 'Austria', 2005),
        ('Ferrari', 'Italy', 1950),
        ('McLaren', 'United Kingdom', 1966),
        ('Aston Martin', 'United Kingdom', 2021);

    INSERT INTO race (race_name, circuit, country, date) VALUES
        ('Monaco Grand Prix', 'Circuit de Monaco', 'Monaco', '2023-05-28'),
        ('British Grand Prix', 'Silverstone Circuit', 'United Kingdom', '2023-07-09'),
        ('Italian Grand Prix', 'Monza Circuit', 'Italy', '2023-09-03'),
        ('Japanese Grand Prix', 'Suzuka Circuit', 'Japan', '2023-09-24'),
        ('Abu Dhabi Grand Prix', 'Yas Marina Circuit', 'United Arab Emirates', '2023-11-26');

    INSERT INTO result (race_id, driver_id, team_id, position, points, DNF) VALUES
        (1, 1, 1, 2, 18, FALSE),
        (1, 2, 2, 1, 25, FALSE),
        (1, 3, 3, 4, 12, FALSE),
        (1, 4, 2, 3, 15, FALSE),
        (1, 5, 5, 5, 10, FALSE),
        (2, 1, 1, 1, 25, FALSE),
        (2, 2, 2, 2, 18, FALSE),
        (2, 3, 3, 3, 15, FALSE),
        (2, 4, 2, NULL, 0, TRUE),
        (2, 5, 5, 4, 12, FALSE),
        (3, 1, 1, 3, 15, FALSE),
        (3, 2, 2, 1, 25, FALSE),
        (3, 3, 3, 2, 18, FALSE),
        (3, 4, 2, 5, 10, FALSE),
        (3, 5, 5, 4, 12, FALSE);
    """
}

def create_test_database(db_name, schema_sql):
    """
    Create and populate a test database
    
    Args:
        db_name: Name of database to create
        schema_sql: SQL schema to apply
    """
    # Connect to default database
    conn = psycopg2.connect(
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    try:
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if exists:
            print(f"Database {db_name} already exists, dropping it...")
            
            # Terminate all connections to the database
            cursor.execute(
                sql.SQL("SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = %s;"),
                [db_name]
            )
            
            # Drop the database
            cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name)))
        
        # Create new database
        print(f"Creating database {db_name}...")
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        
        # Connect to the new database
        conn.close()
        conn = psycopg2.connect(
            dbname=db_name,
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"]
        )
        
        cursor = conn.cursor()
        
        # Execute schema SQL
        print(f"Setting up schema for {db_name}...")
        cursor.execute(schema_sql)
        
        # Commit changes
        conn.commit()
        print(f"Database {db_name} set up successfully!")
        
    except Exception as e:
        print(f"Error setting up database {db_name}: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    print("Setting up test databases for MAC-SQL benchmarking...")
    
    for db_name, schema_sql in SCHEMAS.items():
        create_test_database(db_name, schema_sql)
    
    print("Test database setup complete!") 