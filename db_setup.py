#!/usr/bin/env python
"""
PostgreSQL Database Setup Script

This script creates and initializes the necessary databases and tables for MAC-SQL testing.
"""

import os
import psycopg2
import pandas as pd
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configuration
DB_CONFIG = {
    "user": "postgres",
    "password": "superuser",
    "host": "localhost",
    "port": "5432"
}

def create_database(db_name):
    """Create a PostgreSQL database"""
    # Connect to default database
    conn = psycopg2.connect(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    # Create database
    try:
        cursor = conn.cursor()
        cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"Database '{db_name}' created successfully")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database '{db_name}': {e}")
        return False

def setup_student_club_db():
    """Set up the student_club database with tables and sample data"""
    # Create the database
    if not create_database("student_club"):
        return False
    
    # Connect to the new database
    conn = psycopg2.connect(
        dbname="student_club",
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )
    cursor = conn.cursor()
    
    try:
        # Create tables
        
        # Member table
        cursor.execute("""
        CREATE TABLE member (
            member_id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            email TEXT,
            join_date TEXT,
            graduation_year INTEGER,
            major TEXT,
            role TEXT
        )
        """)
        
        # Event table
        cursor.execute("""
        CREATE TABLE event (
            event_id TEXT PRIMARY KEY,
            event_name TEXT,
            event_date TEXT,
            location TEXT,
            description TEXT,
            budget INTEGER
        )
        """)
        
        # Income table
        cursor.execute("""
        CREATE TABLE income (
            income_id TEXT PRIMARY KEY,
            amount INTEGER,
            date_received TEXT,
            source TEXT,
            notes TEXT,
            link_to_member TEXT
        )
        """)
        
        # Expense table
        cursor.execute("""
        CREATE TABLE expense (
            expense_id TEXT PRIMARY KEY,
            amount INTEGER,
            date_spent TEXT,
            vendor TEXT,
            description TEXT,
            event_id TEXT,
            FOREIGN KEY (event_id) REFERENCES event(event_id)
        )
        """)
        
        # Budget table
        cursor.execute("""
        CREATE TABLE budget (
            budget_id TEXT PRIMARY KEY,
            semester TEXT,
            total_amount INTEGER,
            start_date TEXT,
            end_date TEXT
        )
        """)
        
        # Major table
        cursor.execute("""
        CREATE TABLE major (
            major_id TEXT PRIMARY KEY,
            major_name TEXT,
            department TEXT
        )
        """)
        
        # Zip code table
        cursor.execute("""
        CREATE TABLE zip_code (
            zip TEXT PRIMARY KEY,
            city TEXT,
            state TEXT
        )
        """)
        
        # Insert sample data
        
        # Member data
        members = [
            ('rec123', 'John', 'Doe', 'john.doe@example.com', '2019-09-01', 2023, 'Computer Science', 'President'),
            ('rec124', 'Jane', 'Smith', 'jane.smith@example.com', '2019-09-01', 2022, 'Biology', 'Vice President'),
            ('rec125', 'Michael', 'Johnson', 'michael.j@example.com', '2019-09-15', 2024, 'Mathematics', 'Treasurer'),
            # Add more members to reach a total of 33
        ]
        
        # Insert 30 more members to get to 33 total
        for i in range(30):
            members.append((
                f'rec{126+i}', 
                f'FirstName{i}', 
                f'LastName{i}', 
                f'student{i}@example.com', 
                '2019-10-01', 
                2023 + (i % 4), 
                'Computer Science' if i % 3 == 0 else 'Biology' if i % 3 == 1 else 'Mathematics', 
                'Member'
            ))
        
        for member in members:
            cursor.execute("""
            INSERT INTO member (member_id, first_name, last_name, email, join_date, graduation_year, major, role)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, member)
        
        # Event data
        events = [
            ('event1', 'Welcome Party', '2019-09-05', 'Student Union', 'Welcome party for new members', 500),
            ('event2', 'Tech Workshop', '2019-10-10', 'Engineering Building', 'Workshop on new technologies', 300),
            ('event3', 'End of Year Celebration', '2019-12-15', 'Campus Center', 'Celebration for the end of the year', 800),
        ]
        
        for event in events:
            cursor.execute("""
            INSERT INTO event (event_id, event_name, event_date, location, description, budget)
            VALUES (%s, %s, %s, %s, %s, %s)
            """, event)
        
        # Income data
        incomes = [
            ('rec0s9ZrO15zhzUeE', 50, '2019-10-17', 'Dues', 'Member dues payment', 'rec123'),
            ('rec7f5XMQZexgtQJo', 50, '2019-09-04', 'Dues', 'Member dues payment', 'rec124'),
            ('rec8BUJa8GXUjiglg', 50, '2019-10-08', 'Dues', 'Member dues payment', 'rec125'),
            ('rec8V9BPNIoewWt2z', 50, '2019-10-02', 'Dues', 'Member dues payment', 'rec126'),
            ('recCRWMfFqifuKMc6', 50, '2019-09-18', 'Dues', 'Member dues payment', 'rec127'),
            ('recDaffs123ksdajf', 200, '2019-11-10', 'Fundraising', 'Bake sale fundraiser', None),
            ('recHjkls876lkPOkl', 1000, '2019-09-01', 'Sponsorship', 'Tech company sponsorship', None),
            ('recIopqrs234jklMN', 3000, '2019-08-30', 'School Appropration', 'Funding from university', None),
        ]
        
        # Insert more dues payments to reach the reported total of $1650
        for i in range(30):
            if i < 25:  # Only need 25 more dues payments to reach total of 33 (5 + 25 = 30)
                incomes.append((
                    f'recDues{i}', 
                    50, 
                    f'2019-{9 + (i % 3)}-{1 + (i % 28)}', 
                    'Dues', 
                    'Member dues payment', 
                    f'rec{128+i}'
                ))
        
        for income in incomes:
            cursor.execute("""
            INSERT INTO income (income_id, amount, date_received, source, notes, link_to_member)
            VALUES (%s, %s, %s, %s, %s, %s)
            """, income)
        
        # Expense data
        expenses = [
            ('exp1', 450, '2019-09-03', 'Party Supplies Inc', 'Supplies for welcome party', 'event1'),
            ('exp2', 200, '2019-10-08', 'Tech Store', 'Equipment for workshop', 'event2'),
            ('exp3', 750, '2019-12-10', 'Catering Company', 'Food for end of year celebration', 'event3'),
        ]
        
        for expense in expenses:
            cursor.execute("""
            INSERT INTO expense (expense_id, amount, date_spent, vendor, description, event_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            """, expense)
        
        # Budget data
        budgets = [
            ('budget1', 'Fall 2019', 5000, '2019-09-01', '2019-12-31'),
            ('budget2', 'Spring 2020', 6000, '2020-01-01', '2020-05-31'),
        ]
        
        for budget in budgets:
            cursor.execute("""
            INSERT INTO budget (budget_id, semester, total_amount, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s)
            """, budget)
        
        # Major data
        majors = [
            ('major1', 'Computer Science', 'Engineering'),
            ('major2', 'Biology', 'Sciences'),
            ('major3', 'Mathematics', 'Sciences'),
            ('major4', 'History', 'Humanities'),
            ('major5', 'English', 'Humanities'),
        ]
        
        for major in majors:
            cursor.execute("""
            INSERT INTO major (major_id, major_name, department)
            VALUES (%s, %s, %s)
            """, major)
        
        # Zip code data
        zip_codes = [
            ('12345', 'Springfield', 'IL'),
            ('23456', 'Riverdale', 'NY'),
            ('34567', 'Sunnyville', 'CA'),
            ('45678', 'Laketown', 'MI'),
            ('56789', 'Mountainview', 'CO'),
        ]
        
        for zip_code in zip_codes:
            cursor.execute("""
            INSERT INTO zip_code (zip, city, state)
            VALUES (%s, %s, %s)
            """, zip_code)
        
        # Commit changes
        conn.commit()
        print("Student club database setup completed successfully")
        
        # Verify data
        cursor.execute("SELECT COUNT(*) FROM member")
        member_count = cursor.fetchone()[0]
        print(f"Total members: {member_count}")
        
        cursor.execute("SELECT source, SUM(amount) FROM income WHERE EXTRACT(YEAR FROM TO_DATE(date_received, 'YYYY-MM-DD')) = 2019 GROUP BY source")
        income_by_source = cursor.fetchall()
        print("Income by source in 2019:")
        for source, amount in income_by_source:
            print(f"  - {source}: ${amount}")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error setting up student_club database: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        return False

def main():
    """Main function to set up all databases"""
    print("Setting up databases for MAC-SQL testing...")
    
    # Set up student_club database
    if setup_student_club_db():
        print("Database setup completed successfully")
    else:
        print("Database setup failed")

if __name__ == "__main__":
    main() 