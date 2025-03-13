#!/usr/bin/env python
"""
List all database IDs in the BIRD benchmark
"""

import json
import os

def list_dbs(benchmark_file="minidev/MINIDEV/mini_dev_postgresql.json"):
    """List all database IDs in the benchmark file"""
    try:
        with open(benchmark_file, 'r') as f:
            data = json.load(f)
        
        # Extract unique database IDs
        db_ids = set()
        for item in data:
            db_id = item.get("db_id", "")
            if db_id:
                db_ids.add(db_id)
        
        # Print sorted list
        print(f"Found {len(db_ids)} unique databases in {benchmark_file}:")
        for db_id in sorted(db_ids):
            print(f"- {db_id}")
        
        return db_ids
    except Exception as e:
        print(f"Error: {e}")
        return set()

if __name__ == "__main__":
    list_dbs() 