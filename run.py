#!/usr/bin/env python
"""
MAC-SQL Agent Runner Script
===========================

This script provides a quick way to run the MAC-SQL agent with common options.

Usage:
    python run.py [options]

Options:
    --ui             Run the Streamlit UI interface (PostgreSQL)
    --ui-sqlite      Run the Streamlit UI interface (SQLite)
    --evaluate       Run benchmark evaluation (PostgreSQL)
    --evaluate-sqlite Run benchmark evaluation (SQLite)
    --db NAME        Specify database for evaluation (can be used multiple times)
    --samples N      Number of samples per database
    --model NAME     Model to use (default: meta-llama/Llama-3.3-70B-Instruct-Turbo)
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="MAC-SQL Agent Runner")
    parser.add_argument("--ui", action="store_true", help="Run the Streamlit UI (PostgreSQL)")
    parser.add_argument("--ui-sqlite", action="store_true", help="Run the Streamlit UI (SQLite)")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark evaluation (PostgreSQL)")
    parser.add_argument("--evaluate-sqlite", action="store_true", help="Run benchmark evaluation (SQLite)")
    parser.add_argument("--db", nargs="+", help="Databases to evaluate")
    parser.add_argument("--samples", type=int, default=5, help="Samples per database")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo", help="Model to use")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    if args.ui:
        # Run Streamlit UI with PostgreSQL
        print("Starting MAC-SQL Agent UI with PostgreSQL...")
        cmd = ["streamlit", "run", "mac_sql_agent.py"]
        subprocess.run(cmd)
    
    elif args.ui_sqlite:
        # Run Streamlit UI with SQLite
        print("Starting MAC-SQL Agent UI with SQLite...")
        cmd = ["streamlit", "run", "mac_sql_agent_sqlite.py"]
        subprocess.run(cmd)
    
    elif args.evaluate:
        # Run benchmark evaluation with PostgreSQL
        print("Starting MAC-SQL Agent benchmark evaluation with PostgreSQL...")
        cmd = ["python", "evaluate.py", "--model", args.model, "--samples_per_db", str(args.samples)]
        
        if args.db:
            for db in args.db:
                cmd.extend(["--db", db])
        
        if args.visualize:
            cmd.append("--visualize")
        
        subprocess.run(cmd)
    
    elif args.evaluate_sqlite:
        # Run benchmark evaluation with SQLite
        print("Starting MAC-SQL Agent benchmark evaluation with SQLite...")
        cmd = ["python", "evaluate_sqlite.py", "--model", args.model, "--samples_per_db", str(args.samples)]
        
        if args.db:
            for db in args.db:
                cmd.extend(["--db", db])
        
        if args.visualize:
            cmd.append("--visualize")
        
        subprocess.run(cmd)
    
    else:
        # Show help if no options selected
        print(__doc__)
        print("\nPlease select an option like --ui, --ui-sqlite, --evaluate, or --evaluate-sqlite")

if __name__ == "__main__":
    main() 