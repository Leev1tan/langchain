# MAC-SQL: Memory, Attention, and Composition for Text-to-SQL

MAC-SQL is an innovative agent that uses Memory, Attention, and Composition techniques for text-to-SQL generation. 
This project implements the MAC-SQL agent using LangChain, Together AI models, and provides a Streamlit interface for interaction.

## Features

- **Memory**: Maintains conversation context for better SQL generation
- **Attention**: Focuses on relevant schema information
- **Composition**: Multi-step process for SQL generation
- **Evaluation**: Benchmark evaluation on mini-bird dataset
- **User Interface**: Streamlit web interface for interaction
- **Database Support**: Works with both PostgreSQL and SQLite

## Quick Start

### Setup

1. Create and activate virtual environment:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure PostgreSQL settings in `int.py` (already provided) if using PostgreSQL.

### Running the Application

Use the `run.py` script for easy execution:

```
# Start the Streamlit UI with PostgreSQL
python run.py --ui

# Start the Streamlit UI with SQLite (recommended for benchmark)
python run.py --ui-sqlite

# Run benchmark evaluation with PostgreSQL
python run.py --evaluate --db card_games test_results

# Run benchmark evaluation with SQLite (recommended)
python run.py --evaluate-sqlite --db card_games --visualize
```

Alternatively, you can directly run:

```
# Start the Streamlit UI with PostgreSQL
streamlit run mac_sql_agent.py

# Start the Streamlit UI with SQLite
streamlit run mac_sql_agent_sqlite.py

# Run evaluation with PostgreSQL
python evaluate.py --db card_games test_results

# Run evaluation with SQLite
python evaluate_sqlite.py --db card_games test_results
```

## Project Structure

```
.
├── mac_sql_agent.py         # PostgreSQL-based agent implementation
├── mac_sql_agent_sqlite.py  # SQLite-based agent implementation
├── sqlite_adapter.py        # SQLite database adapter
├── evaluate.py              # PostgreSQL evaluation script
├── evaluate_sqlite.py       # SQLite evaluation script
├── run.py                   # Convenience script for running the project
├── requirements.txt         # Project dependencies
├── int.py                   # PostgreSQL configuration
├── results/                 # Evaluation results
└── dev_20240627/            # Mini-bird benchmark dataset
```

## Technical Details

MAC-SQL leverages LangChain to implement a text-to-SQL agent with memory, attention, and composition capabilities:

1. **Memory**: Conversation buffer memory for context tracking
2. **Attention**: Schema-based retrieval to focus on relevant database information
3. **Composition**: Step-by-step SQL generation:
   - Question understanding
   - Schema analysis
   - Query planning
   - SQL generation
   - SQL validation

## Database Support

The project supports two database backends:

1. **PostgreSQL**: Requires a PostgreSQL server and database setup
2. **SQLite**: Uses the SQLite databases included in the mini-bird benchmark

For benchmark evaluation, the SQLite implementation is recommended as it:
- Works directly with the benchmark databases
- Eliminates SQL dialect conversion issues
- Doesn't require setting up a PostgreSQL server

## Benchmark Evaluation

The agent is evaluated on the mini-bird benchmark which provides a set of natural language questions and expected SQL queries for different databases.

The evaluation script supports:
- Filtering by database
- Limiting samples per database
- Result visualization
- Detailed error analysis

## Requirements

- Python 3.8+
- PostgreSQL (optional)
- Dependencies listed in requirements.txt

## License

This project is for educational purposes only. 