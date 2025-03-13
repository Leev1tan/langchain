# MAC-SQL: Memory, Attention, and Composition for Text-to-SQL

MAC-SQL is an innovative framework that uses Memory, Attention, and Composition techniques for text-to-SQL generation. 
This project implements a multi-agent collaborative architecture based on the [MAC-SQL paper](https://arxiv.org/abs/2312.11242) using LangChain, Together AI models, and provides a Streamlit interface for interaction.

## Features

- **Multi-Agent Architecture**: Three specialized agents work together
  - **Selector**: Handles schema and example selection
  - **Decomposer**: Understands questions and plans queries
  - **Refiner**: Generates and refines SQL queries
- **Memory**: Maintains conversation context and example store
- **Attention**: Focuses on relevant schema information
- **Composition**: Multi-step process for SQL generation
- **Evaluation**: Benchmark evaluation on mini-bird dataset
- **Database Support**: Works with PostgreSQL databases
- **User Interface**: Streamlit web interface with chat interface

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

3. Set up the database:
   ```
   python run_mac_sql.py --setup
   ```
   
   This will import the mini-bird benchmark databases into PostgreSQL.

### Running the Application

Use the `run_mac_sql.py` script for easy execution:

```
# Start the Streamlit UI
python run_mac_sql.py --ui

# Run benchmark evaluation
python run_mac_sql.py --evaluate

# Run a single query
python run_mac_sql.py --question "How many players are there in the card games database?"
```

Alternatively, you can directly run:

```
# Start the Streamlit UI
streamlit run app.py

# Run evaluation
python evaluate.py --benchmark dev_20240627/dev.json --samples_per_db 5
```

## Project Structure

```
.
├── core/                      # Core components
│   ├── __init__.py           # Package initialization
│   ├── agents.py             # Agent implementations
│   └── chat_manager.py       # Agent coordination
├── mac_sql.py                # Main MAC-SQL class
├── app.py                    # Streamlit UI
├── db_setup.py               # Database setup script
├── run_mac_sql.py            # Convenience runner script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Project dependencies
├── results/                  # Evaluation results directory
└── minidev/                  # Mini-bird benchmark dataset
```

## Technical Details

### Multi-Agent Architecture

The MAC-SQL framework features three specialized agents:

1. **Selector Agent**:
   - Focuses on selecting relevant schema information
   - Identifies similar examples from past queries
   - Reduces the size of context sent to other agents

2. **Decomposer Agent**:
   - Understands what the question is asking for
   - Breaks down complex questions into parts
   - Creates a step-by-step plan for SQL generation

3. **Refiner Agent**:
   - Generates SQL based on the understanding and plan
   - Verifies SQL syntax and logic
   - Refines queries when errors occur

### Workflow

1. User submits a natural language question
2. **Selector** identifies relevant schema and examples
3. **Decomposer** analyzes the question and creates a query plan
4. **Refiner** generates, verifies, and refines the SQL query
5. The query is executed against the database
6. Results are presented to the user

## Database Setup

To use MAC-SQL, you'll need to set up a PostgreSQL database and import the mini-bird benchmark datasets:

1. Ensure PostgreSQL is installed and running
2. Configure connection details in `core/chat_manager.py` if needed
3. Run the setup script:
   ```
   python run_mac_sql.py --setup
   ```

## Benchmark Evaluation

The agent is evaluated on the mini-bird benchmark which provides a set of natural language questions and expected SQL queries for different databases.

Evaluation metrics include:
- **Execution Accuracy**: Percentage of queries that produce correct results
- **Per-database Performance**: Breakdown of accuracy by database

## Requirements

- Python 3.8+
- PostgreSQL server
- Dependencies listed in requirements.txt

## Citation

If you find this implementation useful, please cite the original MAC-SQL paper:

```
@inproceedings{macsql-2025,
  title={MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL},
  author={Wang, Bing and Ren, Changyu and Yang, Jian and others},
  booktitle={Proceedings of the International Conference on Computational Linguistics},
  year={2025}
}
``` 