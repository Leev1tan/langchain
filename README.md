# MAC-SQL: Multi-Agent Collaboration for SQL Generation

MAC-SQL is a multi-agent collaborative framework for text-to-SQL generation, based on the original [MAC-SQL paper](https://arxiv.org/abs/2306.00738). This implementation uses advanced Large Language Models (LLMs) and a multi-agent approach to generate accurate SQL from natural language queries.

## Key Features

- **Multi-Agent Architecture**: Three specialized agents collaborate to generate SQL:
  - **Selector Agent**: Identifies relevant tables and columns from the database schema
  - **Decomposer Agent**: Understands the question and creates a query plan
  - **Refiner Agent**: Generates and refines SQL until it executes successfully

- **Enhanced Schema Handling**:
  - Schema caching for improved performance
  - Automatic table name correction
  - Hardcoded schemas for known BIRD benchmark databases

- **Robust SQL Generation**:
  - SQL refinement based on execution errors
  - Table and column name correction for common issues
  - Fallback mechanisms when generation fails

- **Streamlit Dashboard**:
  - Interactive query interface
  - Evaluation results visualization
  - Benchmark runner for testing on BIRD

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running Queries

```python
from mac_sql import MACSQL

# Initialize MAC-SQL
mac_sql = MACSQL(model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo")

# Connect to a database
mac_sql.connect_to_database("your_database")

# Process a query
result = mac_sql.process_query("List all customers who made purchases in the last month")

# Access results
sql_query = result['sql']
data = result['results']
```

### Running the Dashboard

```bash
streamlit run app.py
```

### Running Benchmarks

```bash
python run_evaluation.py --benchmark "path/to/benchmark.json" --num_samples 5 --output_file "results/evaluation.json"
```

## Recent Improvements

1. **Fixed Schema Selection**:
   - Implemented robust schema extraction from database
   - Added fallback mechanisms for known databases
   - Fixed issues with the `select_schema` method

2. **Better Database Connectivity**:
   - Improved handling for BIRD benchmark databases
   - Added specialized database schema caching
   - Enhanced error handling for connection issues

3. **Robust SQL Generation and Refinement**:
   - Implemented automatic error fixing based on error patterns
   - Added specialized handling for common BIRD benchmark tables
   - Enhanced SQL extraction from LLM responses

4. **Error Recovery**:
   - Added sophisticated SQL refinement prompts with database context
   - Implemented fallback query generation
   - Added automatic table name correction

5. **Enhanced Evaluation**:
   - Added detailed SQL comparison for benchmarks
   - Improved similarity scoring for SQL evaluation
   - Added visualization of evaluation results

## Limitations and Future Work

- Currently works best with PostgreSQL databases
- Performance on complex nested queries can be improved
- Adding more specialized handling for additional benchmark databases
- Incorporating user feedback for continual improvement
