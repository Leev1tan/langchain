# MAC-SQL Project Summary

## Project Overview

This project implements a version of MAC-SQL (Multi-Agent Collaborative SQL), a framework for text-to-SQL generation based on the paper "MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL". The framework uses multiple specialized agents working together to transform natural language questions into accurate SQL queries. Our own purposes: PostgreSQL, opensource model like Llama through langchain together API. 

## Architecture
 
The MAC-SQL framework consists of three main agents:

1. **Selector Agent**: Identifies relevant schema elements based on the user query
2. **Decomposer Agent**: Core agent for Text-to-SQL generation with few-shot chain-of-thought reasoning
3. **Refiner Agent**: Refines and fixes SQL queries when execution errors occur

## Implementation Progress

So far, we have:

- Implemented the core components of the MAC-SQL framework
- Enhanced the `ChatManager` class to handle database connections and schema retrieval
- Improved the `RefinerAgent` to better handle errors and refine SQL queries
- Added sophisticated error detection and fallback mechanisms in the `refine_sql` method
- Implemented schema handling and SQL generation for different database types
- Added visualization functionality for evaluation results including SQL syntax accuracy

## Current Challenges

We are currently addressing several challenges:

1. **Schema Handling Issues**: Our implementation struggles with correctly identifying and accessing database schemas, particularly with the student_club database in the BIRD benchmark. 

2. **Table Name Discrepancies**: Tests show issues where the system tries to query non-existent tables like "members", "customer", and "customers" instead of the correct "member" table.

3. **Semantic Accuracy**: While we're achieving good SQL syntax accuracy, semantic correctness (returning correct results) is still low due to schema mismatch issues.

4. **Implementation Differences**: Our implementation differs significantly from the original MAC-SQL repository in how it handles schemas, agents, and SQL generation.

## Latest Findings

After analyzing the original MAC-SQL repository, we've discovered several critical differences:

1. **Agent Architecture**: The original uses a message-passing architecture between dedicated agent classes (`Selector`, `Decomposer`, `Refiner`), each with specific responsibilities.

2. **Schema Handling**:
   - The original has sophisticated schema loading directly from SQLite files(we use PostgreSQL, and stay stick to PostgreSQL)
   - Includes hardcoded fallbacks for specific databases (e.g., student_club database)
   - Uses a schema pruning system that selects only relevant tables and columns

3. **Table Structure**: The student_club database has multiple tables, including:
   - member (the main table - which is what the gold SQL uses)
   - event
   - attendance
   - income
   - and others

4. **Fallback Mechanisms**: Contains hardcoded fallbacks for database schemas like:
   ```python
   if schema_name == 'student_club':
       return "Tables: member (member_id, first_name, last_name, position), event (event_id, event_name, type, date), attendance (attendance_id, link_to_member, link_to_event), income (income_id, amount, date_received, link_to_member)"
   ```

## Evaluation Results

When evaluating on the BIRD benchmark, particularly the student_club database, we observed:

- **SQL Syntax Accuracy**: High (around 80-100%)
- **Execution Accuracy**: Varies, but sometimes reaches 100% with fallbacks
- **Semantic Accuracy**: Low (often 0%) due to schema mismatches and incorrect table access

The discrepancy between syntax and semantic accuracies highlights the importance of correct schema handling - queries can be syntactically correct but semantically wrong if they target the wrong tables.

## Next Steps

To improve our implementation, we should:

1. **Enhanced Schema Handling**:
   - Add hardcoded fallbacks for common databases like student_club
   - Ensure the system correctly identifies and uses primary tables (e.g., "member" for student_club)
   - Implement better schema loading from SQLite files

2. **Table Name Resolution**:
   - Add validation to ensure queries target existing tables
   - Implement table name correction based on schema knowledge

3. **Architecture Alignment**:
   - Consider aligning our implementation more closely with the original MAC-SQL architecture
   - Implement a similar message-passing system between agents if beneficial

4. **Comprehensive Testing**:
   - Test on a wider range of databases and queries
   - Focus on improving semantic accuracy alongside syntax accuracy

These improvements should help bridge the gap between our implementation and the original MAC-SQL framework, leading to better evaluation results on benchmarks like BIRD. 