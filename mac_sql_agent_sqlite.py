import os
import re
import pandas as pd
import streamlit as st
from langchain_together import Together
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sqlite_adapter import SQLiteAdapter, get_sqlite_db_path

def results_are_equivalent(results1, results2):
    """
    Check if two query results are equivalent
    
    Args:
        results1: First query result as pandas DataFrame
        results2: Second query result as pandas DataFrame
    
    Returns:
        Boolean indicating if results are equivalent
    """
    # Handle None results
    if results1 is None and results2 is None:
        return True
    if results1 is None or results2 is None:
        return False
    
    # Handle empty results
    if results1.empty and results2.empty:
        return True
    if results1.empty or results2.empty:
        return False
    
    # Sort both DataFrames to ensure consistent comparison
    if not results1.empty and not results2.empty:
        # Sort by all columns
        results1 = results1.sort_values(by=list(results1.columns)).reset_index(drop=True)
        results2 = results2.sort_values(by=list(results2.columns)).reset_index(drop=True)
        
        # Compare shapes and values
        return results1.shape == results2.shape and results1.equals(results2)
    
    return False

class MACSQLAgent:
    """
    Memory, Attention, and Composition SQL Agent
    
    This agent uses:
    1. Memory: Conversation history and previously executed queries
    2. Attention: Focused retrieval of relevant schema and examples
    3. Composition: Multi-step SQL generation with reflection and refinement
    """
    
    def __init__(self, model_name, api_key=None, db_name=None):
        """Initialize the MAC-SQL Agent with specified LLM and database"""
        if api_key is None:
            api_key = "6e4593b7c0e0279476b65f144273d1ee972a47e3eb543c9649b36aaf6c114a82"  # Default key
        
        # Set up the model
        self.chat = Together(
            model=model_name,
            together_api_key=api_key,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Memory component - stores conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Attention component - schema knowledge and examples store
        self.schema_store = InMemoryStore()
        self.example_store = InMemoryStore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize the database
        self.db_name = db_name or "card_games"
        self.db_path = get_sqlite_db_path(self.db_name)
        
        if self.db_path:
            try:
                self.sqlite_adapter = SQLiteAdapter(self.db_path)
                self.db = self.sqlite_adapter.db
                self.initialize_schema_knowledge()
            except Exception as e:
                print(f"Database connection error: {e}")
        else:
            print(f"Database not found: {self.db_name}")
    
    def initialize_schema_knowledge(self):
        """Initialize schema knowledge by retrieving information from SQLite"""
        try:
            # Get all tables in the database
            tables = self.sqlite_adapter.get_tables()
            
            schema_info = []
            for table in tables:
                # Get table schema
                schema = self.sqlite_adapter.get_table_schema(table)
                
                # Format table information
                table_info = f"Table: {table}\n"
                table_info += "Columns:\n"
                for col in schema["columns"]:
                    table_info += f"  - {col}\n"
                
                if schema["foreign_keys"]:
                    table_info += "Foreign Keys:\n"
                    for fk in schema["foreign_keys"]:
                        table_info += f"  - {fk}\n"
                
                schema_info.append(table_info)
            
            # Store schema information for retrieval
            self.full_schema = "\n".join(schema_info)
            
            # Split schema into chunks for retrieval
            schema_docs = self.text_splitter.split_text(self.full_schema)
            schema_docs = [Document(page_content=chunk) for chunk in schema_docs]
            
            # Store schema documents
            for i, doc in enumerate(schema_docs):
                self.schema_store.mset([(str(i), doc)])
            
            # Create schema retriever
            self.schema_retriever = ParentDocumentRetriever(
                vectorstore=FAISS.from_documents([Document(page_content="schema")], FakeEmbeddings(size=1536)),
                docstore=self.schema_store,
                child_splitter=self.text_splitter,
                search_kwargs={"k": 5}
            )
            
            print(f"Initialized schema knowledge with {len(schema_docs)} chunks")
            
        except Exception as e:
            print(f"Error initializing schema knowledge: {e}")
    
    def retrieve_schema_context(self, query):
        """Retrieve relevant schema information based on the query"""
        try:
            # Get relevant schema chunks
            schema_docs = self.schema_retriever.get_relevant_documents(query)
            schema_context = "\n".join([doc.page_content for doc in schema_docs])
            return schema_context
        except Exception as e:
            print(f"Error retrieving schema context: {e}")
            return self.full_schema
    
    def retrieve_full_schema(self):
        """Retrieve the full database schema"""
        return self.full_schema
    
    def memory_component(self, user_query):
        """Process memory component to get conversation history"""
        # Get conversation history
        history = self.memory.load_memory_variables({})["chat_history"]
        
        # Format history for the prompt
        formatted_history = ""
        for message in history:
            if isinstance(message, HumanMessage):
                formatted_history += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"Assistant: {message.content}\n"
        
        return formatted_history
    
    def generate_sql(self, user_query):
        """Generate SQL query using the MAC approach"""
        # Memory: Get conversation history
        conversation_history = self.memory_component(user_query)
        
        # Attention: Retrieve relevant schema information
        schema_context = self.retrieve_schema_context(user_query)
        
        # Composition: Multi-step SQL generation
        # Step 1: Understand the question
        understanding_prompt = PromptTemplate.from_template(
            """You are an expert in understanding database questions.
            
            Given the following user question and database schema, explain what the question is asking for in detail.
            
            User Question: {question}
            
            Relevant Database Schema:
            {schema}
            
            Conversation History:
            {history}
            
            Explain what the question is asking for, including:
            1. What tables might be involved
            2. What columns are relevant
            3. What operations (filtering, joining, aggregation) might be needed
            4. Any ambiguities in the question that need clarification
            
            Your explanation:"""
        )
        
        understanding_chain = LLMChain(
            llm=self.chat,
            prompt=understanding_prompt,
            output_parser=StrOutputParser()
        )
        
        understanding = understanding_chain.run(
            question=user_query,
            schema=schema_context,
            history=conversation_history
        )
        
        # Step 2: Plan the SQL query
        planning_prompt = PromptTemplate.from_template(
            """You are an expert SQL query planner.
            
            Based on the user question and your understanding, create a step-by-step plan for constructing the SQL query.
            
            User Question: {question}
            
            Your Understanding: {understanding}
            
            Relevant Database Schema:
            {schema}
            
            Create a detailed plan for constructing the SQL query, including:
            1. Which tables to use
            2. How to join them (if needed)
            3. What columns to select
            4. What conditions to apply
            5. Any grouping, ordering, or aggregation needed
            
            Your SQL query plan:"""
        )
        
        planning_chain = LLMChain(
            llm=self.chat,
            prompt=planning_prompt,
            output_parser=StrOutputParser()
        )
        
        plan = planning_chain.run(
            question=user_query,
            understanding=understanding,
            schema=schema_context
        )
        
        # Step 3: Generate the SQL query
        generation_prompt = PromptTemplate.from_template(
            """You are an expert SQL developer for SQLite databases.
            
            Based on the user question, understanding, and plan, write a SQL query that answers the question.
            
            User Question: {question}
            
            Your Understanding: {understanding}
            
            Your Plan: {plan}
            
            Relevant Database Schema:
            {schema}
            
            Write a SQL query that correctly answers the question. The query should be valid SQLite syntax.
            
            SQL Query:"""
        )
        
        generation_chain = LLMChain(
            llm=self.chat,
            prompt=generation_prompt,
            output_parser=StrOutputParser()
        )
        
        sql_query = generation_chain.run(
            question=user_query,
            understanding=understanding,
            plan=plan,
            schema=schema_context
        )
        
        # Clean up the SQL response
        sql_query = self._clean_sql_response(sql_query)
        
        # Step 4: Verify the SQL query
        verification_prompt = PromptTemplate.from_template(
            """You are an expert SQL reviewer.
            
            Review the following SQL query to ensure it correctly answers the user's question and is valid SQLite syntax.
            
            User Question: {question}
            
            SQL Query:
            {sql_query}
            
            Relevant Database Schema:
            {schema}
            
            Check for:
            1. Syntax errors
            2. Logical errors
            3. Missing joins or conditions
            4. Incorrect table or column names
            
            If there are any issues, explain them and provide a corrected query. If the query looks good, just say "The query looks correct."
            
            Your review:"""
        )
        
        verification_chain = LLMChain(
            llm=self.chat,
            prompt=verification_prompt,
            output_parser=StrOutputParser()
        )
        
        verification = verification_chain.run(
            question=user_query,
            sql_query=sql_query,
            schema=schema_context
        )
        
        # If verification suggests corrections, extract the corrected query
        if "The query looks correct" not in verification:
            corrected_query = re.search(r'```sql\s*(.*?)\s*```', verification, re.DOTALL)
            if corrected_query:
                sql_query = corrected_query.group(1).strip()
            else:
                # Try to find SQL without markdown formatting
                corrected_query = re.search(r'SELECT.*?;', verification, re.DOTALL | re.IGNORECASE)
                if corrected_query:
                    sql_query = corrected_query.group(0).strip()
        
        # Update memory with the interaction
        self.memory.save_context(
            {"input": user_query},
            {"output": f"I'll translate that to SQL.\n\nSQL Query: ```sql\n{sql_query}\n```"}
        )
        
        return sql_query, understanding, plan, verification
    
    def _clean_sql_response(self, sql_response):
        """Clean up the SQL response to extract just the SQL query"""
        # Remove markdown code blocks if present
        sql_query = re.search(r'```(?:sql)?\s*(.*?)\s*```', sql_response, re.DOTALL)
        if sql_query:
            return sql_query.group(1).strip()
        
        # If no code blocks, try to extract SELECT statement
        sql_query = re.search(r'(SELECT.*?);', sql_response, re.DOTALL | re.IGNORECASE)
        if sql_query:
            return sql_query.group(1).strip() + ";"
        
        # If all else fails, return the original response
        return sql_response.strip()
    
    def execute_sql_query(self, sql_query, refinement_attempts=0):
        """Execute the SQL query and return the results"""
        try:
            # Execute the query
            results = self.sqlite_adapter.execute_query(sql_query)
            return results, None
        except Exception as e:
            error_message = str(e)
            print(f"Error executing query: {error_message}")
            
            # Attempt to refine the query if there's an error
            if refinement_attempts < 2:  # Limit refinement attempts
                refined_query = self.refine_sql_query(sql_query, error_message)
                return self.execute_sql_query(refined_query, refinement_attempts + 1)
            
            return None, error_message
    
    def refine_sql_query(self, sql_query, error_message):
        """Refine the SQL query based on error message"""
        refinement_prompt = PromptTemplate.from_template(
            """You are an expert SQL developer for SQLite databases.
            
            The following SQL query produced an error. Please fix the query to make it work.
            
            SQL Query:
            {sql_query}
            
            Error Message:
            {error_message}
            
            Database Schema:
            {schema}
            
            Please provide a corrected version of the query that fixes the error.
            
            Corrected SQL Query:"""
        )
        
        refinement_chain = LLMChain(
            llm=self.chat,
            prompt=refinement_prompt,
            output_parser=StrOutputParser()
        )
        
        refined_sql = refinement_chain.run(
            sql_query=sql_query,
            error_message=error_message,
            schema=self.retrieve_full_schema()
        )
        
        # Clean up the refined SQL
        refined_sql = self._clean_sql_response(refined_sql)
        
        return refined_sql
    
    def evaluate_on_benchmark(self, benchmark_data):
        """Evaluate the agent on benchmark data"""
        results = []
        
        for item in benchmark_data:
            question = item.get('question')
            gold_sql = item.get('query')
            db_id = item.get('db_id')
            
            # Skip if database doesn't match
            if db_id != self.db_name:
                continue
            
            print(f"\nEvaluating question: {question}")
            
            # Generate SQL
            generated_sql, understanding, plan, verification = self.generate_sql(question)
            
            # Execute generated SQL
            generated_results, generated_error = self.execute_sql_query(generated_sql)
            
            # Execute gold SQL
            gold_results, gold_error = self.execute_sql_query(gold_sql)
            
            # Check if results match
            results_match = results_are_equivalent(generated_results, gold_results)
            
            result = {
                "question": question,
                "generated_sql": generated_sql,
                "gold_sql": gold_sql,
                "results_match": results_match,
                "generated_error": generated_error,
                "gold_error": gold_error,
                "understanding": understanding,
                "plan": plan,
                "verification": verification
            }
            
            results.append(result)
            
            print(f"Results match: {results_match}")
            print(f"Generated SQL: {generated_sql}")
            print(f"Gold SQL: {gold_sql}")
            
        return results

def main():
    """Streamlit UI for the MAC-SQL Agent"""
    st.title("MAC-SQL Agent")
    st.write("Memory, Attention, and Composition for Text-to-SQL")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Model selection
    model_options = {
        "Meta Llama 3.3 70B Instruct Turbo": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Meta Llama 3.1 70B Instruct": "meta-llama/Llama-3.1-70B-Instruct",
        "Meta Llama 3.1 8B Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "Mistral 7B Instruct v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen 1.5 72B Chat": "Qwen/Qwen1.5-72B-Chat"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0
    )
    
    model_name = model_options[selected_model_name]
    
    # Database selection
    available_dbs = [
        "card_games",
        "california_schools",
        "superhero",
        "student_club",
        "toxicology",
        "thrombosis_prediction",
        "codebase_community",
        "debit_card_specializing",
        "european_football_2",
        "formula_1"
    ]
    
    selected_db = st.sidebar.selectbox(
        "Select Database",
        available_dbs,
        index=0
    )
    
    # Initialize agent
    if "agent" not in st.session_state or st.session_state.model_name != model_name or st.session_state.db_name != selected_db:
        with st.spinner("Initializing agent..."):
            st.session_state.agent = MACSQLAgent(model_name, db_name=selected_db)
            st.session_state.model_name = model_name
            st.session_state.db_name = selected_db
    
    # Query input
    user_query = st.text_area("Enter your question:", height=100)
    
    if st.button("Generate SQL"):
        if user_query:
            with st.spinner("Generating SQL..."):
                # Generate SQL
                sql_query, understanding, plan, verification = st.session_state.agent.generate_sql(user_query)
                
                # Display results
                st.subheader("Generated SQL")
                st.code(sql_query, language="sql")
                
                # Execute SQL
                with st.spinner("Executing SQL..."):
                    results, error = st.session_state.agent.execute_sql_query(sql_query)
                    
                    if error:
                        st.error(f"Error executing SQL: {error}")
                    elif results is not None:
                        st.subheader("Query Results")
                        st.dataframe(results)
                
                # Show the reasoning process
                with st.expander("Show reasoning process"):
                    st.subheader("Question Understanding")
                    st.write(understanding)
                    
                    st.subheader("Query Plan")
                    st.write(plan)
                    
                    st.subheader("Query Verification")
                    st.write(verification)
        else:
            st.warning("Please enter a question.")
    
    # Show schema
    with st.sidebar.expander("Show Database Schema"):
        if "agent" in st.session_state:
            st.text(st.session_state.agent.retrieve_full_schema())

if __name__ == "__main__":
    main() 