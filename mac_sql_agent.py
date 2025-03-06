import os
import re
import psycopg2
import pandas as pd
import streamlit as st
from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# PostgreSQL connection configuration
DB_CONFIG = {
    "dbname": "card_games",
    "user": "postgres",
    "password": "superuser",
    "host": "localhost",
    "port": "5432"
}

def adapt_sql_dialect(sql_query, source_dialect="mysql", target_dialect="postgresql"):
    """
    Convert SQL query from one dialect to another
    
    Args:
        sql_query: Original SQL query
        source_dialect: Source dialect (mysql, sqlite)
        target_dialect: Target dialect (postgresql)
    
    Returns:
        Converted SQL query
    """
    if source_dialect.lower() in ["mysql", "sqlite"] and target_dialect.lower() == "postgresql":
        # 1. Replace backticks with double quotes for identifiers
        sql_query = re.sub(r'`([^`]+)`', r'"\1"', sql_query)
        
        # 2. Handle IFNULL vs COALESCE
        sql_query = re.sub(r'IFNULL\s*\(', 'COALESCE(', sql_query, flags=re.IGNORECASE)
        
        # 3. Handle boolean literals
        sql_query = re.sub(r'\bTRUE\b', 'true', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bFALSE\b', 'false', sql_query, flags=re.IGNORECASE)
        
        return sql_query
    
    # If dialects are the same or unsupported combination, return original
    return sql_query

def results_are_equivalent(results1, results2):
    """Check if two SQL query results are equivalent"""
    import pandas as pd
    
    # Convert to pandas DataFrames for easier comparison
    if not isinstance(results1, pd.DataFrame):
        df1 = pd.DataFrame(results1)
    else:
        df1 = results1
        
    if not isinstance(results2, pd.DataFrame):
        df2 = pd.DataFrame(results2)
    else:
        df2 = results2
    
    # If DataFrames have different shapes, they're not equivalent
    if df1.shape != df2.shape:
        return False
    
    # Check if DataFrames have the same values (ignoring column names)
    # Sort both DataFrames if possible to handle order differences
    try:
        df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
        df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)
        return df1_sorted.equals(df2_sorted)
    except:
        # If sorting fails (e.g., due to mixed types), compare as is
        return df1.equals(df2)

class MACSQLAgent:
    """
    Memory, Attention, and Composition (MAC) SQL Agent.
    
    This agent uses:
    1. Memory: Conversation history and previously executed queries
    2. Attention: Focused retrieval of relevant schema and examples
    3. Composition: Multi-step SQL generation with reflection and refinement
    """
    
    def __init__(self, model_name, api_key=None, db_config=None):
        """Initialize the MAC-SQL Agent with specified LLM"""
        if api_key is None:
            api_key = "6e4593b7c0e0279476b65f144273d1ee972a47e3eb543c9649b36aaf6c114a82"  # Default key from the int.py
        
        # Use provided DB_CONFIG or default
        self.db_config = db_config or DB_CONFIG
        
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
        
        # Initialize the database schema and examples
        self.db_uri = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        try:
            self.db = SQLDatabase.from_uri(self.db_uri)
            self.initialize_schema_knowledge()
        except Exception as e:
            print(f"Database connection error: {e}")
    
    def initialize_schema_knowledge(self):
        """Initialize schema knowledge by retrieving information from PostgreSQL"""
        try:
            # Connect to the database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get all tables in the database
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = []
            for table in tables:
                # Get column information
                cursor.execute(f"""
                    SELECT 
                        column_name, 
                        data_type,
                        column_default,
                        is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    AND table_schema = 'public';
                """)
                columns = cursor.fetchall()
                
                # Format column information
                col_info = []
                for col in columns:
                    col_name, data_type, default, nullable = col
                    col_info.append(f"{col_name} ({data_type}, {'NULL' if nullable=='YES' else 'NOT NULL'}, {'DEFAULT '+str(default) if default else 'NO DEFAULT'})")
                
                # Get primary key information
                cursor.execute(f"""
                    SELECT
                        c.column_name
                    FROM
                        information_schema.table_constraints tc
                    JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                    JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                        AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                    WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = '{table}';
                """)
                pk_cols = [row[0] for row in cursor.fetchall()]
                
                # Get foreign key information
                cursor.execute(f"""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM
                        information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.constraint_schema = kcu.constraint_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.constraint_schema = tc.constraint_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table}'
                """)
                fk_info = cursor.fetchall()
                
                # Sample data
                cursor.execute(f'SELECT * FROM "{table}" LIMIT 3')
                sample_data = cursor.fetchall()
                sample_cols = [desc[0] for desc in cursor.description]
                
                # Create schema document
                schema_doc = f"Table: {table}\n"
                schema_doc += f"Columns: {', '.join(col_info)}\n"
                if pk_cols:
                    schema_doc += f"Primary Key: {', '.join(pk_cols)}\n"
                if fk_info:
                    fk_str = []
                    for fk in fk_info:
                        col, ref_table, ref_col = fk
                        fk_str.append(f"{col} -> {ref_table}({ref_col})")
                    schema_doc += f"Foreign Keys: {', '.join(fk_str)}\n"
                
                # Add sample data
                schema_doc += "Sample Data:\n"
                for i, row in enumerate(sample_data):
                    schema_doc += f"Row {i+1}: " + ", ".join([f"{col}={val}" for col, val in zip(sample_cols, row)]) + "\n"
                
                schema_info.append(schema_doc)
            
            # Store schema information in InMemoryStore
            docs = [Document(page_content=doc, metadata={"source": "schema"}) for doc in schema_info]
            self.schema_store.mset([(str(i), doc) for i, doc in enumerate(docs)])
            
            conn.close()
            
        except Exception as e:
            print(f"Error initializing schema knowledge: {e}")
    
    def retrieve_schema_context(self, query):
        """Retrieve relevant schema context based on the user query"""
        try:
            # Create a fake embeddings model since we're using in-memory
            embeddings = FakeEmbeddings(size=768)
            
            # Create a FAISS vectorstore for retrieval
            vectorstore = FAISS.from_texts(
                ["dummy text"], 
                embeddings, 
                metadatas=[{"source": "schema"}]
            )
            
            # Create a retriever that focuses on schema information
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=self.schema_store,
                child_splitter=self.text_splitter,
                search_kwargs={"k": 5}
            )
            
            # Retrieve relevant schema information
            docs = retriever.get_relevant_documents(query)
            
            if docs:
                return "\n\n".join([doc.page_content for doc in docs])
            else:
                return self.retrieve_full_schema()
                
        except Exception as e:
            print(f"Error retrieving schema context: {e}")
            return self.retrieve_full_schema()
    
    def retrieve_full_schema(self):
        """Fallback method to retrieve the full database schema"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Retrieve all tables from the database
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
            """)
            tables = [row[0] for row in cursor.fetchall()]

            context = ""
            for table in tables:
                # Get column names for the table
                cursor.execute(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    AND table_schema = 'public';
                """)
                columns = [row[0] for row in cursor.fetchall()]

                # Get sample data for each column
                sample_data = {}
                for column in columns:
                    cursor.execute(f'SELECT "{column}" FROM "{table}" LIMIT 3;')
                    data = [str(row[0]) for row in cursor.fetchall()]
                    sample_data[column] = data

                # Format the table and columns with sample data
                context += f"Table {table}\n"
                columns_with_data = []
                for column in columns:
                    data_str = ", ".join(sample_data[column])
                    columns_with_data.append(f"{column} ({data_str})")
                context += "Columns: " + ", ".join(columns_with_data) + "\n\n"

            conn.close()

            if context:
                return context.strip()
            else:
                return "No schema information found."

        except Exception as e:
            print(f"Error retrieving schema: {e}")
            return "Error retrieving database schema information."
    
    def memory_component(self, user_query):
        """Extract relevant information from conversation memory"""
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # If memory exists, format it for the LLM prompt
        if chat_history:
            formatted_history = "\n".join([
                f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
                for msg in chat_history
            ])
            return formatted_history
        
        return ""
    
    def generate_sql(self, user_query):
        """Generate SQL based on user query using the MAC approach"""
        # 1. Memory - Retrieve conversation history
        memory_context = self.memory_component(user_query)
        
        # 2. Attention - Retrieve relevant schema information
        schema_context = self.retrieve_schema_context(user_query)
        
        # 3. Composition - Generate SQL through multi-step reasoning
        # Step 1: Analyze the query and understand requirements
        analysis_prompt = PromptTemplate(
            input_variables=["user_query", "schema", "memory"],
            template="""
            You are tasked with translating a natural language question into an SQL query.
            
            Database Schema Information:
            {schema}
            
            Previous Conversation:
            {memory}
            
            User Question: {user_query}
            
            First, analyze this question and identify:
            1. What tables and columns are needed?
            2. What conditions should be applied?
            3. What kind of operation is needed (SELECT, COUNT, AVG, JOIN, etc.)?
            4. Are there any ambiguities or assumptions that need to be addressed?
            
            Analysis:
            """
        )
        
        # Use the newer chain format with output parser
        parser = StrOutputParser()
        analysis_chain = analysis_prompt | self.chat | parser
        analysis = analysis_chain.invoke({
            "user_query": user_query, 
            "schema": schema_context, 
            "memory": memory_context
        })
        
        # Step 2: Generate the SQL query based on the analysis
        generation_prompt = PromptTemplate(
            input_variables=["user_query", "schema", "analysis", "memory"],
            template="""
            You are an expert SQL generator translating natural language questions to SQL queries.
            
            Database Schema Information:
            {schema}
            
            Previous Conversation:
            {memory}
            
            User Question: {user_query}
            
            Analysis of the question:
            {analysis}
            
            Based on this analysis, generate a precise SQL query that answers the user's question.
            Return ONLY the SQL query without explanation or markdown formatting.
            """
        )
        
        # Use the newer chain format
        generation_chain = generation_prompt | self.chat | parser
        sql_query = generation_chain.invoke({
            "user_query": user_query, 
            "schema": schema_context, 
            "analysis": analysis,
            "memory": memory_context
        })
        
        # Clean up SQL - remove any explanation or markdown
        sql_query = self._clean_sql_response(sql_query)
        
        # Step 3: Verify and refine the SQL query
        verification_prompt = PromptTemplate(
            input_variables=["user_query", "schema", "sql_query"],
            template="""
            You are an expert SQL reviewer. Verify that the following SQL query correctly answers the user's question.
            
            Database Schema Information:
            {schema}
            
            User Question: {user_query}
            
            Generated SQL Query:
            {sql_query}
            
            Is this query correct? If not, provide a corrected version.
            Return ONLY the final SQL query without explanation or markdown formatting.
            """
        )
        
        # Use the newer chain format
        verification_chain = verification_prompt | self.chat | parser
        verified_sql = verification_chain.invoke({
            "user_query": user_query,
            "schema": schema_context,
            "sql_query": sql_query
        })
        
        # Clean up verified SQL
        verified_sql = self._clean_sql_response(verified_sql)
        
        # Update memory with this interaction
        self.memory.save_context(
            {"input": user_query}, 
            {"output": f"Generated SQL: {verified_sql}"}
        )
        
        return verified_sql
    
    def _clean_sql_response(self, sql_response):
        """Clean SQL response to extract only the SQL query"""
        # Remove markdown code blocks
        sql_response = re.sub(r"```sql|```", "", sql_response)
        
        # Remove comments
        sql_response = re.sub(r"/\*.*?\*/", "", sql_response, flags=re.DOTALL)
        
        # Extract only SQL (assuming it starts with SELECT, WITH, etc.)
        sql_patterns = [r"SELECT", r"WITH", r"INSERT", r"UPDATE", r"DELETE", r"CREATE", r"ALTER", r"DROP"]
        for pattern in sql_patterns:
            match = re.search(f"{pattern}.*", sql_response, re.IGNORECASE | re.DOTALL)
            if match:
                sql_response = match.group(0)
                break
        
        return sql_response.strip()
    
    def execute_sql_query(self, sql_query, refinement_attempts=0):
        """Execute the generated SQL query and handle errors if they arise"""
        MAX_REFINEMENT_ATTEMPTS = 2
        try:
            # First adapt the SQL query to PostgreSQL dialect if needed
            if '`' in sql_query:  # Simple check for MySQL/SQLite dialect
                sql_query = adapt_sql_dialect(sql_query, "mysql", "postgresql")
                
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            return pd.DataFrame(rows, columns=columns)
        except psycopg2.Error as e:
            if refinement_attempts < MAX_REFINEMENT_ATTEMPTS:
                refined_query = self.refine_sql_query(sql_query, str(e))
                return self.execute_sql_query(refined_query, refinement_attempts + 1)
            else:
                return f"Error executing query: {e}"
    
    def refine_sql_query(self, sql_query, error_message):
        """Ask the model to refine the SQL query based on the error encountered"""
        prompt_template = PromptTemplate(
            input_variables=["sql_query", "error", "schema"],
            template="""
            The following SQL query generated an error:
            SQL Query: {sql_query}
            Error: {error}
            
            Database Schema Information:
            {schema}
            
            Please refine the query to correct the error and generate a valid SQL query.
            Return only the corrected SQL query, without explanations or markdown formatting.
            """
        )
        
        schema_context = self.retrieve_full_schema()
        refinement_prompt = prompt_template.format(
            sql_query=sql_query, 
            error=error_message,
            schema=schema_context
        )
        
        parser = StrOutputParser()
        chain = self.chat | parser
        refined_sql = chain.invoke(refinement_prompt)
        refined_sql = self._clean_sql_response(refined_sql)
        
        return refined_sql
    
    def evaluate_on_benchmark(self, benchmark_data):
        """
        Evaluate the MAC-SQL agent on a benchmark dataset with SQL dialect adaptation
        
        Args:
            benchmark_data: List of dictionaries with 'question' and 'SQL' keys
            
        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        total = len(benchmark_data)
        results = []
        
        for i, item in enumerate(benchmark_data):
            question = item['question']
            gold_sql = item['SQL']
            
            # Generate SQL for the question
            generated_sql = self.generate_sql(question)
            
            # Adapt gold SQL to PostgreSQL dialect
            adapted_gold_sql = adapt_sql_dialect(gold_sql, "mysql", "postgresql")
            
            # Execute both the generated and gold SQL queries
            try:
                generated_results = self.execute_sql_query(generated_sql)
                gold_results = self.execute_sql_query(adapted_gold_sql)
                
                # Check if results match
                if isinstance(generated_results, pd.DataFrame) and isinstance(gold_results, pd.DataFrame):
                    results_match = results_are_equivalent(generated_results, gold_results)
                else:
                    results_match = False
                    
                if results_match:
                    correct += 1
                    
                results.append({
                    'question_id': item.get('question_id', i),
                    'question': question,
                    'gold_sql': gold_sql,
                    'adapted_gold_sql': adapted_gold_sql,
                    'generated_sql': generated_sql,
                    'results_match': results_match
                })
                
            except Exception as e:
                results.append({
                    'question_id': item.get('question_id', i),
                    'question': question,
                    'gold_sql': gold_sql,
                    'adapted_gold_sql': adapted_gold_sql,
                    'generated_sql': generated_sql,
                    'error': str(e),
                    'results_match': False
                })
        
        # Calculate execution accuracy
        execution_accuracy = correct / total if total > 0 else 0
        
        return {
            'execution_accuracy': execution_accuracy,
            'detailed_results': results
        }

# Streamlit Interface
def main():
    st.set_page_config(page_title="MAC-SQL Agent", page_icon="🤖", layout="wide")
    
    st.title("MAC-SQL Agent: Memory, Attention & Composition for Text-to-SQL")
    st.subheader("Powered by LangChain and Together API")
    
    # Model selection
    st.sidebar.header("Model Selection")
    model_options = {
        "Meta Llama 3.3 70B Instruct": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Meta Llama 3.1 70B Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "Meta Llama 3.1 8B Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        "Mistral 7B Instruct v0.3": "mistral-7B-Instruct-v0.3",
        "Qwen 1.5 72B Chat": "qwen-1.5-72B-Chat",
        "Qwen2.5 7B Instruct": "qwen2.5-7B-Instruct"
    }
    model_choice = st.sidebar.selectbox("Select Model:", options=list(model_options.keys()))
    
    # Database selection
    st.sidebar.header("Database Selection")
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
    db_choice = st.sidebar.selectbox("Select Database:", options=available_dbs)
    
    # API Key input (optional)
    api_key = st.sidebar.text_input("Together API Key (optional):", type="password")
    if not api_key:
        api_key = "6e4593b7c0e0279476b65f144273d1ee972a47e3eb543c9649b36aaf6c114a82"  # Default key
    
    # Initialize agent with selected database
    if 'agent' not in st.session_state or st.session_state.db_choice != db_choice:
        db_config = DB_CONFIG.copy()
        db_config["dbname"] = db_choice
        st.session_state.agent = MACSQLAgent(
            model_name=model_options[model_choice],
            api_key=api_key,
            db_config=db_config
        )
        st.session_state.db_choice = db_choice
    
    # Reset agent if model changes
    if st.sidebar.button("Reset Agent"):
        db_config = DB_CONFIG.copy()
        db_config["dbname"] = db_choice
        st.session_state.agent = MACSQLAgent(
            model_name=model_options[model_choice],
            api_key=api_key,
            db_config=db_config
        )
        st.session_state.chat_history = []
        st.session_state.db_choice = db_choice
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sql" in message:
                st.code(message["sql"], language="sql")
            if "result" in message and isinstance(message["result"], pd.DataFrame):
                st.dataframe(message["result"])
    
    # User input
    user_query = st.chat_input(f"Ask a question about the {db_choice} database")
    
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Generate SQL
        with st.spinner("Generating SQL query..."):
            sql_query = st.session_state.agent.generate_sql(user_query)
        
        # Execute SQL
        with st.spinner("Executing query..."):
            result = st.session_state.agent.execute_sql_query(sql_query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write("Here's the SQL query to answer your question:")
            st.code(sql_query, language="sql")
            
            if isinstance(result, pd.DataFrame):
                st.write("Query Results:")
                st.dataframe(result)
            else:
                st.error(f"Error: {result}")
        
        # Add assistant message to chat history
        assistant_message = {
            "role": "assistant",
            "content": "Here's the SQL query to answer your question:",
            "sql": sql_query
        }
        if isinstance(result, pd.DataFrame):
            assistant_message["result"] = result
        else:
            assistant_message["content"] += f"\nError: {result}"
        
        st.session_state.chat_history.append(assistant_message)
    
    # Benchmark evaluation section
    st.sidebar.header("Benchmark Evaluation")
    if st.sidebar.button("Evaluate on mini-bird Benchmark"):
        st.sidebar.info("Benchmark evaluation starting...")
        # This would typically open a new page or display results in the main area
        st.warning("Benchmark evaluation not fully implemented in UI. Use the evaluation.py script directly.")

if __name__ == "__main__":
    main() 