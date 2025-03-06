import psycopg2
from langchain.prompts import PromptTemplate
from langchain_together import ChatTogether
import re
import pandas as pd
import streamlit as st

# PostgreSQL connection configuration
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "superuser",
    "host": "localhost",
    "port": "5432"
}
class TextToSQLApp:
    def __init__(self, model_name):
        self.chat = ChatTogether(model=model_name, api_key="6e4593b7c0e0279476b65f144273d1ee972a47e3eb543c9649b36aaf6c114a82")

    def retrieve_full_schema(self):
        """
        Retrieve the full schema information from the PostgreSQL database,
        including sample data for each column.
        """
        try:
            conn = psycopg2.connect(**DB_CONFIG)
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
                    cursor.execute(f"SELECT {column} FROM {table} LIMIT 3;")
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
                print(context)
                return context.strip()
            else:
                return "No schema information found."

        except Exception as e:
            print(f"Error retrieving schema: {e}")
            return None

    def generate_sql(self, user_query):
        """Generate an SQL query based on the user's request without extra explanations."""
        schema_context = self.retrieve_full_schema()
        if not schema_context:
            return "No schema information found."

        prompt_template = PromptTemplate(
            input_variables=["user_query", "context"],
            template="""
            The following database schema is provided for understanding the structure.
            {context}
            
            Generate an SQL query for the user request in Ukrainian. Return only the SQL code, without any explanations or additional text. If you're not 100% sure about the relevance of query to database then generate sql query "SELECT 'No Answer'"
            
            User Request: {user_query}

            SQL Query:
            """
        )

        formatted_prompt = prompt_template.format(user_query=user_query, context=schema_context)
        print(formatted_prompt)
        response = self.chat.invoke(formatted_prompt)

        # Extract only the SQL query
        sql_response = response.content.strip()

        # Remove any markdown code blocks and text before/after the SQL query
        sql_response = re.sub(r"^.*?SELECT", "SELECT", sql_response, flags=re.IGNORECASE | re.DOTALL)
        sql_response = re.sub(r";.*$", ";", sql_response, flags=re.DOTALL)

        # Remove any trailing explanations or text after the semicolon
        sql_response = sql_response.split(';')[0] + ';'

        return sql_response.strip()

    def refine_sql_query(self, sql_query, error_message):
        """Ask the model to refine the SQL query based on the error encountered."""
        prompt_template = PromptTemplate(
            input_variables=["sql_query", "error"],
            template="""
            The following SQL query generated an error:
            SQL Query: {sql_query}
            Error: {error}

            Please refine the query to correct the error and generate a valid SQL query.

            Return only the corrected SQL query, without explanations.

            Refined SQL Query:
            """
        )
        refinement_prompt = prompt_template.format(sql_query=sql_query, error=error_message)
        response = self.chat.invoke(refinement_prompt)

        # Extract only the SQL query
        refined_sql = response.content.strip()
        refined_sql = re.sub(r"^.*?SELECT", "SELECT", refined_sql, flags=re.IGNORECASE | re.DOTALL)
        refined_sql = re.sub(r";.*$", ";", refined_sql, flags=re.DOTALL)
        refined_sql = refined_sql.strip()

        return refined_sql

    def execute_sql_query(self, sql_query, refinement_attempts=0):
        """Execute the generated SQL query and handle errors if they arise."""
        MAX_REFINEMENT_ATTEMPTS = 2
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            return pd.DataFrame(rows, columns=columns)
        except psycopg2.Error as e:
            if refinement_attempts < MAX_REFINEMENT_ATTEMPTS:
                refined_query = self.refine_sql_query(sql_query, str(e))

                # Ensure the refined query is cleaned as well
                refined_query = re.sub(r"^.*?SELECT", "SELECT", refined_query, flags=re.IGNORECASE | re.DOTALL)
                refined_query = re.sub(r";.*$", ";", refined_query, flags=re.DOTALL)
                refined_query = refined_query.strip()

                return self.execute_sql_query(refined_query, refinement_attempts + 1)
            else:
                return f"Помилка виконання запиту: {e}"

    def is_relevant_query(self, sql_query):
        """Check if the SQL query is relevant by ensuring it references tables in the PostgreSQL schema."""
        try:
            # Connect to the PostgreSQL database to retrieve the actual list of tables
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
            """)
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Check if any of these tables are mentioned in the SQL query
            for table in tables:
                if re.search(rf"\b{table}\b", sql_query, re.IGNORECASE):
                    return True
            return False

        except Exception as e:
            print(f"Error checking query relevance: {e}")
            return False

    def run(self):
        """
        Main function to process a user query from start to finish.
        """
        # Get user input for the query
        user_query = input("Введіть свій запит: ")

        # Step 1: Generate SQL based on user query
        sql_query = self.generate_sql(user_query)
        print(f"Generated SQL: {sql_query}")

        # Step 2: Execute the SQL and check for errors
        execution_result = self.execute_sql_query(sql_query)
        if isinstance(execution_result, str) and "Error" in execution_result:
            print(execution_result)
            return None
        else:
            return execution_result

# Streamlit Interface
st.title("Інтерфейс Text-to-SQL з використанням MAC-SQL і Llama 3 70B")
model_options = {
    "Meta Llama 3.2 11B Vision Instruct Turbo": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "Qwen2.5 7B Instruct Turbo": "qwen2.5-7B-Instruct-Turbo",
    "Meta Llama 3.2 3B Instruct Turbo": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "Meta Llama Vision Free": "meta-llama/Llama-Vision-Free",
    "Meta Llama 3.1 8B Instruct Turbo": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Meta Llama 3.1 70B Instruct Turbo": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "Gryphe MythoMax L2 Lite (13B)": "gryphe-mythomax-l2-lite-13B",
    "Meta Llama 3 70B Instruct Lite": "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
    "Meta Llama 3 8B Instruct Lite": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "Gemma-2 Instruct (9B)": "gemma-2-9B-Instruct",
    "Mistral (7B) Instruct v0.3": "mistral-7B-Instruct-v0.3",
    "Qwen 1.5 Chat (72B)": "qwen-1.5-72B-Chat"
}
model_choice = st.selectbox("Виберіть модель AI:", options=list(model_options.keys()))

user_query = st.text_input("Введіть своє запитання щодо бази даних:", "")

if st.button("Надіслати запит"):
    if user_query:
        selected_model = model_options[model_choice]
        app = TextToSQLApp(model_name=selected_model)

        # Generate SQL
        sql_query = app.generate_sql(user_query)
        st.write(f"Generated SQL Query:\n{sql_query}")

        # Check if the query is relevant
        if "SELECT" in sql_query:
            if app.is_relevant_query(sql_query):
                result = app.execute_sql_query(sql_query)
                if isinstance(result, pd.DataFrame):
                    st.write("Query Results:")
                    st.dataframe(result)
                else:
                    st.write(result)
            else:
                st.write("Не релевантний запит")
        else:
            st.write("Не вдалося створити SQL запит")
    else:
        st.write("Будь ласка, задайте питання")

