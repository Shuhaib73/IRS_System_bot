import os
import time
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Importing Flask related modules for web application development
from flask import Flask, render_template, redirect, request, url_for, session, get_flashed_messages, jsonify

# Importing the Pinecone library and its components
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Importing the ChatGroq class from langchain_groq
from langchain_groq import ChatGroq

from src.db_configuration.pinecone_db import TextPreprocessor, PineconeConfig


# Creating a Flask application instance
app = Flask(__name__)

# Call the function to load environment variables from a .env file
load_dotenv()

# Set the Flask secret key for session management
app.config['SECRET_KEY'] = os.getenv('APP_SEC_KEY')

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
groq_api_key = os.getenv('GROQ_API_KEY')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Retrieve the user inputs from the form
            index_name = request.form.get('idx_name')
            index_name = index_name.lower()
            llm_model = request.form.get('option')
            embedding_model = request.form.get('embedding')

        except Exception as e:
            print(e, type(e))

        # Create or get Pinecone index
        pinecone_obj = PineconeConfig(index_name=index_name)
        pinecone_obj.create_index()

        # Store the index name, embeddings & llm model name in the session
        session['index_name'] = index_name
        session['llm_model'] = llm_model
        session['embedding_model'] = embedding_model

        # Clear the messages from the flash
        get_flashed_messages()

        # Check if files were uploaded
        
        files = request.files.getlist('file')

        TextPreprocess_obj = TextPreprocessor()

        # Extract text from files and create chunks
        extracted_text = TextPreprocess_obj.get_text(files)
        text_chunks = TextPreprocess_obj.get_text_chunks(extracted_text)

        # Inject the data into the Pinecone index
        pinecone_obj.inject_data(text_chunks=text_chunks)

        # Add a short delay to ensure indexing is complete before redirecting
        time.sleep(1)

        # Redirect to the next page
        return redirect(url_for('irs_llm_page'))

    # Handle GET request: Render the template for the form
    return render_template('irs_page.html')


# Define a route for the 'IRS_LLM_Page' URL endpoint
@app.route('/IRS_LLM_Page', methods=["GET", "POST"])
def irs_llm_page():

    # return render_template('irs_page_chat.html')

    # Retrieve the LLM model name and embedding model names from the session
    llm_model_name = session.get("llm_model")

    if request.method == "POST":

        if llm_model_name in ['gemini-1.5-flash-latest']:

            llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest')

            # Retrieve the query message from the form
            query = request.form['msg']

            # Retrieve the conversation history from the session or initialize it 
            conversation_history = session.get('conversation_history', [])

            # retrieve the index name from the session
            index_name = session.get('index_name')

            pinecone_obj = PineconeConfig(index_name=index_name)
            prompt = PineconeConfig.cus_prompt()

            chain = pinecone_obj.get_chain(llm=llm, prompt=prompt)

            # Invoke the chain with the query and conversation history
            full_query = "\n".join(conversation_history) + "\nUser: " + query

            # Invoke the chain with the query and return the result as JSON
            res = chain.invoke(full_query)

            # Update the conversation history
            conversation_history.append(f"User: {query}")
            conversation_history.append(f"{res}")

            # Save the updated history to the session
            session['conversation_history'] = conversation_history

            return jsonify({'response': res})
        
    
        # Handle the 'Llama3' LLM model case
        elif llm_model_name in ['llama-3.1-8b-instant']:

            # Initializing GROQ chat with provided API key, model name, and settings
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model=llm_model_name,
                temperature=0.2
            )

            # Retrieve the query message from the form
            query = request.form['msg']

            # retrieve the index name from the session
            index_name = session.get('index_name')

            pinecone_obj = PineconeConfig(index_name=index_name)
            prompt = PineconeConfig.cus_prompt()

            chain = pinecone_obj.get_chain(llm=llm, prompt=prompt)

            # Invoke the chain with the query and return the result as JSON
            res = chain.invoke(query)

            return jsonify({'response': res})

        
        # Handle 'gemma' LLM model case
        elif llm_model_name in ['gemma2-9b-it']:

            # Initializing GROQ chat with provided API key, model name, and settings
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model=llm_model_name,
                temperature=0.2
            )

            # Retrieve the query message from the form
            query = request.form['msg']

            # retrieve the index name from the session
            index_name = session.get('index_name')

            pinecone_obj = PineconeConfig(index_name=index_name)
            prompt = PineconeConfig.cus_prompt()

            chain = pinecone_obj.get_chain(llm=llm, prompt=prompt)

            # Invoke the chain with the query and return the result as JSON
            res = chain.invoke(query)

            return jsonify({'response': res})
   
    return render_template('irs_page_chat.html')



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
