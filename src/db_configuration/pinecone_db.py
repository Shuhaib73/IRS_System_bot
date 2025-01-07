import os
import io
from dotenv import load_dotenv
import time 
from pypdf import PdfReader

import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# from langchain_community.document_loaders import UnstructuredURLLoader

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)


class TextPreprocessor:
    """
    Class for handling text extraction and preprocessing.
    """
    
    def get_text(self, pdf_docs):
        '''Process the files and return the content as a single string, handling encoding issues.'''

        text = ""

        for pdf in pdf_docs:
            # Ensure that pdf is a file-like object
            if hasattr(pdf, 'read'):
                try:
                    # Read file-like object into BytesIO
                    pdf_stream = io.BytesIO(pdf.read())
                    pdf_reader = PdfReader(pdf_stream)
                    
                    # Extract text from each page
                    for page in pdf_reader.pages:
                        page_text = page.extract_text() or ""  # Extract text or default to empty string
                        
                        # Handle potential UnicodeEncodeError by cleaning text
                        text += ''.join(c for c in page_text if ord(c) < 128)  # Only keep ASCII characters

                except Exception as e:
                    print(f"Error processing file: {e}")
                    
        return text
        
    def get_text_chunks(self, text):
        """
        Splits the extracted text into smaller chunks for further processing.
        
        :param text: The extracted text from the PDF.
        :return: A list of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )

        chunks = text_splitter.split_text(text)

        return chunks



class PineconeConfig:
    def __init__(self, index_name):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.embedding_model = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=self.google_api_key)
        

    def create_index(self):
        """
        Creates a Pinecone index if it doesn't exist.
        """
        if self.index_name in self.pc.list_indexes().names():
            message = f"{self.index_name}, Index already exists. Here is the Index Description: {self.pc.Index(self.index_name).describe_index_stats()}"
            return message 
        
        else:
            # Create index if not exists
            self.pc.create_index(
            name=self.index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
            
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(2)
        message = f"Index Created Successfully! Here is the index description:\n {self.pc.Index(self.index_name).describe_index_stats()}"

        return message
    

    def inject_data(self, text_chunks):
        """
        Creates and returns a Pinecone vector store for storing text chunks and their embeddings.
        
        :param text_chunks: List of text chunks to store in Pinecone.
        :param index_name: The name of the Pinecone index where the vectors will be stored.
        :return: A Pinecone vector store.
        """

        # Create and return the vector store
        vector_store = PineconeVectorStore.from_texts(
            text_chunks, 
            embedding=self.embedding_model, 
            index_name=self.index_name
        )
        
        return vector_store
    

    @staticmethod
    def cus_prompt():
            
        template = """
        <|system|>>
        You are an AI Assistant that follows instructions extremely well.
        Please be truthful and answer the question as detailed as possible from the provided context and history. 
        Make sure to:
        - Structure your response in **bullet points** or **numbered lists** for clarity.
        - Use new lines for each point to keep the response organized.
        - Add **examples or analogies** if appropriate.

        For questions about methods, processes, or detailed concepts, respond step by step. Use formatting like:
        1. **Step 1**: Explanation.
        2. **Step 2**: Explanation.
        
        If the question includes greetings (e.g., "Hi", "Hello", "Thank you"), reply warmly and include an appropriate emoji to convey friendliness. Avoid emojis for technical answers.

        CONTEXT: {context}
        </s>
        <|user|>
        {query}
        </s>
        <|assistant|>
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt


    

    def get_chain(self, llm, prompt):
        
        output_parser = StrOutputParser()
        vector_store = PineconeVectorStore(
            index_name=self.index_name, 
            embedding=self.embedding_model, 
            pinecone_api_key=self.pinecone_api_key
        )

        retriever_vectorstore = vector_store.as_retriever(search_kwargs={"k": 2})
        
        chain = (
            {"context": retriever_vectorstore, "query": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )

        return chain
    
    def get_hybrid_chain(self, llm):

        prompt = PineconeConfig.cus_prompt()
        output_parser = StrOutputParser()
        vector_store = PineconeVectorStore(
            index_name=self.index_name, 
            embedding=self.embedding_model, 
            pinecone_api_key=self.pinecone_api_key
        )

        retriever_vectorstore = vector_store.as_retriever(search_kwargs={"k": 2})
        
        chain = (
            {"context": retriever_vectorstore, "query": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )

        return chain
    

    
