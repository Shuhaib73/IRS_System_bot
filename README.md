## Retrieval-Augmented Generation (RAG) System: Powered by Langchain 🤖

"Welcome to RAG & IRS (Information Retrieval System): AI-Powered Platform for Dynamic Information Retrieval and Query Response 🧑‍💻"

### **Access the Web Application**: http://18.208.146.50:8080/

---
## Demo 

![Demo](https://github.com/Shuhaib73/IRS_System_bot/blob/main/static/architectures/demo_rag.gif)

----

### 📖 **Features**

✅ AI-Powered Query Responses: Leverage LLMs (Large Language Models) and vector databases, which store word embeddings, to understand and respond to user queries with precision. By performing semantic similarity searches on these embeddings, the system provides context-aware responses tailored to the user's needs. 🤖

✅ Real-Time Information Retrieval: Instantly retrieve and present the most relevant data, documents, or knowledge from the database based on user inputs. The system efficiently handles queries over multiple PDFs to deliver quick, reliable responses. ⏱️

✅ Customizable Knowledge Bases: Tailor information retrieval models to suit the specific needs of your industry, data sources, or organizational structure. Adapt the system to provide more relevant and precise insights based on the unique content of your documents. ⚙️

✅ Multiple PDF Handling: Effortlessly process and search through multiple PDFs to provide comprehensive, context-aware responses from a wide range of document sources. Users can upload multiple files and query them simultaneously, making document-based searches efficient and scalable. 📄📄


---
## <br>**➲ RAG Architecture** :

<img src="https://github.com/Shuhaib73/IRS_System_bot/blob/main/static/architectures/RAG_Google.png" alt="RAG Pipeline" style="width: 750px; height: 350px; border: 2px solid #ccc; border-radius: 8px; display: inline-block; margin-right: 10px;">

----
## 🛠️ **Technologies Used**

- **Python** 🐍: The core programming language that powers the app.  
- **Flask**: A Backend web framework for building web applications.
- **HTML & CSS**: The markup language used to structure the content and layout of the web page and CSS styles the HTML content, controlling the appearance, such as colors, fonts, and layouts..
- **Pinecone**: A vector database for performing semantic similarity searches, enabling fast and relevant responses based on the query's context and the indexed documents.
- **LangChain** 🔗:  An open-source framework for developing applications powered by language models, used for building a robust query-response system within IRS, supporting document-based questions.
- **GoogleGenerativeAIEmbeddings**: Leveraging Google’s generative AI embeddings to create semantic vectors of text, improving the accuracy of search and response generation.
- **Ensemble Retriever**: A powerful component for combining multiple retrieval methods, improving the accuracy and relevance of search results and responses.
- **Groq**:  High-performance hardware for accelerating machine learning workloads, optimizing retrieval and response time for large data.
- **LLM Models:**
    - Gemini-1.5-Flash-Latest: A cutting-edge LLM model optimized for language understanding and generation tasks, enhancing query comprehension and generating accurate responses.
    - LLAMA-3.1-8B-Instant: A versatile language model supporting a broad range of NLP tasks, improving contextual understanding and generating relevant responses quickly.
    - Gemma2-9B-IT: A powerful generative model designed for information retrieval, improving semantic matching between queries and data, tailored for high-accuracy results.
----


-----

## 🌟 Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Shuhaib73/IRS_System_bot/tree/main
   
   ```

2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

3. Add .env file
   ```bash
    GOOGLE_API_KEY = "AIzajdafl-------------------"
    PINECONE_API_KEY = "pcsk------------"
    APP_SEC_KEY = "-----"
    GROQ_API_KEY = "gsk-----------"
   ```


3. Run the app:
   ```bash
       python app.py
   ```

4. Access the app by opening a web browser and navigating to the provided URL.
    - By default, Flask will run on http://localhost:5000.

7. Upload the PDF documents you want to analyze.

8. Click the "Upload Documents" button to process the documents and generate vector embeddings.

9. Engage in interactive conversations with the documents by typing your questions in the chat input box.



----

## 📧 **Contact**

For questions, feedback, or contributions, please contact:  **Shuhaib**  
**Email**: mohamed.shuhaib73@gmail.com
**LinkedIn**: https://www.linkedin.com/in/mohamedshuhaib/

---

