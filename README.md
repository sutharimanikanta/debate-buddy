🧠 AI-Powered Debate Generator
An interactive Streamlit web application that generates intelligent, context-aware debates from uploaded PDF documents using retrieval-augmented generation (RAG) techniques. Powered by LangChain, Google Generative AI, and GROQ APIs, this tool is designed to transform your documents into dynamic AI-generated debates on any topic you provide.

🚀 Features
📄 PDF Upload & Parsing: Upload academic papers, articles, or reports in PDF format.

🔍 Semantic Chunking: Intelligently splits documents into context-preserving chunks.

🤖 Vector Store Creation: Stores embeddings in a persistent ChromaDB database.

🧠 RAG-Based Generation: Retrieves relevant chunks and generates debates with GROQ LLM.

🎙️ Custom Topic Input: Enter any debate topic based on the document’s context.

💾 Session Management: Persistent vector store across user sessions.

🏗️ Project Structure
bash
Copy
Edit
.
├── main.py                           # Streamlit app entry point
├── .env                              # Environment variables (API keys)
├── requirements.txt                  # Python dependencies
├── services/
│   ├── summary_service.py            # Handles PDF processing and debate generation
├── chroma_db/                        # Persistent vector database (auto-created)
📥 Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/debate-generator.git
cd debate-generator
Create a virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables

Create a .env file with your API keys:

ini
Copy
Edit
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
🧪 Running the App
To start the Streamlit app:

bash
Copy
Edit
streamlit run main.py
🧰 How It Works
Upload a PDF file.

The app splits the PDF into semantic chunks using SemanticChunker.

GoogleGenerativeAIEmbeddings converts chunks into vectors.

ChromaDB persists the vectors in a local vector store (chroma_db/).

Enter any debate topic.

The app retrieves top-matching document chunks and uses GROQ API to generate a structured debate response.

📚 Dependencies
streamlit

langchain

langchain-google-genai

chromadb

python-dotenv

🧠 Sample Use Cases
Academic debates based on research papers

Legal document analysis and argument generation

Corporate reports → pros & cons discussions

Classroom tools for debate preparation

✅ To-Do
 Voice-enabled debate output

 Export debates to PDF

 Integration with OpenAI as fallback

 User authentication and file history
