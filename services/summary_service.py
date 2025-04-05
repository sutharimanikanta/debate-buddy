# C:\Users\MANIKANTA\Desktop\R\RAG\debate\services\summary_service.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import io
import os

# Define the persistent directory for Chroma
PERSIST_DIRECTORY = "chroma_db"  # Same directory as in main.py


def process_pdf(pdf_file: io.BytesIO, google_api_key: str):
    """
    Load a PDF file, split it into semantic chunks, and return the chunks.
    """
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

    # Load the PDF
    loader = PyPDFLoader("temp.pdf")
    doc = loader.load()

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        api_key=google_api_key, model="models/embedding-001"
    )

    # Split the document into semantic chunks
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="gradient",
        breakpoint_threshold_amount=0.8,
    )
    chunks = splitter.split_documents(doc)

    return chunks


def generate_debate(topic: str, retriever, groq_api_key: str) -> str:
    """
    Generate a structured debate using the provided topic and retrieved context.
    """
    # Retrieve relevant context from the vector store using the retriever
    context = retriever.invoke(topic)

    # Create the debate prompt
    prompt = ChatPromptTemplate.from_template(
        """You are moderating a structured debate between two experts. 
        The topic of discussion is: **{topic}**

        Context: {context}

        - **Expert 1 (For the topic)**: Provide arguments supporting the topic.
        - **Expert 2 (Against the topic)**: Provide counterarguments against the topic.
        - After both sides present their arguments, summarize the key points made by each.
        
        Keep the debate formal, logical, and engaging.
        
        Now, let's begin the debate on: **{topic}**"""
    )

    # Initialize the language model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mistral-saba-24b")

    # Create the chain
    chain = RunnablePassthrough() | prompt | llm | StrOutputParser()

    # Invoke the chain
    response = chain.invoke(
        {"topic": topic, "context": "\n".join([doc.page_content for doc in context])}
    )

    return response


def process_pdf_and_create_vector_store(pdf_file, google_api_key):
    """
    Process a PDF file, create a Chroma vector store, and persist it to disk.
    """
    # Process the PDF into semantic chunks
    chunks = process_pdf(pdf_file, google_api_key)

    # Initialize Chroma with a persistent directory
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(
            api_key=google_api_key, model="models/embedding-001"
        ),
        persist_directory=PERSIST_DIRECTORY,  # Persistent database
    )

    # Persist the database to disk
    vector_store.persist()

    return vector_store
