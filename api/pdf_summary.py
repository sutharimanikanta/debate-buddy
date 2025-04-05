# C:\Users\MANIKANTA\Desktop\R\RAG\debate\api\pdf_summary.py
from fastapi import APIRouter, File, UploadFile
from services.conversations import Conversations
from services.summary_service import process_pdf, generate_debate
from langchain_community.vectorstores import Chroma
import os

router = APIRouter()


@router.post("/upload_pdf")
async def upload(pdf_file: UploadFile = File(...)):
    # Process the uploaded PDF
    google_api_key = os.getenv(
        "GOOGLE_API_KEY"
    )  # Fetch Google API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")  # Fetch Groq API key from environment

    if not google_api_key or not groq_api_key:
        return {
            "error": "Missing GOOGLE_API_KEY or GROQ_API_KEY in environment variables"
        }

    # Process the PDF into semantic chunks
    chunks = process_pdf(pdf_file.file, google_api_key)

    # Store the chunks in a Chroma vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(
            api_key=google_api_key, model="models/embedding-001"
        ),
    )

    # Create a retriever for the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # Generate a debate on a default topic (e.g., "The impact of AI on jobs")
    debate_topic = "The impact of AI on jobs"
    debate_response = generate_debate(debate_topic, retriever, groq_api_key)

    # Return the debate response
    return {"debate": debate_response}
