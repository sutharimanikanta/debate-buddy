# C:\Users\MANIKANTA\Desktop\R\RAG\debate\main.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from services.summary_service import generate_debate, process_pdf
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Streamlit App Title
st.title("AI-Powered Debate Generator")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.error("API keys are missing. Please check your .env file.")
else:
    # File Uploader for PDF
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Specify a persistent directory for Chroma
    PERSIST_DIRECTORY = "chroma_db"

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if pdf_file is not None:
        if st.session_state.vector_store is None:
            # Process the uploaded PDF
            with st.spinner("Processing PDF..."):
                try:
                    chunks = process_pdf(pdf_file, GOOGLE_API_KEY)

                    # Initialize Chroma with a persistent directory
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=GoogleGenerativeAIEmbeddings(
                            api_key=GOOGLE_API_KEY, model="models/embedding-001"
                        ),
                        persist_directory=PERSIST_DIRECTORY,  # Persistent database
                    )

                    # Persist the database to disk
                    st.session_state.vector_store.persist()

                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    # Input for Debate Topic
    topic = st.text_input("Enter the debate topic")

    # Button to Generate Debate
    if st.button("Generate Debate") and topic:
        if st.session_state.vector_store is None:
            st.error("Please upload and process a PDF first.")
        else:
            with st.spinner("Generating debate..."):
                try:
                    # Initialize the retriever
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": 2}
                    )

                    # Generate the debate
                    debate_response = generate_debate(topic, retriever, GROQ_API_KEY)

                    # Display the debate
                    st.markdown(debate_response)
                except Exception as e:
                    st.error(f"Error generating debate: {str(e)}")  # import io
# from typing import Optional
# Cd “debate”
# "C:\Users\MANIKANTA\Desktop\R\RAG\intern\venv\Scripts\Activate"
# Streamlit run main.py

# import requests
# import streamlit as st
# from services.conversations import Conversations
# from services.summary_service import continue_conversation, set_openai_api_key
# from streamlit_chat import message as chat_message


# @st.cache_resource
# def handle_pdf_upload(pdf_file: io.BytesIO) -> Optional[Conversations]:
#     if pdf_file is not None:
#         files = {"pdf_file": pdf_file.getvalue()}
#         response = requests.post("http://localhost:8001/upload_pdf/", files=files)
#         response.raise_for_status()

#         messages = response.json()["conversations"]["messages"]
#         conversations = Conversations()

#         for m in messages:
#             conversations.add_message(m["role"], m["content"])

#         return conversations
#     return None


# def main():
#     st.title("PDF Summarizer")

#     API_KEY = st.text_input(
#         "Type your OPENAI_API_KEY here",
#         type="password",
#         key="api_key",
#         help="You can get api_key at https://platform.openai.com/account/api-keys",
#     )
#     set_openai_api_key(API_KEY)

#     if "conversations" not in st.session_state:
#         st.session_state.conversations = Conversations()

#     if "uploaded" not in st.session_state:
#         st.session_state.uploaded = False

#     pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

#     if pdf_file is not None and st.session_state.uploaded is False:
#         print("handle_pdf_upload")
#         conversations = handle_pdf_upload(pdf_file)
#         st.session_state.uploaded = True
#         st.session_state.conversations = conversations

#     question = st.text_input("Type your question here")

#     if st.button("Ask", key="ask_button"):
#         if question:
#             print("continue_conversation")
#             st.session_state.conversations = continue_conversation(
#                 st.session_state.conversations, question
#             )

#     if st.button("Clear All cache", key="clear_cache"):
#         st.cache_resource.clear()
#         st.session_state.conversations = Conversations()

#     if "conversations" in st.session_state:
#         for i, message in enumerate(
#             reversed(st.session_state.conversations.get_messages())
#         ):
#             if message.role == "assistant":
#                 chat_message(message.content, key=str(i))
#             else:
#                 chat_message(message.content, key=str(i) + "_user", is_user=True)


# if __name__ == "__main__":
#     main()
