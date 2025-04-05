# C:\Users\MANIKANTA\Desktop\R\RAG\debate\api\main.py
from fastapi import FastAPI
from api.pdf_summary import router as pdf_summary_router

app = FastAPI()

app.include_router(pdf_summary_router)
