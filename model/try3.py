import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
import faiss  # Import the FAISS library

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Specify PDF files to read
pdf_files = ["GuideLinesIAndDandSTP.pdf", "NamamiGangeprogramme.pdf"]

# Function to read and process PDF content
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_vector_store():
    # Read PDF and create vector store
    raw_text = get_pdf_text(pdf_files)
    if not raw_text:  # Check if raw_text is empty
        raise ValueError("No text extracted from PDF files. Please check your PDFs.")
    
    text_chunks = get_text_chunks(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Check if the vector store has been populated
    if vector_store.index.ntotal == 0:
        raise ValueError("FAISS index is empty. Ensure the text chunks are being created correctly.")

    # Save the FAISS index to a file
    index_file = "faiss_index.index"
    faiss.write_index(vector_store.index, index_file)  # Save the FAISS index
    print(f"FAISS index saved to {index_file}")

if __name__ == "__main__":
    create_vector_store()





### Run in from notebook