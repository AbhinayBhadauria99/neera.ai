from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS  # Updated import for FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import faiss
import os
import numpy as np

app = FastAPI()

# Load the FAISS index
def load_vector_store():
    index_file = "faiss_index.index"
    if not os.path.exists(index_file):
        raise FileNotFoundError("FAISS index file not found.")
    index = faiss.read_index(index_file)
    return index

vector_store = load_vector_store()

# Setup the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say, "Answer is not available in the context." Do not provide incorrect answers.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Namami Gange QA API!"}

@app.post("/ask/")
async def ask_question(question: str):
    try:
        # Convert the question into a suitable format for searching
        # (You may need to implement a function that converts the question into an embedding if needed)
        question_embedding = np.array([question])  # Placeholder: adjust this based on your actual vector format
        D, I = vector_store.search(question_embedding, k=5)  # Retrieve top 5 documents

        # Create a conversational chain
        chain = get_conversational_chain()
        
        # Assuming I contains the indices of the retrieved documents; 
        # you may need to implement a function to convert these indices to actual document contents
        context_docs = [str(doc) for doc in I]  # Placeholder for actual document retrieval
        
        # Generate a response
        response = chain({"input_documents": context_docs, "question": question}, return_only_outputs=True)
        return JSONResponse(content={"answer": response["output_text"]})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
