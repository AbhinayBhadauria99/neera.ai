from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

app = FastAPI()

# Define a Pydantic model for the request body
class QuestionRequest(BaseModel):
    question: str

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
async def ask_question(question_request: QuestionRequest):
    question = question_request.question  # Get the question from the request body
    try:
        # Retrieve relevant documents from the FAISS index
        D, I = vector_store.search(np.array([question]), k=5)  # Retrieve top 5 documents
        # Create a conversational chain
        chain = get_conversational_chain()
        response = chain({"input_documents": [str(doc) for doc in I], "question": question}, return_only_outputs=True)
        return JSONResponse(content={"answer": response["output_text"]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)











###http://127.0.0.1:8000/