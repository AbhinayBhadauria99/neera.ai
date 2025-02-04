import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Load general responses from CSV
responses_df = pd.read_csv('soft-talk.csv')
general_responses = pd.Series(responses_df.response.values, index=responses_df.question).to_dict()

# Specify PDF files to read
pdf_files = ["GuideLinesIAndDandSTP.pdf", "NamamiGangeprogramme.pdf"]  # Replace with your actual PDF file names

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

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    st.session_state.vector_store = vector_store

def load_vector_store():
    if "vector_store" not in st.session_state:
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
    return st.session_state.vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say, "answer is not available in the context." Do not provide incorrect answers.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def handle_user_input(user_question):
    # Check for general expressions first
    lower_question = user_question.lower().strip()
    if lower_question in general_responses:
        st.write("Reply:", general_responses[lower_question])
    else:
        # Load the vector store and process the question with embeddings
        vector_store = load_vector_store()
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Neera:", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Neera 1.0 : Insights on Namami Gange 🌊")

    # Process PDFs only once at startup and then show the initial greeting
    if "vector_store" not in st.session_state:
        with st.spinner("Wait! we are getting things ready for you..."):
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Good to go. You can start asking your doubts!")
            st.session_state.greeted = False  # Reset greeting after processing

    # Send the initial greeting message after processing PDFs
    if "greeted" not in st.session_state or not st.session_state.greeted:
        st.session_state.greeted = True
        st.write("Hey, I am Neera 1.0, how can I help you?")

    # User question input field with auto-clear mechanism
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""  # Initialize session state for user question

    # Display input box and get user input
    user_question = st.text_input("Ask a question related to water conservation and namami gange scheme:", 
                                  key="user_question_input", value=st.session_state.user_question)
    
    # Button to submit question and clear input field
    if st.button("Submit"):
        if user_question:
            handle_user_input(user_question)
            # Clear the input field by setting session state
            st.session_state.user_question = ""  # Update session state to clear input for the next render

if __name__ == "__main__":
    main()
