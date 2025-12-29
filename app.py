import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Using Gemini Embeddings instead of OpenAI or HuggingFace
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📄", layout="wide")

    st.header("Chat with Multiple PDFs 📄 ")
    st.text_input("Ask a questions about your PDF files!")
    
    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # 1. get the PDFs text
                raw_text = get_pdf_text(pdf_docs)
                
                # 2. get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # 3. create vector store using Gemini
                vectorstore = get_vectorstore(text_chunks)
                st.write("Vector Store Created using Gemini Embeddings!")

if __name__ == '__main__':
    main()