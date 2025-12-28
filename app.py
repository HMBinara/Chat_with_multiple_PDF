import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create conversational retrieval chain with memory."""
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain

def handle_user_input(user_question):
    """Handle user questions and display chat history."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF files first!")
        return
    
    with st.spinner("Thinking..."):
        response = st.session_state.conversation({'question': user_question})
    
    st.session_state.chat_history = response['chat_history']
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User message
            st.write(f"**🧑 You:** {message.content}")
        else:  # Bot message
            st.write(f"**🤖 Assistant:** {message.content}")
    
    # Display source documents if available
    if 'source_documents' in response and response['source_documents']:
        with st.expander("📚 Source Documents"):
            for i, doc in enumerate(response['source_documents']):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content[:300] + "...")
                st.divider()

def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Header
    st.title("📚 Chat with Multiple PDFs")
    st.markdown("Upload your PDF documents and ask questions about their content!")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        st.markdown("---")
        
        pdf_docs = st.file_uploader(
            "Upload your PDF files here",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to chat with"
        )
        
        if st.button("Process PDFs", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
            else:
                with st.spinner("Processing your documents..."):
                    try:
                        # Get PDF text
                        st.info("📖 Extracting text from PDFs...")
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("No text could be extracted from the PDFs!")
                            return
                        
                        # Get text chunks
                        st.info("✂️ Splitting text into chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        st.success(f"Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        st.info("🧠 Creating vector embeddings...")
                        vectorstore = get_vectorstore(text_chunks)
                        
                        # Create conversation chain
                        st.info("💬 Setting up conversation chain...")
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        
                        st.success("✅ PDFs processed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Upload one or more PDF files
        2. Click 'Process PDFs' button
        3. Wait for processing to complete
        4. Ask questions in the main area
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app uses:
        - **LangChain** for document processing
        - **OpenAI** for embeddings and chat
        - **FAISS** for vector storage
        - **Streamlit** for the interface
        """)
    
    # Main chat area
    st.markdown("### 💬 Chat Interface")
    
    user_question = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is the main topic of these documents?",
        key="user_input"
    )
    
    if user_question:
        handle_user_input(user_question)
    
    # Display chat history if exists
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Conversation History")
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.experimental_rerun()

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found! Please set it in your .env file.")
        st.info("Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        st.stop()
    
    main()
