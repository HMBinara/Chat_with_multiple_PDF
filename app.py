import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

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

def get_conversation_response(vectorstore, question, chat_history):
    """Get response using RAG pattern with conversational memory."""
    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)
    
    # Format context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Format chat history
    history_text = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    
    # Create prompt with context and history
    prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise and informative.

Chat History:
{history_text}

Context:
{context}

Question: {question}

Answer:"""
    
    # Get response from LLM
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    
    return response.content, relevant_docs

def handle_user_input(user_question):
    """Handle user questions and display chat history."""
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process PDF files first!")
        return
    
    with st.spinner("Thinking..."):
        # Get response with relevant documents
        answer, source_docs = get_conversation_response(
            st.session_state.vectorstore, 
            user_question, 
            st.session_state.messages
        )
    
    # Add messages to history
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Display all messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"**🧑 You:** {message['content']}")
        else:
            st.write(f"**🤖 Assistant:** {message['content']}")
    
    # Display source documents
    if source_docs:
        with st.expander("📚 Source Documents"):
            for i, doc in enumerate(source_docs):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
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
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
                        st.session_state.vectorstore = get_vectorstore(text_chunks)
                        
                        # Clear previous messages
                        st.session_state.messages = []
                        
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
    if st.session_state.messages:
        st.markdown("---")
        st.markdown("### 📜 Conversation History")
        if st.button("Clear History"):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.rerun()

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found! Please set it in your .env file.")
        st.info("Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        st.stop()
    
    main()
