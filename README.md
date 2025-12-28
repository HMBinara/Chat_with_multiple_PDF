# Chat with Multiple PDFs 📚

🚀 An AI-powered LLM application that allows users to chat with multiple PDF documents simultaneously using LangChain, Python, and Streamlit.

## Features

- 📄 **Upload Multiple PDFs**: Upload and process multiple PDF documents at once
- 💬 **Interactive Chat**: Ask questions about your documents in natural language
- 🧠 **Smart Retrieval**: Uses FAISS vector store for efficient document search
- 💭 **Conversational Memory**: Maintains context throughout the conversation
- 📊 **Source Citations**: See which parts of the documents were used to answer your questions
- ⚡ **Real-time Processing**: Fast text extraction and chunking

## Technologies Used

- **Streamlit**: Web interface and user interaction
- **LangChain**: Document processing and conversational AI
- **OpenAI**: GPT-3.5-turbo for chat and embeddings
- **FAISS**: Vector store for semantic search
- **PyPDF2**: PDF text extraction
- **Python-dotenv**: Environment variable management

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/HMBinara/Chat_with_multiple_PDF.git
cd Chat_with_multiple_PDF
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. **Run the application**
```bash
streamlit run app.py
```

2. **Upload PDFs**
   - Click on the "Browse files" button in the sidebar
   - Select one or more PDF files
   - Click "Process PDFs" button

3. **Ask Questions**
   - Wait for the processing to complete
   - Type your question in the text input field
   - View the AI-generated response based on your documents

4. **View Sources**
   - Expand "Source Documents" to see relevant excerpts
   - Clear history using the "Clear History" button if needed

## How It Works

1. **Text Extraction**: Extracts text from uploaded PDF files using PyPDF2
2. **Text Chunking**: Splits text into manageable chunks (1000 characters with 200 overlap)
3. **Embeddings**: Converts chunks into vector embeddings using OpenAI
4. **Vector Store**: Stores embeddings in FAISS for fast similarity search
5. **RAG Pattern**: Uses Retrieval Augmented Generation with conversational memory
   - Retrieves relevant document chunks based on user questions
   - Maintains chat history for context-aware responses
   - Generates answers using GPT-3.5-turbo with retrieved context
6. **Response Generation**: Combines retrieved context with chat history for contextual answers

## Configuration

You can customize the following parameters in `app.py`:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `temperature`: LLM creativity level (default: 0.7)
- `model_name`: OpenAI model (default: "gpt-3.5-turbo")
- `k`: Number of relevant chunks to retrieve (default: 3)

## Requirements

- Python 3.8 or higher
- OpenAI API key
- Internet connection

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

**Error: OPENAI_API_KEY not found**
- Make sure you've created a `.env` file with your API key

**No text extracted from PDFs**
- Ensure PDFs contain actual text (not just images)
- Try with a different PDF file

**Memory errors with large PDFs**
- Process fewer PDFs at once
- Reduce chunk_size parameter

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI for providing the GPT models and embeddings
- LangChain for the excellent framework
- Streamlit for the intuitive web framework
