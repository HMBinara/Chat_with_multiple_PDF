# Chat_with_multiple_PDF
🚀 An AI-powered LLM application that allows users to chat with multiple PDF documents simultaneously using LangChain, Python, and Streamlit.

## How to Run

### Prerequisites
- Python 3.8 or higher

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Chat_with_multiple_PDF
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the root directory.
2. Add your API keys (if using OpenAI or HuggingFace Hub):
   ```env
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
   ```

### Usage

Run the application using Streamlit:
```bash
streamlit run app.py
```
