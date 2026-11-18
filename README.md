# InsightIQ — AI Research Assistant (LLM + RAG)

InsightIQ is a portfolio-ready PDF question-answering application. Upload PDFs, retrieve relevant passages, and ask a Groq-hosted Llama model questions. Answers are instructed to stay within the retrieved evidence and cite source filenames and page numbers.

## Features
- Multiple PDF uploads
- Text extraction and overlapping chunking
- TF-IDF retrieval to find relevant passages
- Llama 3.3 70B generation through Groq
- Evidence panel with source filename and page number
- Adjustable chunk size, overlap, and retrieval count
- Streamlit dashboard with session-based chat history

## Architecture
PDF → PyPDF extraction → text chunks → TF-IDF retrieval → relevant evidence → Groq LLM → cited answer

## Requirements
- Python 3.10–3.12 recommended
- Groq API key (free account availability/limits may change)
- Internet connection for LLM requests

## Run on Windows (VS Code / PowerShell)

1. Extract the ZIP and open the `InsightIQ_LLM_Project` folder in VS Code.
2. Open Terminal → New Terminal.
3. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this once in that terminal:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   Then activate again.

4. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Create a Groq API key at https://console.groq.com/keys.
6. Copy `.env.example` and rename the copy to `.env`. Put your key after `GROQ_API_KEY=`.
   Never upload `.env` to GitHub.
7. Start the app:

   ```powershell
   streamlit run app.py
   ```

8. Browser opens at `http://localhost:8501`. Upload a text-based PDF in the sidebar, click **Build knowledge base**, ask a question, then click **Generate answer**.

## GitHub upload

1. Create a new public repository named `insightiq-llm-rag-assistant`.
2. In VS Code open the project folder.
3. Make sure `.env` is not present in the files staged for upload. `.gitignore` excludes it.
4. Run:

   ```bash
   git init
   git add .
   git commit -m "Build InsightIQ RAG PDF assistant"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/insightiq-llm-rag-assistant.git
   git push -u origin main
   ```

5. Replace `YOUR-USERNAME` with your GitHub username.
6. Add a screenshot of the running app to a `screenshots/` folder and embed it here:

   `![InsightIQ dashboard](screenshots/dashboard.png)`

## Deploy (optional)
You can deploy on Streamlit Community Cloud. Connect the GitHub repository, set the main file to `app.py`, and add `GROQ_API_KEY` under App settings → Secrets. Do not commit API keys.

## Interview explanation
"I built InsightIQ, a Retrieval-Augmented Generation application that lets users query their own PDF documents. I implemented PDF parsing, overlapping text chunking, TF-IDF-based retrieval, and LLM response generation through Groq. The system passes only the most relevant passages to the model and asks it to cite source pages, reducing unsupported answers. I also built a Streamlit interface with configurable retrieval settings."

## Important limitations
- TF-IDF is a lightweight lexical retrieval baseline, not semantic vector search. A future improvement is sentence-transformer embeddings + FAISS/Chroma.
- Scanned/image-only PDFs need OCR before their text can be searched.
- LLM answers can still be wrong; verify critical information in the source document.
