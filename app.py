import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
st.set_page_config(page_title="InsightIQ | AI Research Assistant", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0b1220 0%,#111c32 55%,#17243b 100%); color:#eef4ff;}
[data-testid="stSidebar"] {background:#101a2b;}
.block-container {padding-top:2rem; max-width:1200px;}
.hero {padding:24px 28px;border:1px solid #2c4264;border-radius:18px;background:linear-gradient(120deg,#172a46,#172238);margin-bottom:20px;}
.hero h1 {font-size:2.3rem;margin:0;color:#f4f8ff;}
.hero p {color:#b8c9e3;margin:8px 0 0 0;}
.panel {padding:16px;border:1px solid #2c4264;border-radius:14px;background:#142239;}
.small {color:#9fb4d2;font-size:.9rem;}
.stButton>button {border-radius:10px;background:#3478f6;color:white;border:0;font-weight:600;}
</style>
""", unsafe_allow_html=True)

if "chunks" not in st.session_state: st.session_state.chunks = []
if "source_names" not in st.session_state: st.session_state.source_names = []
if "history" not in st.session_state: st.session_state.history = []

def extract_chunks(uploaded_files, chunk_words=180, overlap=35):
    chunks, names = [], []
    for file in uploaded_files:
        reader = PdfReader(file)
        pages = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            words = text.split()
            for start in range(0, len(words), chunk_words-overlap):
                piece = " ".join(words[start:start+chunk_words])
                if piece.strip():
                    chunks.append({"text":piece, "source":file.name, "page":page_no})
        names.append(file.name)
    return chunks, names


def retrieve(question, chunks, top_k=4):
    if not chunks:
        return []

    corpus = [c["text"] for c in chunks]

    # Word-based retrieval
    try:
        word_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        word_matrix = word_vectorizer.fit_transform(
            corpus + [question]
        )
        word_scores = cosine_similarity(
            word_matrix[-1], word_matrix[:-1]
        ).flatten()
    except ValueError:
        word_scores = [0.0] * len(corpus)

    # Character-based retrieval
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )
    char_matrix = char_vectorizer.fit_transform(
        corpus + [question]
    )
    char_scores = cosine_similarity(
        char_matrix[-1], char_matrix[:-1]
    ).flatten()

    # Combine both retrieval scores
    scores = (0.7 * word_scores) + (0.3 * char_scores)

    best_indices = scores.argsort()[::-1][:top_k]

    results = [
        chunks[i] for i in best_indices
        if scores[i] > 0.01
    ]

    return results


def ask_llm(question, context):
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    client = Groq(api_key=key)
    prompt = f"""You are InsightIQ, a careful document research assistant.
Answer ONLY using the supplied document excerpts. If the answer is not present, say:
'I couldn't find that information in the uploaded documents.'
Do not invent facts. Cite evidence using [Source: filename, page N].
Be clear, structured, and concise.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=900
    )
    return response.choices[0].message.content

st.markdown("""
<div class="hero">
  <h1>🧠 InsightIQ</h1>
  <p>Your AI research assistant — upload PDFs, ask questions, and get evidence-backed answers with page citations.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📚 Knowledge base")
    uploads = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    chunk_words = st.slider("Chunk size (words)", 100, 350, 180, 10)
    overlap = st.slider("Chunk overlap (words)", 10, 80, 35, 5)
    top_k = st.slider("Evidence passages to retrieve", 1, 8, 4)
    if st.button("Build knowledge base", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Reading PDFs and preparing searchable passages..."):
                st.session_state.chunks, st.session_state.source_names = extract_chunks(uploads, chunk_words, overlap)
            st.success(f"Indexed {len(st.session_state.chunks)} passages from {len(st.session_state.source_names)} PDF(s).")
            with st.expander("Preview extracted PDF text"):
             for i, chunk in enumerate(st.session_state.chunks[:3]):
                st.write(f"Passage {i+1}")
                st.write(chunk["text"])
                st.write(f"Source: {chunk['source']} | Page: {chunk['page']}")
    st.markdown("---")
    st.markdown("**Project stack**")
    st.caption("Streamlit · Groq LLM · TF-IDF retrieval · PyPDF")
    if st.session_state.source_names:
        st.caption("Loaded: " + ", ".join(st.session_state.source_names))

left, right = st.columns([1.8,1])
with left:
    st.subheader("💬 Ask your documents")
    if not st.session_state.chunks:
        st.info("Start by uploading PDF files in the sidebar and selecting **Build knowledge base**.")
    question = st.text_area("Your question", placeholder="Example: Summarize the key findings and list the supporting page numbers.", height=100)
    if st.button("Generate answer", type="primary", disabled=not bool(st.session_state.chunks)):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            matches = retrieve(question, st.session_state.chunks, top_k)
            if not matches:
                st.warning("No relevant passages found. Try rephrasing your question.")
            else:
                context = "\n\n".join([f"[Source: {m['source']}, page {m['page']}]\n{m['text']}" for m in matches])
                with st.spinner("Searching evidence and generating your answer..."):
                    try:
                        answer = ask_llm(question, context)
                        st.session_state.history.insert(0, {"q":question,"a":answer,"sources":matches})
                    except Exception as e:
                        st.error(str(e))
    for item in st.session_state.history:
        with st.container(border=True):
            st.markdown("**You asked:** " + item["q"])
            st.markdown(item["a"])
            with st.expander("🔎 View retrieved evidence"):
                for source in item["sources"]:
                    st.caption(f"{source['source']} · Page {source['page']}")
                    st.write(source["text"])

with right:
    st.subheader("✨ Workspace")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.metric("PDF documents", len(st.session_state.source_names))
    st.metric("Indexed passages", len(st.session_state.chunks))
    st.metric("Questions asked", len(st.session_state.history))
    st.markdown('<p class="small">Answers are grounded in retrieved document passages. Always verify important information against the original PDF.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.subheader("🚀 Try these prompts")
    for prompt in ["Give me a 5-point summary.", "What are the main findings?", "List important dates and figures.", "What limitations are mentioned?"]:
        if st.button(prompt, use_container_width=True):
            st.session_state["suggested_question"] = prompt
            st.rerun()
    if "suggested_question" in st.session_state:
        st.caption("Suggested: " + st.session_state["suggested_question"])
    st.markdown("---")
    st.caption("Internship portfolio project · Retrieval-Augmented Generation (RAG)")
