# 🎥 YouTube RAG QA Assistant

A **Retrieval-Augmented Generation (RAG)** Streamlit app that lets you ask questions about **any YouTube video** using its transcript.

---

## 🚀 Features
- Fetch YouTube transcript automatically.
- Generate embeddings with OpenAI.
- Retrieve relevant chunks.
- Get concise GPT answers.
- Debug mode to view transcript chunks.
- One-click deploy to Render.

---

## 🧩 Tech Stack
- **LangChain** (RAG pipeline)
- **OpenAI GPT-4o-mini**
- **FAISS** (vector search)
- **Streamlit** (UI)
- **YouTube Transcript API**
- **Render** (deployment)

---

## ⚙️ Local Setup

```bash
git clone https://github.com/<your-username>/youtube-rag-qa.git
cd youtube-rag-qa
pip install -r requirements.txt
cp .env.example .env
