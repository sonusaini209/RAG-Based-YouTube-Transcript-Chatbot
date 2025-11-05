import os
import re
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube RAG QA", layout="wide")
st.title("YouTube Video QA Assistant")
st.caption("Ask any question about a YouTube video using RAG")

# LOAD ENV VARIABLES
load_dotenv()

# Utility: Robust YouTube video ID extraction
def extract_video_id(url_or_id):
    # Direct ID pass-through
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id
    # Find ID in a YouTube link
    match = re.search(r"(?:v=|\/|embed\/|watch\?v=|youtu\.be\/)([0-9A-Za-z_-]{11})", url_or_id)
    if match:
        return match.group(1)
    return None  # Could not extract

# Video input accepts URL or ID
video_input = st.text_input("Enter YouTube video URL or ID:")
query = st.text_input("Ask a question about the video:")

if video_input:
    video_id = extract_video_id(video_input)
    if not video_id:
        st.warning("Could not extract a YouTube video ID from your input.")
        st.stop()

    with st.spinner("Fetching transcript..."):
        try:
            ytt_api = YouTubeTranscriptApi()
            fetched_transcript = ytt_api.fetch(video_id)
            transcript = " ".join(chunk.text for chunk in fetched_transcript)
        except TranscriptsDisabled:
            transcript = ""
            st.warning("Transcript disabled for this video.")
        except Exception as e:
            transcript = ""
            st.warning(f"Transcript fetch failed: {e}")

    if transcript:
        # Split transcript into chunks for RAG
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        # Embeddings & vectorstore from .env key
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Prompt & LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
            input_variables=['context', 'question']
        )
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        st.success("Transcript indexed and model ready!")

        # Query interface
        if query:
            with st.spinner("Answering..."):
                answer = main_chain.invoke(query)
            st.markdown("### Answer:")
            st.write(answer.strip())

            # Toggle for debug info
            if st.toggle("Show context chunks"):
                context_docs = retriever.invoke(query)
                st.markdown("### Retrieved Chunks")
                for i, doc in enumerate(context_docs, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.code(doc.page_content)
                    st.divider()
    else:
        st.warning("No transcript found for this video.")
else:
    st.info("Enter a YouTube video URL or video ID to begin.")

