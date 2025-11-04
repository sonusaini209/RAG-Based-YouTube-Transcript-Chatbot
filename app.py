import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# LOAD ENV VARIABLES
load_dotenv()

# STREAMLIT CONFIG
st.set_page_config(page_title="YouTube RAG QA", layout="wide")
st.title("YouTube Video QA Assistant")
st.caption("Ask questions about any YouTube video using RAG (Retrieval-Augmented Generation)")

# FUNCTIONS
def fetch_transcript(video_id):
    """Fetch transcript for a given YouTube video ID."""
    try:
        fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join(chunk['text'] for chunk in fetched)
        return transcript
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No English transcript found for this video.")
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
    return None


def format_docs(retrieved_docs):
    """Combine text chunks for RAG context."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# STREAMLIT UI
video_url = st.text_input("Enter YouTube Video URL or ID:")
query = st.text_input("Ask a question about the video:")
debug_mode = st.toggle("Debug Mode (Show retrieved chunks)", value=False)

if video_url:
    # Extract video ID if a full link is provided
    if "youtube.com" in video_url or "youtu.be" in video_url:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_url)
        video_id = match.group(1) if match else video_url
    else:
        video_id = video_url

    with st.spinner("Fetching transcript..."):
        transcript = fetch_transcript(video_id)

    if transcript:
        # SPLIT TEXT
       
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents([transcript])

        # CREATE EMBEDDINGS & VECTORSTORE
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # DEFINE PROMPT + LLM
        prompt = PromptTemplate(
            template="""
            You are a helpful AI assistant.
            Use ONLY the following context to answer the question.
            If the answer cannot be found in the context, say "I don’t know."

            Context:
            {context}

            Question: {question}

            Answer:
            """,
            input_variables=["context", "question"]
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        # BUILD RAG CHAIN
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        st.success("Document processed successfully!")

        # HANDLE USER QUERY
        if query:
            with st.spinner("Thinking..."):
                answer = main_chain.invoke(query)

            st.markdown("### Answer:")
            st.write(answer.strip())

            # Optional Debug Mode
            if debug_mode:
                st.divider()
                st.markdown("### Retrieved Context (Debug Info)")
                context_docs = retriever.invoke(query)
                for i, doc in enumerate(context_docs, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)
                    st.divider()

    else:
        st.warning("Could not fetch transcript. Try another video.")
else:
    st.info("Enter a YouTube link or video ID to begin.")
