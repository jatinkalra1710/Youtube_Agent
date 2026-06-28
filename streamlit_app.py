import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="YouTube AI Agent",
    page_icon="🎥",
    layout="wide"
)

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# ---------------- MODELS ----------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.2,
    max_new_tokens=1024,
    huggingfacehub_api_token=HF_TOKEN
)

chat_model = ChatHuggingFace(llm=llm)

embedding_model = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",
    huggingfacehub_api_token=HF_TOKEN
)


# ---------------- HELPERS ----------------
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None


def fetch_transcript(video_id):
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(
        video_id,
        languages=["pa", "hi", "en"]
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

    return transcript


def build_vectorstore(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript])

    vectorstore = FAISS.from_documents(
        docs,
        embedding_model
    )

    return vectorstore


def ask_question(vectorstore, question, language):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke(question)

    context = "\n".join(
        doc.page_content for doc in retrieved_docs
    )

    template = """
You are an expert assistant.

Answer only from transcript.
Respond in {language}.

If answer is not found, say:
"I Don't Know"

Context:
{context}

Question:
{question}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question", "language"]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": question,
        "language": language
    })

    answer = chat_model.invoke(final_prompt)

    return answer.content


# ---------------- UI ----------------
st.title("🎥 YouTube AI Agent")
st.caption("Ask anything from any YouTube video transcript")

with st.sidebar:
    st.header("Settings")

    language = st.selectbox(
        "Response Language",
        ["English", "Hindi", "Punjabi"]
    )

    st.info(
        """
Supported transcript languages:
- Punjabi
- Hindi
- English
"""
    )


video_url = st.text_input(
    "Paste YouTube Video URL"
)

if video_url:
    video_id = extract_video_id(video_url)

    if video_id:
        st.video(video_url)

        if "transcript" not in st.session_state:
            with st.spinner("Fetching transcript..."):
                try:
                    transcript = fetch_transcript(video_id)
                    st.session_state.transcript = transcript
                    st.session_state.vectorstore = build_vectorstore(transcript)
                    st.success("Transcript loaded successfully!")

                except Exception as e:
                    st.error(f"Error: {e}")

    else:
        st.error("Invalid YouTube URL")


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if "vectorstore" in st.session_state:
    question = st.chat_input("Ask your question...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(
                    st.session_state.vectorstore,
                    question,
                    language
                )

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
