import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.2,
    max_new_tokens=1024,
    huggingfacehub_api_token=HF_TOKEN
)

chat_model = ChatHuggingFace(llm=llm)


class QueryRequest(BaseModel):
    url: str
    question: str
    language: str = "English"


def extract_video_id(url: str):
    match = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([0-9A-Za-z_-]{11})",
        url
    )

    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    return match.group(1)


@app.post("/ask")
def ask_question(data: QueryRequest):
    try:
        video_id = extract_video_id(data.url)

        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(
            video_id,
            languages=["pa", "hi", "en"]
        )

        transcript = " ".join(
            chunk.text for chunk in transcript_list
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.create_documents([transcript])

        # Lazy load embeddings (important for deployment)
        embedding_model = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-m3",
            huggingfacehub_api_token=HF_TOKEN
        )

        vectorstore = FAISS.from_documents(
            docs,
            embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        retrieved_docs = retriever.invoke(data.question)

        context = "\n".join(
            doc.page_content for doc in retrieved_docs
        )

        template = """
You are an expert assistant.

Answer ONLY from the transcript context.
Respond in {language}.

If the answer is not found in the context, say:
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
            "question": data.question,
            "language": data.language
        })

        answer = chat_model.invoke(final_prompt)

        return {
            "answer": answer.content
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
