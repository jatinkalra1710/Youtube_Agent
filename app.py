import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.2,
    max_new_tokens=1024,
    huggingfacehub_api_token=HF_TOKEN
)

chat_model = ChatHuggingFace(llm=llm)

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small"
)


class QueryRequest(BaseModel):
    url: str
    question: str
    language: str = "English"


def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1)


@app.post("/ask")
def ask_question(data: QueryRequest):
    video_id = extract_video_id(data.url)

    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=["pa", "hi", "en"])

    transcript = " ".join(chunk.text for chunk in transcript_list)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript])

    vectorstore = FAISS.from_documents(
        docs,
        embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke(data.question)

    context = "\n".join(doc.page_content for doc in retrieved_docs)

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
        "question": data.question,
        "language": data.language
    })

    answer = chat_model.invoke(final_prompt)

    return {
        "answer": answer.content
    }

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
