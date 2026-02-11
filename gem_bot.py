import streamlit as st
#For local test
#from dotenv import load_dotenv
#import os
#load_dotenv()
#gemini_api_key = os.getenv("gemini_api_key")

gemini_api_key = st.secrets["gemini_api_key"]


emb_model = "gemini-embedding-001"  #here im using gemini-embedding-001 
chroma_path = "data/chroma_gem"
coll_name = "personal_memory_gem"

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

emb_fun = GoogleGenerativeAIEmbeddings(model=emb_model,google_api_key=gemini_api_key)

vector_db = Chroma(
    persist_directory=chroma_path,
    collection_name=coll_name,
    embedding_function=emb_fun
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.3
)

def memory_store(text, memory_type="conversation"):
    doc = Document(
        page_content=text,
        metadata={"type": memory_type}
    )
    vector_db.add_documents([doc])

def memory_retrieve(query, k=5):
    docs = vector_db.similarity_search(query, k)
    return "\n".join([doc.page_content for doc in docs])

def reset_memory():
    global vector_db
    vector_db.delete_collection()
    vector_db = Chroma(
        persist_directory=chroma_path,
        collection_name=coll_name,
        embedding_function=emb_fun
    )


def questioner(user_in):
    if user_in.lower() in ["reset", "wipe memory"]:
        reset_memory()
        return "Memory is wiped"

    memory_context = memory_retrieve(user_in)

    prompt = f"""
You are a personal AI bot.

Use the following memory only if relevant.
If memory is insufficient, answer normally.

Memory:
{memory_context}

User Question:
{user_in}
"""

    response = llm.invoke(prompt).content

    memory_store(f"user: {user_in}", "conversation")
    memory_store(f"bot: {response}", "conversation")

    return response


