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
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

emb_fun = GoogleGenerativeAIEmbeddings(model=emb_model,google_api_key=gemini_api_key)

vector_db = Chroma(
    persist_directory=chroma_path,
    collection_name=coll_name,
    embedding_function=emb_fun
)

llm_main_model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=gemini_api_key,
    temperature=0.3
)
#Act as a sub model when main ran out of quota
llm_sub_model = ChatGoogleGenerativeAI(
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
    try:
        response = llm_main_model.invoke(prompt).content
    except ChatGoogleGenerativeAIError as e :
        err_txt = str(e)
        if "RESOURCE_EXHAUSTED" is err_txt or "quota" in err_txt.lower():
            try:
                response = llm_sub_model.invoke(prompt).content
                response += "\n gen3 input quota over using gen2.5"
            except ChatGoogleGenerativeAIError as e2:
                e2_txt = str(e2)
                if "RESOURCE_EXHAUSTED" is e2_txt or "quota" in e2_txt.lower():
                    return "Both model quota is Done plese contact pregadesh"
                else:
                    return "Some Other error happned please contact pregadesh."
        else:
            return "Error happned in gen3 model please contact pregadesh"

    memory_store(f"user: {user_in}", "conversation")
    memory_store(f"bot: {response}", "conversation")

    return response