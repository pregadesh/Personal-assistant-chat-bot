
emb_model = "nomic-embed-text"
llm_model = "llama3"
chroma_path= "data/chroma"
coll_name = "personal_memory"

import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
emb_fun = OllamaEmbeddings(model=emb_model)

vector_db = Chroma(
    persist_directory= chroma_path,
    collection_name=coll_name,
    embedding_function=emb_fun
)

#memory :
def memory_store(text, memory_type="conversation"):
    doc = Document(
        page_content=text,
        metadata={"type":memory_type}
    )
    vector_db.add_documents([doc])

def memory_retrive(query,k=4):
    docs = vector_db.similarity_search(query,k)
    return "\n".join([i.page_content for i in docs])

#LLM
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model=llm_model)

#reseter and recreator
def reset_memory():
    global vector_db
    vector_db.delete_collection()
    vector_db = Chroma(
        persist_directory= chroma_path,
        collection_name= coll_name,
        embedding_function= emb_fun
    )

def questioner (user_in):
    if user_in.lower() in ["/reset","wipe memory"]:
        reset_memory()
        return "Memory is wiped"
    memory_context = memory_retrive(user_in)
    prompt = f""" 
Hi this your AI bot 

Use the following memory only if relevent.
If memory is insufficient, answer normally.

Memory:
{memory_context}

User Question :
{user_in}
"""
    reponse = llm.invoke(prompt)
    memory_store(f"user: {user_in}","conversation")
    memory_store(f"bot: {reponse}","conversation")

    return reponse
    
'''
questioner("Hi bot myself PREGADESH")

questioner("what is my name ?")

questioner("wipe memory")

questioner("What is my name ?")
'''