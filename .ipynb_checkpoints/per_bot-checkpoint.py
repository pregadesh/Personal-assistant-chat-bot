{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "99ed362a-da13-4f6e-b300-934d23ca76fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "emb_model = \"nomic-embed-text\"\n",
    "llm_model = \"llama3\"\n",
    "chroma_path= \"data/chroma\"\n",
    "coll_name = \"personal_memeory\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f4c07dfc-b87e-4c1b-bd94-6c2b6bfcba1f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from langchain_community.embeddings import OllamaEmbeddings\n",
    "from langchain_chroma import Chroma\n",
    "from langchain_core.documents import Document\n",
    "from langchain_community.llms import Ollama"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "04149f0b-cf1e-490c-99d3-0a81477bf94d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_138391/2802808089.py:1: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the `langchain-ollama package and should be used instead. To use it run `pip install -U `langchain-ollama` and import as `from `langchain_ollama import OllamaEmbeddings``.\n",
      "  emb_fun = OllamaEmbeddings(model=emb_model)\n"
     ]
    }
   ],
   "source": [
    "emb_fun = OllamaEmbeddings(model=emb_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5d5089a1-0ef8-4aec-98aa-61eb34bf4923",
   "metadata": {},
   "outputs": [],
   "source": [
    "vector_db = Chroma(\n",
    "    persist_directory= chroma_path,\n",
    "    collection_name=coll_name,\n",
    "    embedding_function=emb_fun\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "def98b3f-2a51-492d-9c1f-57b9ba8335ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "#memory :\n",
    "def memory_store(text, memory_type=\"conversation\"):\n",
    "    doc = Document(\n",
    "        page_content=text,\n",
    "        metadata={\"type\":memory_type}\n",
    "    )\n",
    "    vector_db.add_documents([doc])\n",
    "\n",
    "def memory_retrive(query,k=4):\n",
    "    docs = vector_db.similarity_search(query,k)\n",
    "    return \"\\n\".join([i.page_content for i in docs])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1cf7c38b-7250-4c27-bf30-49dbf25fba5a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_138391/1643800419.py:3: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the `langchain-ollama package and should be used instead. To use it run `pip install -U `langchain-ollama` and import as `from `langchain_ollama import OllamaLLM``.\n",
      "  llm = Ollama(model=llm_model)\n"
     ]
    }
   ],
   "source": [
    "#LLM\n",
    "from langchain_ollama import OllamaLLM\n",
    "llm = Ollama(model=llm_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "823faea1-a195-4734-af3f-3b9db30d1438",
   "metadata": {},
   "outputs": [],
   "source": [
    "#reseter update\n",
    "def reset_memory():\n",
    "    vector_db.delete_collection()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7a5a0e75-b504-4c38-a6fc-61d425c6d876",
   "metadata": {},
   "outputs": [],
   "source": [
    "#recreater:\n",
    "def recreater():\n",
    "    global vector_db\n",
    "    vector_db = Chroma(\n",
    "        persist_directory= chroma_path,\n",
    "        collection_name= coll_name,\n",
    "        embedding_function= emb_fun\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4bba2734-495f-4405-b1f9-9fcfba478ba3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#reseter update\n",
    "def reset_memory():\n",
    "    vector_db.delete_collection()\n",
    "    recreater()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e41208fc-f441-4cf3-bf2c-3e7517c6ade4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def questioner (user_in):\n",
    "    if user_in.lower() in [\"/reset\",\"wipe memory\"]:\n",
    "        reset_memory()\n",
    "        return \"Memory is wiped\"\n",
    "    memory_context = memory_retrive(user_in)\n",
    "    prompt = f\"\"\" \n",
    "Hi this your AI bot \n",
    "\n",
    "Use the following memory to answer accurately.\n",
    "If memory is insufficient, answer normally.\n",
    "\n",
    "Memory:\n",
    "{memory_context}\n",
    "\n",
    "User Question :\n",
    "{user_in}\n",
    "\"\"\"\n",
    "    reponse = llm.invoke(prompt)\n",
    "    memory_store(user_in,memory_type=\"conversation\")\n",
    "\n",
    "    return reponse\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "06b7076b-e524-48be-ad99-c72457b3a4ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"Hello Pregadesh! Nice to meet you. I'm your friendly AI bot. How can I assist you today?\""
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "questioner(\"Hi bot myself PREGADESH\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5bddc847-d9a5-4687-9d4b-52472981a331",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'According to our memory, your name is PREGADESH.'"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "questioner(\"what is my name ?\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "08ce67ff-79f8-4436-a0c5-a420a9110878",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Memory is wiped'"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "questioner(\"wipe memory\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "da42a20a-f28b-4aad-bd84-4d6da1b6667c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"I'm happy to help! Since I have access to the provided memory, I can confidently answer that... **you don't have a name**. The memory doesn't specify your name, so I'll just stick with what's in the memory.\""
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "questioner(\"What is my name ?\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2854424a-e4f1-456c-ba3f-b086aa34473b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
