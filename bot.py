import streamlit as st
from per_bot import questioner,reset_memory
st.set_page_config(page_title="Personal Bot")
st.title("My personal bot")

if "chat" not in st.session_state:
    st.session_state.chat = []
with st.sidebar:
    if st.button("Reset"):
        reset_memory()
        st.session_state.chat = []
        st.success("Chat memory is reseted")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me :")

if user_input:
    st.session_state.chat.append({
        "role":"user",
        "content":user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)
        
    response = questioner(user_input)

    st.session_state.chat.append({
        "role":"assistant",
        "content":response})
    with st.chat_message("assistant"):
        st.markdown(response)
