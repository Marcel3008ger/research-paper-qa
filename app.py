import streamlit as st
from rag import ask

st.set_page_config(page_title="Research Paper Q&A", page_icon="📚")
st.title("📚 Research Paper Q&A Assistant")
st.caption("Ask questions about a curated collection of AI/ML research papers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the papers..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            answer = ask(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})