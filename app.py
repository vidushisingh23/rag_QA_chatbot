# app.py
import streamlit as st
from chatbot import LoanRAGChatbot

st.set_page_config(page_title="Loan Approval RAG Q&A Chatbot", layout="wide")
st.markdown("""
    <h1 style='text-align: center;'>📊 Loan Approval RAG Q&A Chatbot (Offline)</h1>
    <p style='text-align: center;'>Ask a question about loan approvals based on retrieved knowledge.</p>
""", unsafe_allow_html=True)

# Load the chatbot
@st.cache_resource(show_spinner="Loading model and embeddings...")
def load_chatbot():
    return LoanRAGChatbot("loan_knowledge_base.txt")

bot = load_chatbot()

# Sample questions section
st.markdown("### 💡 Try These Sample Questions")
sample_questions = [
    "What kind of applicants get loans approved?",
    "What kind of applicants get loans rejected?",
    "What are common traits of applicants with credit history 1?",
    "What factors lead to loan rejection?"
]
selected_question = st.selectbox("Select a question to see an answer:", [""] + sample_questions)

# Input
query = st.text_input("🔍 Ask your question here:", placeholder="E.g., What kind of applicants get loans approved?", value=selected_question if selected_question else "")

if query:
    with st.spinner("Retrieving answer..."):
        answer, contexts = bot.get_answer(query)

    st.markdown("### 🤖 Answer")
    st.success(answer if answer else "Sorry, I couldn't find a clear answer.")

    with st.expander("📄 Retrieved Context"):
        for i, chunk in enumerate(contexts):
            st.markdown(f"**Chunk {i+1}:** {chunk}")