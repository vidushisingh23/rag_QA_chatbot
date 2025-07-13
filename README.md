# RAG-QA Chatbot (Offline)

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** using Hugging Face models. The chatbot is capable of answering user queries based on uploaded documents (e.g., PDFs, text files) by retrieving the most relevant context and generating natural language answers — all **without using APIs or internet access**.

---

### Project Context 

-  **Offline-first** design — no API keys or internet required
-  **Fully local** using Hugging Face transformer models and FAISS
-  **Lightweight and modular**, designed for secure and low-resource environments
-  **Streamlit-based UI** for an interactive and user-friendly experience
-  Ideal for personal, academic, or enterprise-level document Q&A systems

---

## Features

- 📄 Upload your own documents (PDF/TXT)
- 🔍 Embedding-based semantic search using `sentence-transformers`
- 🧠 Local LLM for answer generation
- 🛡️ Completely offline — No external API required
- 🔧 Modular Python backend (`chatbot.py`)
- 🌐 Simple UI using `Streamlit` (`app.py`)

---

## 📁 Project Structure

rag_QA_chatbot/
│
├── app.py                      # Streamlit app for the chatbot UI
├── chatbot.py                  # Core logic for semantic search + answer generation
├── knowledge_base.py           # to convert .csv dataset into text file
│
├── loan_knowledge_base.txt     # Text file with applicant records for querying
├── requirements.txt            # List of dependencies to install
├── train_dataset.csv           # reference data


