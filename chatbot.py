import os
import pandas as pd
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

# File paths
CSV_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
EMBED_FILE = "embeddings.npy"
FAISS_INDEX_FILE = "faiss.index"

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_FILE)
    df = df[["instruction", "response", "category", "intent"]].dropna().reset_index(drop=True)
    return df

@st.cache_resource
def load_index_and_embeddings():
    if not os.path.exists(EMBED_FILE) or not os.path.exists(FAISS_INDEX_FILE):
        st.error("Please run build_index.py to generate embeddings and index.")
        st.stop()
    embeddings = np.load(EMBED_FILE)
    index = faiss.read_index(FAISS_INDEX_FILE)
    return embeddings, index


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_bot_response(user_input, model, df, index, threshold=0.5):
    query_embedding = model.encode([user_input])
    distances, indices = index.search(query_embedding, 3)
    best_idx = indices[0][0]
    best_score = distances[0][0]

    if best_score < threshold:
        return df.iloc[best_idx]['response']
    else:
        # Fallback
        fuzzy_scores = df["instruction"].apply(lambda x: fuzz.partial_ratio(x.lower(), user_input.lower()))
        top_fuzzy_idx = fuzzy_scores.idxmax()
        return df.iloc[top_fuzzy_idx]['response']


#Streamlit -UI
st.set_page_config(page_title="Support Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Customer Support Chatbot")

df = load_data()
embeddings, index = load_index_and_embeddings()
model = load_model()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.chat_input("Ask your queries here...")

if user_input:
    st.session_state.chat_history.append(("You", user_input))

    response = get_bot_response(user_input, model, df, index)
    st.session_state.chat_history.append(("Bot", response))


for sender, msg in st.session_state.chat_history:
    if sender == "You":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)
