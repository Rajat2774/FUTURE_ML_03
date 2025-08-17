# app_rag_groq.py
import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
import base64

CSV_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
EMBED_FILE = "embeddings.npy"
FAISS_INDEX_FILE = "faiss.index"

st.set_page_config(page_title="Customer Support Chatbot", page_icon="🤖", layout="centered")


def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    
    encoded = base64.b64encode(data).decode()
    
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("bg3.jpeg")


st.markdown("""
<style>
body, .stApp {
    background-color: #8cc8ed !important; 
}
header, footer {visibility: hidden;}
.block-container {
    max-width: 1200px !important;
    padding-top:1rem !important;
    margin-top: -2rem !important;  
    padding-left: 1rem;
    padding-right: 1rem;
    margin: auto;
}
.chat-header,
.chat-container,
.stForm {
    width: 100%;
    max-width: 900px;  
    margin: auto;
}
.chat-header {
    background: linear-gradient(90deg, rgba(64, 92, 255, 1) 0%,rgba(112, 110, 255, 1) 100%);
    padding: 20px;
    color: white;
    font-weight: bold;
    font-size: 26px;
    text-align: center;
    border-radius: 15px;
    margin-bottom: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

.chat-container {
    background: #EAEBED;
    border-radius: 15px;
    padding: 15px;
    margin: auto;
    display: flex;
    flex-direction: column;
    height: 70vh;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}
@media (max-width: 1024px) {
    .chat-header,
    .chat-container,
    .stForm {
        max-width: 95%;
    }
}

@media (max-width: 600px) {
    .chat-header,
    .chat-container,
    .stForm {
        max-width: 100%;
        border-radius: 0 !important;
    }
}
.chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    padding-right: 10px;
    margin-bottom: 10px;
}

.user-bubble {
    background: linear-gradient(90deg, rgba(64, 92, 255, 1) 0%,rgba(112, 110, 255, 1) 100%);
    padding: 10px 14px;
    border-radius: 15px;
    color: white;
    max-width: 75%;
    margin: 5px 0;
    align-self: flex-end;
}
.bot-bubble {
    background-color: white;
    padding: 10px 14px;
    border-radius: 15px;
    max-width: 75%;
    color:black;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin: 5px 0;
    align-self: flex-start;
}
.chat-message {
    display: flex;
    flex-direction: column;
}

.stTextInput > div > div > input {
    border: none !important;
    outline: none !important;
    background-color: #EAEBED !important; 
    color: black !important;
    border-radius: 1px !important;
    padding: 10px 12px !important;
    font-size: 16px !important;
    caret-color: black !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(255, 255, 255, 0.8) !important;
}
.stForm {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 10px !important;
    margin:auto;
}
.stForm > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.stForm button {
    background: linear-gradient(90deg, rgba(64, 92, 255, 1) 0%,rgba(112, 110, 255, 1) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 8px 14px !important;
    font-size: 18px !important;
    cursor: pointer !important;
    transition: 0.2s ease-in-out;
}
.stForm button:hover {
    background-color: #2e89cc !important; 
}
.stTextInput > div > div > input::placeholder {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

model_choice = 'gemma2-9b-it'
top_k = 3
temperature = 0.2
max_tokens = 512
use_direct_answer = True
direct_answer_threshold = 0.80

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / norm

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_FILE)
    df = df[["instruction", "response", "category", "intent"]].dropna().reset_index(drop=True)
    return df

@st.cache_resource(show_spinner=True)
def load_index_and_embeddings():
    if not os.path.exists(EMBED_FILE) or not os.path.exists(FAISS_INDEX_FILE):
        st.error("Embeddings or FAISS index missing. Please generate them first.")
        st.stop()
    embeddings = np.load(EMBED_FILE).astype("float32")
    index = faiss.read_index(FAISS_INDEX_FILE)
    return embeddings, index

@st.cache_resource(show_spinner=True)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ_API_KEY. Set it in .streamlit/secrets.toml or as an env var.")
        st.stop()
    return Groq(api_key=api_key)

def retrieve(df: pd.DataFrame, index, all_embeddings: np.ndarray, query_embedding: np.ndarray, k: int):
    D, I = index.search(query_embedding[np.newaxis, :].astype("float32"), k)
    idxs = I[0].tolist()
    dists = D[0].tolist()
    qn = _normalize(query_embedding[np.newaxis, :])[0]
    best_vec = all_embeddings[idxs[0]]
    best_cos = float((_normalize(best_vec[np.newaxis, :])[0] * qn).sum())
    return idxs, dists, best_cos

def build_context(df: pd.DataFrame, indices: list[int]) -> str:
    chunks = []
    for rank, i in enumerate(indices, start=1):
        inst = df.iloc[i]["instruction"]
        resp = df.iloc[i]["response"]
        cat = df.iloc[i]["category"]
        intent = df.iloc[i]["intent"]
        chunk = (
            f"[S{rank}] Category: {cat} | Intent: {intent}\n"
            f"Instruction: {inst}\n"
            f"Answer: {resp}"
        )
        chunks.append(chunk)
    return "\n\n".join(chunks)

def generate_with_groq(client: Groq, model: str, context: str, question: str, temperature: float, max_tokens: int) -> str:
    system_prompt = (
        "You are a professional customer support chatbot. "
        "Use ONLY the provided context to answer the user's question. "
        "If the answer isn't in the context, say you don't have that information "
        "and suggest the closest relevant guidance. Keep answers concise and friendly."
    )
    user_prompt = (
        f"Context (multiple sources labeled [S1], [S2], ...):\n{context}\n\n"
        f"User question: {question}\n\n"
        "Instructions:\n"
        "- If multiple sources agree, summarize clearly.\n"
        "- If sources differ, note the difference briefly.\n"
        "- If context is insufficient, say so and propose next steps.\n"
        "- Do NOT invent details outside the context.\n"
        "- If helpful, you may reference source tags like [S1], [S2]."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _strip_think(resp.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Groq API error: {e}"


df = load_data()
embeddings, index = load_index_and_embeddings()
embedder = load_embedder()
groq_client = load_groq_client()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.markdown('<div class="chat-header">🤖 E-commerce Customer Support Chatbot</div>', unsafe_allow_html=True)


chat_html = '<div class="chat-container"><div class="chat-messages" id="chat-messages">'
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        bubble_class = "user-bubble"
    else:
        bubble_class = "bot-bubble"
    chat_html += f'<div class="chat-message"><div class="{bubble_class}">{msg["content"]}</div></div>'
chat_html += '<div id="end-of-chat"></div></div></div>'
st.markdown(chat_html, unsafe_allow_html=True)

st.markdown(
    """
    <script>
    var elem = document.getElementById('end-of-chat');
    if (elem) {
        elem.scrollIntoView({behavior: "smooth", block: "end"});
    }
    </script>
    """,
    unsafe_allow_html=True
)

with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([10, 1], gap="small")
    with cols[0]:
        user_input = st.text_input("", placeholder="Ask your queries...", label_visibility="collapsed")
    with cols[1]:
        send_clicked = st.form_submit_button("➤")



if send_clicked and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input, "meta": {}})

    q_emb = embedder.encode(user_input).astype("float32")
    idxs, dists, best_cos = retrieve(df, index, embeddings, q_emb, top_k)
    context = build_context(df, idxs)

    if use_direct_answer and best_cos >= direct_answer_threshold:
        answer_text = df.iloc[idxs[0]]["response"].strip()
        mode = f"Direct answer (cos={best_cos:.2f})"
    else:
        answer_text = generate_with_groq(
            client=groq_client,
            model=model_choice,
            context=context,
            question=user_input,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        mode = f"RAG via {model_choice} (cos={best_cos:.2f})"

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_text,
        "meta": {
            "indices": idxs,
            "distances": dists,
            "best_cos": best_cos,
            "mode": mode,
            "context": context
        }
    })
    st.rerun()


