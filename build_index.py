import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

CSV_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
EMBED_FILE = "embeddings.npy"
FAISS_INDEX_FILE = "faiss.index"

df = pd.read_csv(CSV_FILE)
df = df[["instruction", "response", "category", "intent"]].dropna().reset_index(drop=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoding instructions...")
embeddings = model.encode(df["instruction"].tolist(), show_progress_bar=True)
np.save(EMBED_FILE, embeddings)


index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, FAISS_INDEX_FILE)

print(f"Saved embeddings to: {EMBED_FILE}")
print(f"Saved FAISS index to: {FAISS_INDEX_FILE}")
