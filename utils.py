import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import json
from chromadb.config import Settings

def build_chroma_db():
    if os.path.exists("shl_db"):
        print("✅ shl_db already exists. Skipping build.")
        return

    print("🔨 Building Chroma vector DB...")

    df = pd.read_csv("SHL.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df["embedding"] = df["Description"].apply(lambda desc: model.encode(desc).tolist())
    df.to_csv("SHL_Product_Catalog_With_Embeddings.csv", index=False)

    df = pd.read_csv("SHL_Product_Catalog_With_Embeddings.csv")
    chroma_client = chromadb.PersistentClient(path="./shl_db", settings=Settings(anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection(name="shl_assessments")

    for index, row in df.iterrows():
        embedding = row["embedding"]
        if isinstance(embedding, str):
            embedding = json.loads(embedding)

        metadata = {
            "Assessment name": row["Assessment Name"],
            "Remote Testing": row["Remote Testing"],
            "Adaptive/IRT": row["Adaptive/IRT"],
            "Test Type": row["Test Type"],
            "Duration": row["Duration"],
            "URL": row["URL"]
        }

        collection.add(
            ids=[str(index)],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    print("✅ Assessments stored in ChromaDB!")
