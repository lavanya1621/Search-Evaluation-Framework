# **Search Evaluation Framework** 

## **Overview**  
This framework evaluates **Hybrid Search vs. Dense Search** using **PostgreSQL, Sentence Transformers, TF-IDF, and Athina LLM**. It:  
✅ Converts **text data to embeddings** & stores in PostgreSQL  
✅ Implements **Hybrid Search (Dense + Sparse)** and **Dense Search**  
✅ Uses **Athina LLM** to compare search effectiveness based on relevance  

## **Files & Their Roles**  
📌 `insert.py` – Converts dataset to embeddings & inserts into PostgreSQL  
📌 `hybridsearch.py` – Implements **indexing & query strategies** for Hybrid & Dense Search  
📌 `athinaeval.py` – Uses **Athina LLM** to compare search strategies  

## **How It Works**  
1️⃣ **Data Preparation:**  
   - Converts text into **dense embeddings** using `sentence-transformers/all-mpnet-base-v2`  
   - Generates **TF-IDF sparse vectors**  
   - Stores embeddings in **PostgreSQL**  

2️⃣ **Search Execution:**  
   - **Dense Search:** Finds similar results using **cosine similarity**  
   - **Hybrid Search:** Combines **dense & sparse** scores for ranking  

3️⃣ **Evaluation with Athina:**  
   - Formats results into **Athina LLM-compatible datasets**  
   - Evaluates **Does Response Answer Query?** and **Context Contains Enough Information?**  
   - Compares **Hybrid vs. Dense Search** for accuracy  

---

## **Mermaid Diagram**  
```mermaid
graph TD;
    A[User Query] -->|Generate Embeddings| B[Sentence Transformer]
    A -->|Generate Sparse Vectors| C[TF-IDF Vectorizer]
    B -->|Store in DB| D[PostgreSQL]
    C -->|Store in DB| D
    A -->|Run Dense Search| E[Cosine Similarity]
    A -->|Run Hybrid Search| F[Weighted Combination]
    E -->|Return Results| G[Dense Results]
    F -->|Return Results| H[Hybrid Results]
    G -->|Format for Athina| I[Athina Dataset]
    H -->|Format for Athina| I
    I -->|Run Evaluations| J[Athina LLM]
    J -->|Compare & Score| K[Final Results]
