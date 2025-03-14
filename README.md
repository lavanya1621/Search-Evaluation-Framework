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

## 🔍 Search Process  
```mermaid
graph TD;
    A[User Query] -->|Generate Dense & Sparse Vectors| B[SentenceTransformer & TF-IDF]
    B -->|Vectorized Query| C[PostgreSQL Database]
    
    subgraph Database
        C -->|Search using Dense Vectors| D[Dense Search]
        C -->|Search using Hybrid Approach| E[Hybrid Search]
    end
    
    D -->|Top-K Results| F[Results Formatting]
    E -->|Top-K Results| F
    
    F -->|Format for Athina| G[Athina Evaluation]
    
    subgraph Athina LLM
        G -->|Compare Dense vs Hybrid| H[Does Response Answer Query?]
        G --> I[Context Contains Enough Information?]
    end
    
    H & I --> J[Final Evaluation & Scores]
    
    J -->|Comparison Results| K[Best Performing Search Approach]
