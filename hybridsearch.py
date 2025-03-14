import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load environment variables
load_dotenv()

# Database connection
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_query_vectors(query):
    """Generate dense and sparse vectors for a given query."""
    dense_vector = model.encode(query).tolist()
    
   
    cur.execute("SELECT text_data FROM finance_data LIMIT 1000;") 
    texts = [row[0] for row in cur.fetchall()]
    
    vectorizer = TfidfVectorizer(max_features=768)
    vectorizer.fit(texts)  
    
    sparse_vector = vectorizer.transform([query]).toarray()[0].tolist()
    sparse_vector += [0] * (768 - len(sparse_vector))  
    return dense_vector, sparse_vector

def search_finance_data(query, top_k=10):
    """Perform Dense, Sparse, and Hybrid searches."""
    dense_vec, sparse_vec = get_query_vectors(query)

    # Dense search
    cur.execute(
        """
        SELECT text_data, 1 - (dense_vector <=> %s::vector) AS score
        FROM finance_data
        ORDER BY score DESC
        LIMIT %s;
        """, (dense_vec, top_k)
    )
    dense_results = cur.fetchall()



    # Hybrid search
    cur.execute(
        """
        SELECT text_data, 
               (0.5 * (1 - (dense_vector <=> %s::vector)) + 0.5 * (1 - (sparse_vector <=> %s::vector))) AS final_score
        FROM finance_data
        ORDER BY final_score DESC
        LIMIT %s;
        """, (dense_vec, sparse_vec, top_k)
    )
    hybrid_results = cur.fetchall()

    return {
        "dense_results": dense_results,
        
        "hybrid_results": hybrid_results
    }


test_query = "What is the financial impact of market downturns?"
results = search_finance_data(test_query)

print("\n🔍 Dense Search Results:")
for res in results["dense_results"]:
    print(res)



print("\n⚡ Hybrid Search Results:")
for res in results["hybrid_results"]:
    print(res)

cur.close()
conn.close() 