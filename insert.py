import os  
import json
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load environment variables
load_dotenv()

try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cur = conn.cursor()
    print("✅ Database connection successful!")

    # Drop the table if it exists
    cur.execute("DROP TABLE IF EXISTS finance_data;")
    
    # Create table with UNIQUE constraint on text_data
    cur.execute('''
        CREATE TABLE finance_data (
            id SERIAL PRIMARY KEY,
            text_data TEXT UNIQUE,
            dense_vector VECTOR(768),
            sparse_vector VECTOR(768)
        )
    ''')
    conn.commit()
    print("✅ Table created.")

    # Load JSONL file
    with open("financebench_open_source.jsonl", "r") as file:
        data = [json.loads(line) for line in file]
    print(f"✅ Loaded {len(data)} records from JSONL file.")

    # Generate dense embeddings using SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    dense_vectors = [model.encode(entry.get("answer", "")).tolist() for entry in data]

    # Generate sparse embeddings using TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=742)
    text_data = [
        f"Question: {entry.get('question', '')} "
        f"Answer: {entry.get('answer', '')} "
        f"Company: {entry.get('company', '')} "
        f"Justification: {entry.get('justification', '')} "
        f"Document: {entry.get('doc_name', '')} "
        f"Question Type: {entry.get('question_type', '')}"
        for entry in data
    ]
    
    sparse_vectors = tfidf_vectorizer.fit_transform(text_data).toarray().tolist()

    # Ensure sparse vectors have correct dimensions (Pad to 768D)
    sparse_vectors = [vec + [0] * (768 - len(vec)) if len(vec) < 768 else vec[:768] for vec in sparse_vectors]
    
    # Combine text, dense, and sparse vectors for insertion
    records = [(text, dense, sparse) for text, dense, sparse in zip(text_data, dense_vectors, sparse_vectors)]

    # Insert data using ON CONFLICT to skip duplicates
    insert_query = """
    INSERT INTO finance_data (text_data, dense_vector, sparse_vector)
    VALUES %s
    ON CONFLICT (text_data) DO NOTHING
    """
    execute_values(cur, insert_query, records)
    conn.commit()
    print("✅ Data successfully inserted into the database.")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    cur.close()
    conn.close()
