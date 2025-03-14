import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


from athina.evals import DoesResponseAnswerQuery, ContextContainsEnoughInformation, Faithfulness
from athina.loaders import Loader
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.runner.run import EvalRunner


load_dotenv()


OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_query_vectors(query):
    """Generate dense and sparse vectors for a given query."""
    dense_vector = model.encode(query).tolist()
    
    # Retrieve stored sparse vectorization method
    cur.execute("SELECT text_data FROM finance_data LIMIT 1000;")  # Sample texts to fit TF-IDF
    texts = [row[0] for row in cur.fetchall()]
    
    vectorizer = TfidfVectorizer(max_features=768)
    vectorizer.fit(texts)  # Fit on existing data
    
    sparse_vector = vectorizer.transform([query]).toarray()[0].tolist()
    sparse_vector += [0] * (768 - len(sparse_vector))  # Pad to 768D
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

def format_dense_for_athina(query, search_results):
    """Format dense search results for Athina evaluation."""
   
    context = [text for text, _ in search_results["dense_results"]]
    
    response = f"DENSE : Based on the search results, here's information about '{query}':\n"
  
    for i, (text, _) in enumerate(search_results["dense_results"][:3]):
        response += f"\n• {text[:200]}..."
    
    return {
        "query": query,
        "context": context,
        "response": response,
       
    }

def format_hybrid_for_athina(query, search_results):
    """Format hybrid search results for Athina evaluation."""
  
    context = [text for text, _ in search_results["hybrid_results"]]
    

    response = f"HYBRID Based on the search results, here's information about '{query}':\n"
  
    for i, (text, _) in enumerate(search_results["hybrid_results"][:3]):
        response += f"\n• {text[:200]}..."
    
    return {
        "query": query,
        "context": context,  
        "response": response,
       
    }


example_queries = [
    "By how much did Pepsico increase its unsecured five year revolving credit agreement on May 26, 2023? Answer: $400,000,000 increase. ",
    "Has Microsoft increased its debt on balance sheet between FY2023 and the FY2022 period?",
    "Which region had the Highest EBITDAR Contribution for MGM during FY2022?",
    "What drove the increase in Ulta Beauty's merchandise inventories balance at end of FY2023?"
   
    
]


athina_dataset = []


for query in example_queries:
    print(f"\n📊 Processing query: {query}")

    search_results = search_finance_data(query)
 
    dense_data = format_dense_for_athina(query, search_results)
    hybrid_data = format_hybrid_for_athina(query, search_results)
    
   
    athina_dataset.append(dense_data)
    athina_dataset.append(hybrid_data)
    
    print("\n🔍 Dense Search Top Result:")
    print(search_results["dense_results"][0][0][:200] + "..." if search_results["dense_results"] else "No results")
    
    print("\n⚡ Hybrid Search Top Result:")
    print(search_results["hybrid_results"][0][0][:200] + "..." if search_results["hybrid_results"] else "No results")


dataset = Loader().load_dict(athina_dataset)



print("\n📋 Evaluation Dataset Preview:")
preview_df = pd.DataFrame(dataset)
print("Sample dataset before sending:", athina_dataset[:2])  # Check first 2 queries

# Print all available columns
print("Available columns:", list(preview_df.columns))

# Print just the first few rows of the entire DataFrame to see what's available
print("Preview of data:")
print(preview_df.head())

# Try to print only the query column which should exist
try:
    print(preview_df[["query"]])
except KeyError as e:
    print(f"Column error: {e}")
    # If even 'query' doesn't exist, just print the first column
    if len(preview_df.columns) > 0:
        print(preview_df[[preview_df.columns[0]]])
eval_model = "gpt-3.5-turbo" # You can use "gpt-3.5-turbo" if you prefer
eval_suite = [
    DoesResponseAnswerQuery(model=eval_model),
    ContextContainsEnoughInformation(model=eval_model)
    
   
]

print("\n🧪 Running evaluation suite...")

try:
    batch_eval_result = EvalRunner.run_suite(
        evals=eval_suite,
        data=dataset,
        max_parallel_evals=1
    )
    
    # Display results
    print("\n✅ Evaluation complete!")
    print(batch_eval_result)
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(batch_eval_result)
    
    # Compare dense vs hybrid
    dense_results = results_df[results_df['response'].str.contains('DENSE')]
    hybrid_results = results_df[results_df['response'].str.contains('HYBRID')]
    
    # Calculate average scores
    print("\n📊 Average Scores:")
    dense_avg = dense_results.mean(numeric_only=True)
    hybrid_avg = hybrid_results.mean(numeric_only=True)
    
    dense_answer_score = dense_avg["Does Response Answer Query passed"]
    dense_context_score = dense_avg["Context Contains Enough Information passed"]
    hybrid_answer_score = hybrid_avg["Does Response Answer Query passed"]
    hybrid_context_score = hybrid_avg["Context Contains Enough Information passed"]

    dense_points = (dense_answer_score + dense_context_score) 
    hybrid_points = (hybrid_answer_score + hybrid_context_score) 
    
    if dense_points > hybrid_points:
        print("Overall winner: Dense Search",dense_points)
        print("Hybrid Search Points:",hybrid_points)
    
    elif hybrid_points > dense_points:
        print("Overall winner: Hybrid Search",hybrid_points)
        print("Dense Search Points:",dense_points)
    else:
        print("Overall result: Tie")


                    
    
    
    
except Exception as e:
    print(f"\n❌ Evaluation failed: {str(e)}")


cur.close()
conn.close()
