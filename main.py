import uvicorn
from fastapi import FastAPI
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LinearRegression

# FastAPI app setup
app = FastAPI()

FEEDBACK_FILE = "feedback.csv"

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def create_breed_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> torch.Tensor:
    df['breed_text'] = df.apply(create_breed_text, axis=1)
    texts = df['breed_text'].tolist()
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def get_recommendation(query: str, df: pd.DataFrame, breed_embeddings: torch.Tensor, model: SentenceTransformer, top_k: int = 1) -> pd.DataFrame:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, breed_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    recommended_breeds = df.iloc[top_results.indices.cpu().numpy()].copy()
    recommended_breeds['similarity_score'] = top_results.values.cpu().numpy()
    recommended_breeds['query'] = query
    return recommended_breeds

@app.post("/ask")
def ask_breed(query_request: QueryRequest):
    query = query_request.query
    try:
        recommendations = get_recommendation(query, df, breed_embeddings, semantic_model, top_k=3)
        result = recommendations[['Dog_name', 'similarity_score']].to_dict(orient="records")
        return {"recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    # Example query
    query = "I have young kids and limited time for grooming. Which breed would suit my family?"
    data_file = "akc-data-latest.csv"
    df = load_data(data_file)

    # Initialize semantic model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    breed_embeddings = create_breed_embeddings(df, model)

    # Get initial recommendation
    initial_rec = get_recommendation(query, df, breed_embeddings, model, top_k=3)
    print("Initial Recommendations:")
    print(initial_rec[["Dog_name", "similarity_score"]])

# Run FastAPI app when directly executing the script
if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
