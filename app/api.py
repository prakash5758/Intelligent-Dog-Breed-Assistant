from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from app.data_loader import load_data

# DATA_FILE = "data/akc-data-latest.csv"
# df = load_data(DATA_FILE)
# print(df.shape)
# from app.nlp_pipeline import create_breed_embeddings, get_recommendation
from sentence_transformers import SentenceTransformer

def create_breed_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> torch.Tensor:
    """
    Create an aggregated text for each breed and encode it into embeddings.
    """
    # Combine descriptive columns into one text field
    df['breed_text'] = df.apply(
        lambda row: " ".join(
            [str(row[col]) for col in [
                'description',
                'temperament',
                'energy_level_category',
                'trainability_category',
                'demeanor_category',
                'grooming_frequency_category',
                'shedding_category'
            ] if pd.notna(row[col])
            ]
        ),
        axis=1
    )
    texts = df['breed_text'].tolist()
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings



def get_recommendation(query: str, df: pd.DataFrame, breed_embeddings: torch.Tensor,
                       model: SentenceTransformer, top_k: int = 1) -> pd.DataFrame:
    """
    Embed the user query, compute cosine similarity with breed embeddings,
    and return the top_k recommendations.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, breed_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    recommended_breeds = df.iloc[top_results.indices.cpu().numpy()].copy()
    recommended_breeds['similarity_score'] = top_results.values.cpu().numpy()
    recommended_breeds['query'] = query
    return recommended_breeds



class QueryRequest(BaseModel):
    user_id: str
    query: str

DATA_FILE = "data/akc-data-latest.csv"
app = FastAPI()

# Load dataset and initialize the model at startup
df = load_data(DATA_FILE)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
breed_embeddings = create_breed_embeddings(df, semantic_model)

@app.post("/ask")
def ask_breed(query_request: QueryRequest):
    query = query_request.query
    try:
        recommendations = get_recommendation(query, df, breed_embeddings, semantic_model, top_k=3)
        result = recommendations[['Dog_name', 'similarity_score']].to_dict(orient="records")
        return {"recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
