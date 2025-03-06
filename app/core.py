import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from app.data_loader import load_data


def create_breed_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> torch.Tensor:
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
            ] if pd.notna(row[col])]
        ), axis=1
    )
    texts = df['breed_text'].tolist()
    return model.encode(texts, convert_to_tensor=True)


def get_recommendation(query: str, df: pd.DataFrame, breed_embeddings: torch.Tensor,
                       model: SentenceTransformer, top_k: int = 1) -> pd.DataFrame:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, breed_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    recommended_breeds = df.iloc[top_results.indices.cpu().numpy()].copy()
    recommended_breeds['similarity_score'] = top_results.values.cpu().numpy()
    recommended_breeds['query'] = query
    return recommended_breeds


def load_model_and_data():
    DATA_FILE = "data/akc-data-latest.csv"
    df = load_data(DATA_FILE)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_breed_embeddings(df, model)
    return df, model, embeddings
