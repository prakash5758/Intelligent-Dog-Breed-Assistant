import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LinearRegression
import os

FEEDBACK_FILE = "feedback.csv"

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load your dog breed dataset.
    Expected columns: Dog_name, description, temperament,
    energy_level_category, trainability_category, demeanor_category,
    grooming_frequency_category, shedding_category, etc.
    """
    return pd.read_csv(file_path)

def create_breed_text(row: pd.Series) -> str:
    """
    Combine descriptive fields to create a rich text representation.
    """
    columns_to_include = [
        'description',
        'temperament',
        'energy_level_category',
        'trainability_category',
        'demeanor_category',
        'grooming_frequency_category',
        'shedding_category'
    ]
    texts = [str(row[col]) for col in columns_to_include if pd.notna(row[col])]
    return " ".join(texts)

def create_breed_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> torch.Tensor:
    """
    Create an aggregated text for each breed and encode into embeddings.
    """
    df['breed_text'] = df.apply(create_breed_text, axis=1)
    texts = df['breed_text'].tolist()
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def get_recommendation(query: str, df: pd.DataFrame, breed_embeddings: torch.Tensor, model: SentenceTransformer, top_k: int = 1) -> pd.DataFrame:
    """
    Embed the user query and compute cosine similarity with breed embeddings.
    Return the top_k recommendations.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, breed_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    recommended_breeds = df.iloc[top_results.indices.cpu().numpy()].copy()
    recommended_breeds['similarity_score'] = top_results.values.cpu().numpy()
    # Save additional features for re-ranking: for simplicity, we store the similarity score.
    recommended_breeds['query'] = query
    return recommended_breeds

def collect_feedback(query: str, recommended_breed: pd.Series, rating: float):
    """
    Save the user feedback. In a production system, this might be triggered
    via a web interface.
    The feedback file stores: query, Dog_name, similarity_score, rating.
    """
    feedback_data = {
        "query": [query],
        "Dog_name": [recommended_breed["Dog_name"]],
        "similarity_score": [recommended_breed["similarity_score"]],
        "rating": [rating]
    }
    feedback_df = pd.DataFrame(feedback_data)
    if os.path.exists(FEEDBACK_FILE):
        feedback_df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    else:
        feedback_df.to_csv(FEEDBACK_FILE, index=False)
    print("Feedback saved.")

def train_reranker(feedback_file: str):
    """
    Train a simple regression model using collected feedback.
    Features could be the original similarity score (and optionally, additional features).
    The target is the human rating.
    """
    feedback_df = pd.read_csv(feedback_file)
    if feedback_df.empty:
        print("No feedback data available for training.")
        return None

    X = feedback_df[["similarity_score"]]  # feature: similarity score
    y = feedback_df["rating"]              # target: human rating

    model = LinearRegression()
    model.fit(X, y)
    print("Re-ranking model trained.")
    return model

def re_rank_recommendations(initial_rec: pd.DataFrame, reranker_model: LinearRegression) -> pd.DataFrame:
    """
    Adjust the initial recommendations using the re-ranking model.
    We predict a new score using the similarity score.
    """
    if reranker_model is None:
        return initial_rec

    initial_rec = initial_rec.copy()
    # Predict a new score based on the learned regression model.
    initial_rec["adjusted_score"] = reranker_model.predict(initial_rec[["similarity_score"]])
    # Re-rank based on adjusted_score (higher means better according to human feedback).
    ranked_rec = initial_rec.sort_values(by="adjusted_score", ascending=False)
    return ranked_rec

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


print(main())
