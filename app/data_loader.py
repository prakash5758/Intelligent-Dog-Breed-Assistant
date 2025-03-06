import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dog breed dataset.
    Expected columns include: Dog_name, description, temperament,
    energy_level_category, trainability_category, demeanor_category,
    grooming_frequency_category, shedding_category, etc.
    """
    return pd.read_csv(file_path)

def create_breed_text(row: pd.Series) -> str:
    """
    Combine descriptive fields into a single text representation.
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
