
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _compose_text(row: pd.Series) -> str:
    """Combine fields into one text string for vectorization."""
    parts = [row.get("title", ""), row.get("skills", ""), row.get("domain", ""), row.get("location", "")]
    return " ".join([p for p in parts if p])

def build_model(df: pd.DataFrame):
    texts = df.apply(_compose_text, axis=1).values
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def _profile_to_text(skills, interests, education, location_pref):
    return " ".join([skills or "", interests or "", education or "", location_pref or ""])

def recommend(df, vectorizer, matrix, *, skills, interests, education, location_pref, top_k=5):
    profile_text = _profile_to_text(skills, interests, education, location_pref)
    if not profile_text.strip():
        return df.head(top_k).assign(score=0)

    qvec = vectorizer.transform([profile_text])
    sims = cosine_similarity(qvec, matrix)[0]

    results = df.copy()
    results["score"] = sims * 100

    # Simple location boost
    if location_pref:
        mask_loc = results["location"].str.contains(location_pref, case=False, na=False)
        results.loc[mask_loc, "score"] *= 1.2

    # Sort & return top_k
    return results.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

