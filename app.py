import pickle
import re
from pathlib import Path
from difflib import get_close_matches

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


app = Flask(__name__)


YEAR_PATTERN = re.compile(r"\((\d{4})\)")
TEXT_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
COLLAB_WEIGHT = 0.6
CONTENT_WEIGHT = 0.4
GENOME_RELEVANCE_THRESHOLD = 0.75
CACHE_FILE = Path("hybrid_recommender.pkl")
DATA_FILES = [
    "movies.csv",
    "rating.csv",
    "tag.csv",
    "genome_tags.csv",
    "genome_scores.csv",
]


def extract_year(title):
    match = YEAR_PATTERN.search(title or "")
    return match.group(1) if match else "Unknown"


def normalize_text(value):
    value = TEXT_PATTERN.sub(" ", str(value or "").lower())
    return " ".join(value.split())


def clean_title(title):
    title = YEAR_PATTERN.sub("", title or "")
    return normalize_text(title)


def parse_genres(value):
    if not isinstance(value, str) or value == "(no genres listed)":
        return []
    return [genre.strip() for genre in value.split("|") if genre.strip()]


def unique_join(values, limit=None):
    seen = []
    seen_lookup = set()

    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen_lookup:
            continue
        seen_lookup.add(normalized)
        seen.append(normalized)
        if limit is not None and len(seen) >= limit:
            break

    return " ".join(seen)


def build_soup(row):
    title_tokens = clean_title(row["title"])
    genre_tokens = " ".join(genre.lower().replace("-", " ") for genre in row["genre_list"])
    year = row["year"]
    decade = f"{year[:3]}0s" if year != "Unknown" else "unknown decade"

    weighted_title = " ".join([title_tokens] * 3)
    weighted_genres = " ".join([genre_tokens] * 2)
    weighted_tags = row["tag_text"]
    weighted_genome = row["genome_tag_text"]

    return " ".join(
        part
        for part in [
            weighted_title,
            weighted_genres,
            weighted_tags,
            weighted_genome,
            decade,
        ]
        if part
    ).strip()


def load_movies():
    df = pd.read_csv("movies.csv")
    df["title"] = df["title"].fillna("")
    df["genres"] = df["genres"].fillna("(no genres listed)")
    df["year"] = df["title"].apply(extract_year)
    df["clean_title"] = df["title"].apply(clean_title)
    df["genre_list"] = df["genres"].apply(parse_genres)
    df["genre_text"] = df["genre_list"].apply(lambda genres: ", ".join(genres) if genres else "Unknown")
    return df


def load_tag_features():
    tags = pd.read_csv("tag.csv", usecols=["movieId", "tag"])
    tags["tag"] = tags["tag"].fillna("")
    tags = tags[tags["tag"].astype(str).str.strip().ne("")]
    return tags.groupby("movieId")["tag"].agg(lambda values: unique_join(values, limit=20))


def load_genome_features():
    genome_tags = pd.read_csv("genome_tags.csv", usecols=["tagId", "tag"])
    genome_scores = pd.read_csv(
        "genome_scores.csv",
        usecols=["movieId", "tagId", "relevance"],
        dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    )
    genome_scores = genome_scores[genome_scores["relevance"] >= GENOME_RELEVANCE_THRESHOLD]

    merged = genome_scores.merge(genome_tags, on="tagId", how="left")
    merged["tag"] = merged["tag"].fillna("")
    merged = merged[merged["tag"].astype(str).str.strip().ne("")]
    return merged.groupby("movieId")["tag"].agg(lambda values: unique_join(values, limit=15))


def build_feature_matrix(df):
    title_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=20000)
    soup_vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
    genre_binarizer = MultiLabelBinarizer()

    title_matrix = title_vectorizer.fit_transform(df["clean_title"])
    soup_matrix = soup_vectorizer.fit_transform(df["soup"])
    genre_matrix = csr_matrix(genre_binarizer.fit_transform(df["genre_list"]))

    return hstack(
        [
            title_matrix * 1.8,
            soup_matrix * 1.3,
            genre_matrix * 2.2,
        ]
    ).tocsr()


def build_collaborative_matrix(movie_ids):
    ratings = pd.read_csv(
        "rating.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    ratings = ratings[ratings["movieId"].isin(movie_ids)].copy()

    movie_index_lookup = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    ratings["movie_idx"] = ratings["movieId"].map(movie_index_lookup)
    ratings = ratings.dropna(subset=["movie_idx"])

    user_codes, unique_users = pd.factorize(ratings["userId"], sort=True)
    movie_codes = ratings["movie_idx"].astype("int32").to_numpy()
    rating_values = ratings["rating"].to_numpy(dtype=np.float32)

    matrix = csr_matrix(
        (rating_values, (movie_codes, user_codes.astype(np.int32))),
        shape=(len(movie_ids), len(unique_users)),
        dtype=np.float32,
    )
    counts = np.bincount(movie_codes, minlength=len(movie_ids)).astype(np.float32)
    return matrix, counts


def min_max_scale(scores):
    scores = np.asarray(scores, dtype=np.float32)
    minimum = float(scores.min())
    maximum = float(scores.max())

    if maximum - minimum < 1e-9:
        return np.zeros_like(scores, dtype=np.float32)

    return (scores - minimum) / (maximum - minimum)


def prepare_recommender():
    if CACHE_FILE.exists():
        with CACHE_FILE.open("rb") as cache_handle:
            cached_payload = pickle.load(cache_handle)
        if cached_payload.get("signature") == get_data_signature():
            return (
                cached_payload["movies"],
                cached_payload["content_matrix"],
                cached_payload["collaborative_matrix"],
                cached_payload["rating_counts"],
            )

    df = load_movies()
    df["tag_text"] = df["movieId"].map(load_tag_features()).fillna("")
    df["genome_tag_text"] = df["movieId"].map(load_genome_features()).fillna("")
    df["soup"] = df.apply(build_soup, axis=1)

    content_matrix = build_feature_matrix(df)
    collaborative_matrix, rating_counts = build_collaborative_matrix(df["movieId"].tolist())

    df["rating_count"] = rating_counts.astype(int)
    cache_payload = {
        "signature": get_data_signature(),
        "movies": df,
        "content_matrix": content_matrix,
        "collaborative_matrix": collaborative_matrix,
        "rating_counts": rating_counts,
    }
    with CACHE_FILE.open("wb") as cache_handle:
        pickle.dump(cache_payload, cache_handle, protocol=pickle.HIGHEST_PROTOCOL)
    return df, content_matrix, collaborative_matrix, rating_counts


def get_data_signature():
    return {
        file_name: (
            Path(file_name).stat().st_mtime,
            Path(file_name).stat().st_size,
        )
        for file_name in DATA_FILES
    }


movies, content_matrix, collaborative_matrix, rating_counts = prepare_recommender()
title_lookup = {title.lower(): idx for idx, title in enumerate(movies["title"])}


def fuzzy_match_title(user_title):
    normalized = clean_title(user_title)
    if not normalized:
        return None

    exact_matches = movies.loc[movies["clean_title"] == normalized, "title"].tolist()
    if exact_matches:
        return exact_matches[0]

    candidate_pool = movies["clean_title"].tolist()
    close_matches = get_close_matches(normalized, candidate_pool, n=1, cutoff=0.45)
    if not close_matches:
        return None

    matched_clean_title = close_matches[0]
    matched_titles = movies.loc[movies["clean_title"] == matched_clean_title, "title"].tolist()
    return matched_titles[0] if matched_titles else None


def serialize_movie(row, final_score=None, query_genres=None):
    shared_genres = []
    if query_genres is not None:
        shared_genres = sorted(set(query_genres).intersection(row["genre_list"]))

    return {
        "title": row["title"],
        "year": row["year"],
        "genres": row["genre_text"],
        "rating_count": int(row["rating_count"]),
        "similarity_score": round(float(final_score) * 100, 1) if final_score is not None else None,
        "shared_genres": ", ".join(shared_genres) if shared_genres else "No direct genre overlap",
    }


def get_hybrid_scores(idx):
    content_scores = cosine_similarity(content_matrix[idx], content_matrix).flatten()
    collaborative_scores = cosine_similarity(collaborative_matrix[idx], collaborative_matrix).flatten()

    content_scores = min_max_scale(content_scores)
    collaborative_scores = min_max_scale(collaborative_scores)

    confidence = np.clip(np.log1p(rating_counts) / np.log1p(50), 0.0, 1.0)
    adjusted_collaborative = collaborative_scores * confidence

    final_scores = (COLLAB_WEIGHT * adjusted_collaborative) + (CONTENT_WEIGHT * content_scores)
    final_scores[idx] = -1.0
    return final_scores


def get_recommendations(title, top_n=10):
    matched_title = fuzzy_match_title(title)
    if not matched_title:
        return None, []

    idx = title_lookup[matched_title.lower()]
    hybrid_scores = get_hybrid_scores(idx)
    ranked_indices = hybrid_scores.argsort()[::-1]
    recommendation_indices = ranked_indices[:top_n]

    searched_movie = serialize_movie(movies.iloc[idx])
    query_genres = movies.iloc[idx]["genre_list"]
    recommendations = [
        serialize_movie(movies.iloc[movie_idx], hybrid_scores[movie_idx], query_genres)
        for movie_idx in recommendation_indices
    ]

    return searched_movie, recommendations


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        movie_title = request.form["movie"]
        searched_movie, recommendations = get_recommendations(movie_title)

        if not searched_movie:
            message = f'Couldn\'t find "{movie_title}". Showing random picks instead.'
            fallback = [
                serialize_movie(row)
                for _, row in movies.sample(10).iterrows()
            ]
            return render_template(
                "results.html",
                message=message,
                searched_movie=None,
                recommendations=fallback,
            )

        return render_template(
            "results.html",
            message=f'Top 10 hybrid recommendations for "{searched_movie["title"]}"',
            searched_movie=searched_movie,
            recommendations=recommendations,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
