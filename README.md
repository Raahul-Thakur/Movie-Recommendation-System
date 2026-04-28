# Movie Recommendation System

A Flask-based movie recommendation web application that suggests movies similar to a title entered by the user. The project uses the MovieLens 20M dataset and combines content-based signals with collaborative filtering signals to produce hybrid recommendations.

The app lets a user search for a movie, finds the closest matching title in the dataset, displays that searched movie, and returns the top 10 most similar movies with match scores and shared genre information.

## Project Overview

This project is a hybrid movie recommender system. It does not rely on only one signal such as movie name or genre. Instead, it builds a richer representation of each movie using:

- Movie title tokens
- Movie genres
- Release year and decade
- User-applied tags
- MovieLens Tag Genome relevance tags
- User rating behavior

The recommendation engine then combines:

- Content-based similarity: compares movies by title, genre, user tags, genome tags, and release decade.
- Collaborative filtering similarity: compares movies based on patterns in user ratings.
- Popularity confidence adjustment: reduces the influence of collaborative scores for movies with very few ratings.

The final score is calculated using a weighted hybrid approach:

- 60% collaborative filtering score
- 40% content-based score

## Features

- Search any movie title from the MovieLens dataset.
- Fuzzy title matching for imperfect user input.
- Top 10 similar movie recommendations.
- Match percentage for each recommendation.
- Shared genre display between the searched movie and recommended movies.
- Random fallback recommendations when no close movie match is found.
- Cached recommender model using `hybrid_recommender.pkl` to avoid rebuilding the feature matrices every time the app starts.

## Tech Stack

- Python: Core programming language
- Flask: Web application framework
- Pandas: CSV loading, cleaning, and feature preparation
- NumPy: Numerical operations and scoring
- SciPy: Sparse matrix handling
- scikit-learn: TF-IDF vectorization, genre binarization, and cosine similarity
- HTML/CSS: Frontend templates and styling
- Pickle: Local model/cache serialization

## Project Structure

```text
Recommendation System/
|-- app.py
|-- model.ipynb
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- movies.csv
|-- rating.csv
|-- tag.csv
|-- link.csv
|-- genome_scores.csv
|-- genome_tags.csv
|-- recommender.pkl
|-- hybrid_recommender.pkl
|-- static/
|   `-- style.css
`-- templates/
    |-- index.html
    `-- results.html
```

## Dataset Used

This project uses the MovieLens 20M dataset.

MovieLens 20M is a public movie recommendation dataset created by GroupLens. It contains approximately:

- 20,000,263 ratings
- 465,564 tag applications
- 27,278 movies
- 138,493 users
- Tag Genome data with more than 11 million movie-tag relevance scores

The local project uses these files:

| Local file | Purpose |
| --- | --- |
| `movies.csv` | Movie IDs, titles, and genres |
| `rating.csv` | User ratings for movies |
| `tag.csv` | User-generated movie tags |
| `link.csv` | IMDb and TMDb IDs for movies |
| `genome_scores.csv` | Relevance score between movies and genome tags |
| `genome_tags.csv` | Tag names for Tag Genome IDs |

## Where to Find the Dataset

You can download the dataset from either of these sources:

- Official GroupLens page: https://grouplens.org/datasets/movielens/20m/
- Kaggle mirror: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset

The official MovieLens 20M README is available here:

https://files.grouplens.org/datasets/movielens/ml-20m-README.html

Note: GroupLens names the files as `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`, `genome-scores.csv`, and `genome-tags.csv`. Some Kaggle versions use names such as `movie.csv`, `rating.csv`, `tag.csv`, `link.csv`, `genome_scores.csv`, and `genome_tags.csv`. This project expects the local filenames shown in the project structure above.

## How the Recommendation System Works

### 1. Data Loading

`app.py` loads movie metadata, ratings, user tags, and Tag Genome data from the CSV files.

### 2. Movie Cleaning

Movie titles are cleaned by:

- Extracting the release year from titles such as `Toy Story (1995)`
- Removing the year from the title for text matching
- Lowercasing text
- Removing punctuation
- Splitting genres into genre lists

### 3. Feature Engineering

Each movie receives a combined text feature called a `soup`. This includes:

- Weighted title text
- Weighted genre text
- User tags
- Highly relevant genome tags
- Decade information

The app uses TF-IDF vectorizers to convert title and soup text into numerical vectors. It also uses `MultiLabelBinarizer` to convert genres into a sparse genre matrix.

### 4. Collaborative Matrix

The ratings data is converted into a sparse movie-user matrix where:

- Rows represent movies
- Columns represent users
- Values represent user ratings

Cosine similarity is then used to compare movies based on rating patterns.

### 5. Hybrid Scoring

For a searched movie, the system calculates:

- Content similarity against all other movies
- Collaborative similarity against all other movies
- Confidence-adjusted collaborative scores based on rating count

The final recommendation score is:

```text
final_score = (0.6 * collaborative_score) + (0.4 * content_score)
```

The searched movie itself is removed from the results, and the highest-scoring 10 movies are returned.

## Step-by-Step Execution

### 1. Clone or Open the Project

Open a terminal in the project folder:

```bash
cd "Recommendation System"
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset Files

Download the MovieLens 20M dataset and place these files in the project root:

```text
movies.csv
rating.csv
tag.csv
link.csv
genome_scores.csv
genome_tags.csv
```

If your downloaded files have different names, rename them to match the filenames above.

### 5. Run the Flask App

```bash
python app.py
```

### 6. Open the App in a Browser

Visit:

```text
http://127.0.0.1:5000/
```

Search for a movie, for example:

```text
Toy Story
Iron Man
The Dark Knight
Jumanji
```

### 7. First Run Behavior

On the first run, the app may take some time to start because it builds the content matrix and collaborative filtering matrix from the large MovieLens files.

After building the recommender, the app saves a cache file:

```text
hybrid_recommender.pkl
```

Future runs are faster because the app loads this cache when the dataset files have not changed.

## Important Notes

- The dataset files are large, especially `rating.csv` and `genome_scores.csv`.
- Do not commit the raw dataset files or generated `.pkl` files to GitHub unless you intentionally want to store large files with Git LFS.
- The recommender is currently focused on similarity, not personalized recommendations for a logged-in user.
- The app does not currently display posters, actors, directors, or plot summaries because those fields are not included in the base MovieLens 20M dataset.

## Future Improvements

- Add movie posters and descriptions using TMDb IDs from `link.csv`.
- Add actors, directors, and plot summaries from an external movie API.
- Add user login and personalized recommendation history.
- Add filters for genre, year, and minimum rating count.
- Add evaluation metrics for recommendation quality.
- Deploy the app on Render, Railway, or another Python-friendly hosting platform.

## Depolyed at:

URL: https://movie-recommendation-system-52bz.onrender.com
