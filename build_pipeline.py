import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

print("Starting Content-Based ML Pipeline build...")

# 1. Load MovieLens Small dataset
movies_path = 'ml-latest-small/movies.csv'
tags_path = 'ml-latest-small/tags.csv'
links_path = 'ml-latest-small/links.csv'
meta_path = 'movies_metadata.csv'

movies = pd.read_csv(movies_path)
tags = pd.read_csv(tags_path)
links = pd.read_csv(links_path)

print(f"Loaded {len(movies)} movies, {len(tags)} tags, and {len(links)} links from MovieLens.")

# 2. Aggregate user tags per movieId
tags_grouped = tags.groupby('movieId')['tag'].apply(
    lambda x: ' '.join(set(str(t).strip().lower() for t in x if pd.notna(t)))
).reset_index()

df = pd.merge(movies, tags_grouped, on='movieId', how='left')
df['tag'] = df['tag'].fillna('')

# 3. Merge with links to get tmdbId and imdbId
df = pd.merge(df, links, on='movieId', how='left')
df['tmdbId_clean'] = pd.to_numeric(df['tmdbId'], errors='coerce')

# 4. Load movies_metadata.csv for posters, overview, vote_average, release_date
if os.path.exists(meta_path):
    print("Loading movies_metadata.csv for poster, overview, and ratings...")
    meta = pd.read_csv(meta_path, low_memory=False)[['id', 'overview', 'poster_path', 'vote_average', 'vote_count', 'tagline', 'release_date']]
    meta['id_clean'] = pd.to_numeric(meta['id'], errors='coerce')
    meta = meta.dropna(subset=['id_clean'])
    meta = meta.drop_duplicates(subset=['id_clean'])
    
    df = pd.merge(df, meta, left_on='tmdbId_clean', right_on='id_clean', how='left')
else:
    print("movies_metadata.csv not found, proceeding with MovieLens data only.")
    df['overview'] = ''
    df['poster_path'] = ''
    df['vote_average'] = 0.0
    df['vote_count'] = 0
    df['tagline'] = ''
    df['release_date'] = ''

df = df.drop_duplicates(subset=['movieId']).reset_index(drop=True)

# 5. Extract release year from title
def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else None

df['year'] = df['title'].apply(extract_year)

# 6. Clean text fields
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')
df['poster_path'] = df['poster_path'].fillna('')
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0.0).round(1)
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).astype(int)

# Normalize genres string
df['genres_clean'] = df['genres'].str.replace('|', ' ', regex=False).str.replace('(no genres listed)', '', regex=False).fillna('')

# 7. Construct Feature Soup for TF-IDF
# Emphasize genres (3x) and tags (2x) plus tagline and overview
df['content_soup'] = (
    (df['genres_clean'] + ' ') * 3 +
    (df['tag'] + ' ') * 2 +
    df['tagline'] + ' ' +
    df['overview']
).str.strip()

# If soup is empty for any movie, fallback to title
df['content_soup'] = df['content_soup'].apply(lambda x: x if len(x.strip()) > 0 else 'movie')

print("Example processed row:")
print(df[['movieId', 'title', 'year', 'genres', 'vote_average', 'poster_path']].head(2))

# 8. Train TF-IDF Vectorizer
print("Fitting TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=12000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
tfidf_matrix = tfidf.fit_transform(df['content_soup'])
print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

# 9. Build index mappings
movie_to_idx = {int(row['movieId']): i for i, row in df.iterrows()}
idx_to_movie = {i: int(row['movieId']) for i, row in df.iterrows()}

# 10. Save artifacts to ml_artifacts/
os.makedirs('ml_artifacts', exist_ok=True)
joblib.dump(tfidf_matrix, 'ml_artifacts/tfidf_matrix.joblib')
joblib.dump(tfidf, 'ml_artifacts/tfidf_vectorizer.joblib')
joblib.dump(movie_to_idx, 'ml_artifacts/movie_to_idx.joblib')
joblib.dump(idx_to_movie, 'ml_artifacts/idx_to_movie.joblib')

# Save clean dataframe
df_save = df[['movieId', 'title', 'year', 'genres', 'genres_clean', 'tag', 'tagline', 'overview', 'poster_path', 'vote_average', 'vote_count']]
df_save.to_csv('ml_artifacts/movies_cleaned.csv', index=False)
df_save.to_pickle('ml_artifacts/movies_cleaned.pkl')

print("All ML artifacts successfully built and saved to ml_artifacts/ directory!")
