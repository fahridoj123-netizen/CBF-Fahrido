import os
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

class ContentBasedRecommender:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        artifacts_dir = getattr(settings, 'ML_ARTIFACTS_DIR', os.path.join(settings.BASE_DIR, 'ml_artifacts'))
        print(f"Loading Content-Based ML artifacts from {artifacts_dir}...")

        # Load clean dataset
        df_path = os.path.join(artifacts_dir, 'movies_cleaned.pkl')
        if not os.path.exists(df_path):
            df_path = os.path.join(artifacts_dir, 'movies_cleaned.csv')
            self.movies_df = pd.read_csv(df_path)
        else:
            self.movies_df = pd.read_pickle(df_path)

        # Load TF-IDF matrix & mappings
        self.tfidf_matrix = joblib.load(os.path.join(artifacts_dir, 'tfidf_matrix.joblib'))
        self.movie_to_idx = joblib.load(os.path.join(artifacts_dir, 'movie_to_idx.joblib'))
        self.idx_to_movie = joblib.load(os.path.join(artifacts_dir, 'idx_to_movie.joblib'))
        
        # Ensure fast dictionary lookup for movie details
        self.movies_dict = {
            int(row['movieId']): self._format_movie_record(row)
            for _, row in self.movies_df.iterrows()
        }

        # Cache unique genres
        all_genres = set()
        for g_str in self.movies_df['genres'].dropna():
            for g in str(g_str).split('|'):
                g = g.strip()
                if g and g != '(no genres listed)':
                    all_genres.add(g)
        self.genres_list = sorted(list(all_genres))

        print(f"Loaded {len(self.movies_df)} movies and TF-IDF matrix {self.tfidf_matrix.shape} successfully.")

    def _format_movie_record(self, row):
        poster = row.get('poster_path', '')
        if pd.notna(poster) and str(poster).strip():
            poster_url = f"https://image.tmdb.org/t/p/w500{poster.strip()}"
        else:
            poster_url = None

        genres_raw = str(row.get('genres', ''))
        genres_list = [g.strip() for g in genres_raw.split('|') if g.strip() and g.strip() != '(no genres listed)']

        tags_raw = str(row.get('tag', ''))
        tags_list = [t.strip() for t in tags_raw.split() if t.strip()]

        return {
            'movieId': int(row['movieId']),
            'title': str(row.get('title', 'Unknown Title')),
            'year': int(row['year']) if pd.notna(row.get('year')) and row.get('year') else None,
            'genres': genres_raw,
            'genres_list': genres_list,
            'tags_list': tags_list[:8],  # limit tags for clean display
            'tagline': str(row.get('tagline', '')),
            'overview': str(row.get('overview', '')),
            'poster_url': poster_url,
            'vote_average': float(row.get('vote_average', 0.0)) if pd.notna(row.get('vote_average')) else 0.0,
            'vote_count': int(row.get('vote_count', 0)) if pd.notna(row.get('vote_count')) else 0,
        }

    def get_movie(self, movie_id):
        return self.movies_dict.get(int(movie_id))

    def get_recommendations(self, movie_id, k=10):
        try:
            movie_id = int(movie_id)
            k = max(1, min(int(k), 50))
        except (ValueError, TypeError):
            return None

        if movie_id not in self.movie_to_idx:
            return None

        idx = self.movie_to_idx[movie_id]
        target_movie = self.movies_dict[movie_id]
        target_genres_set = set(target_movie['genres_list'])
        target_tags_set = set(target_movie['tags_list'])

        # 1. Real-time Cosine Similarity calculation
        # Vector shape: (1, num_features), Matrix shape: (N, num_features)
        movie_vec = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(movie_vec, self.tfidf_matrix).flatten()

        # 2. Retrieve top-k indices (excluding query movie itself)
        candidate_indices = sim_scores.argsort()[::-1]
        
        recommendations = []
        relevant_count = 0

        for cand_idx in candidate_indices:
            if cand_idx == idx:
                continue

            cand_movieId = self.idx_to_movie[cand_idx]
            cand_movie = self.movies_dict[cand_movieId]
            score = float(sim_scores[cand_idx])

            # Calculate vector angle theta in degrees: theta = arccos(clip(score, -1, 1))
            clamped_score = max(-1.0, min(1.0, score))
            angle_rad = math.acos(clamped_score)
            angle_deg = math.degrees(angle_rad)

            # Precision check: does candidate share at least one genre (or tag if no genre) with target movie?
            cand_genres_set = set(cand_movie['genres_list'])
            shared_genres = target_genres_set.intersection(cand_genres_set)
            
            if target_genres_set:
                is_relevant = len(shared_genres) > 0
            else:
                shared_tags = target_tags_set.intersection(set(cand_movie['tags_list']))
                is_relevant = len(shared_tags) > 0

            if is_relevant:
                relevant_count += 1

            rec_item = dict(cand_movie)
            rec_item['similarity_score'] = round(score, 4)
            rec_item['similarity_percentage'] = round(score * 100, 1)
            rec_item['vector_angle_deg'] = round(angle_deg, 2)
            rec_item['is_relevant'] = is_relevant
            rec_item['shared_genres'] = list(shared_genres)

            recommendations.append(rec_item)
            if len(recommendations) >= k:
                break

        # Calculate Precision@K
        precision_at_k = round((relevant_count / max(1, len(recommendations))) * 100, 1)

        return {
            'target_movie': target_movie,
            'k': k,
            'recommendations': recommendations,
            'precision_at_k': precision_at_k,
            'relevant_count': relevant_count,
        }

    def search_movies(self, query=None, genre=None, page=1, per_page=24):
        filtered = self.movies_df

        if genre and genre != 'All':
            filtered = filtered[filtered['genres'].str.contains(genre, case=False, na=False, regex=False)]

        if query and query.strip():
            q = query.strip()
            # Search in title, tag, or overview using literal text matching (regex=False)
            mask = (
                filtered['title'].str.contains(q, case=False, na=False, regex=False) |
                filtered['tag'].str.contains(q, case=False, na=False, regex=False) |
                filtered['genres_clean'].str.contains(q, case=False, na=False, regex=False)
            )
            filtered = filtered[mask]

        total_count = len(filtered)
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_records = filtered.iloc[start_idx:end_idx]

        items = [self.movies_dict[int(row['movieId'])] for _, row in page_records.iterrows()]

        return {
            'items': items,
            'total_count': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages,
            'has_prev': page > 1,
            'has_next': page < total_pages,
            'prev_page': page - 1,
            'next_page': page + 1,
        }

    def get_featured_movies(self, limit=10):
        # Select popular movies with high vote count & good rating for showcase
        valid = self.movies_df[self.movies_df['poster_path'].str.len() > 0]
        sorted_df = valid.sort_values(by=['vote_count', 'vote_average'], ascending=[False, False])
        return [self.movies_dict[int(row['movieId'])] for _, row in sorted_df.head(limit).iterrows()]
