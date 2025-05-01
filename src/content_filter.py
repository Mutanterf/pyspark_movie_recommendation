from pyspark.sql.functions import split, explode, col, count, avg
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F
from pyspark.ml.feature import IDF
from pyspark.ml.linalg import DenseVector
import math

def create_genre_vectors(movies_df):
    genres_exploded = movies_df.withColumn("genre", explode(split(col("genres"), "\\|")))
    genre_indexed = genres_exploded.groupBy("movieId").pivot("genre").agg(count("*")).na.fill(0)
    return genre_indexed

def build_user_profiles(ratings_df, genre_vectors):
    user_rated_movies = ratings_df.filter(ratings_df.rating >= 4.0)
    user_profiles = user_rated_movies.join(genre_vectors, on="movieId", how="inner")

    genre_cols = genre_vectors.columns[1:]  # remove movieId
    exprs = [avg(col(g)).alias(g) for g in genre_cols]
    return user_profiles.groupBy("userId").agg(*exprs)

def cosine_similarity(user_vec, movie_vec):
    dot = sum([user_vec[i] * movie_vec[i] for i in range(len(user_vec))])
    norm_u = math.sqrt(sum([x**2 for x in user_vec]))
    norm_m = math.sqrt(sum([x**2 for x in movie_vec]))
    return dot / (norm_u * norm_m) if norm_u and norm_m else 0.0


def recommend_by_content(userId, ratings_df, movies_df):
    genre_vectors = create_genre_vectors(movies_df)
    user_profiles = build_user_profiles(ratings_df, genre_vectors)

    genre_cols = genre_vectors.columns[1:]
    user_profile = user_profiles.filter(user_profiles.userId == userId).collect()
    if not user_profile:
        return []

    user_vec = [user_profile[0][g] for g in genre_cols]

    scored = genre_vectors.rdd.map(lambda row: (row["movieId"], row.asDict())) \
        .map(lambda x: (x[0], cosine_similarity(user_vec, [x[1][g] for g in genre_cols])))

    top = scored.sortBy(lambda x: -x[1]).take(10)
    movie_ids = [x[0] for x in top]
    return movies_df.filter(col("movieId").isin(movie_ids))