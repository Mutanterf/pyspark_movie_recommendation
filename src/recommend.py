from pyspark.sql.functions import explode, col

def generate_recommendations(model, movies, top_n=10):
    raw = model.recommendForAllUsers(top_n)

    recs = raw.select("userId", explode("recommendations").alias("rec")).select("userId", col("rec.movieId").alias("movieId"), col("rec.rating").alias("score"))

    enriched = recs.join(movies, on="movieId", how="left")
    return enriched