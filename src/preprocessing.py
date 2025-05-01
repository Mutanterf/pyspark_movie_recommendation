from pyspark.sql.functions import col, lpad, concat, lit

def preprocess_ratings(ratings):
    return ratings.na.drop().filter(col("rating") > 0)

def enrich_movies(movies, links):
    links = links.withColumn("imdbId", lpad(col("imdbId").cast("string"), 7, "0"))
    links = links.withColumn("imdb_url", concat(lit("https://www.imdb.com/title/tt"), col("imdbId")))
    return movies.join(links.select("movieId", "imdb_url"), on="movieId", how="left")