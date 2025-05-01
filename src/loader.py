from pyspark.sql import SparkSession
import os

def init_spark(app_name="ALSRecommender"):
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g pyspark-shell"
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark, path):
    ratings = spark.read.csv(os.path.join(path, "rating.csv"), header=True, inferSchema=True)
    movies = spark.read.csv(os.path.join(path, "movie.csv"), header=True, inferSchema=True)
    links = spark.read.csv(os.path.join(path, "link.csv"), header=True, inferSchema=True)
    return ratings, movies, links

