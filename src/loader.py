from pyspark.sql import SparkSession
import os

def init_spark(app_name="ALSRecommender"):
    # Указываем Python для мастера
    os.environ["PYSPARK_PYTHON"] = r"C:\\Users\\mutan\\AppData\\Local\\Programs\\Python\\Python39\\python.exe"

    # Для воркера укажите путь к Python на машине воркера
    # os.environ["PYSPARK_PYTHON"] = r"D:\Programs\Python39\python.exe"
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g pyspark-shell"
    spark = SparkSession.builder.appName("MovieRecommender").master("spark://172.20.10.2:7077").config("spark.driver.host", "172.20.10.2").config("spark.executor.memory", "4G").getOrCreate()
    return spark

def load_data(spark, path):
    ratings = spark.read.csv(os.path.join(path, "rating.csv"), header=True, inferSchema=True)
    movies = spark.read.csv(os.path.join(path, "movie.csv"), header=True, inferSchema=True)
    links = spark.read.csv(os.path.join(path, "link.csv"), header=True, inferSchema=True)
    return ratings, movies, links

