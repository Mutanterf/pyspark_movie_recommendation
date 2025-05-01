import os
os.environ["PYSPARK_PYTHON"] = r"C:\\Users\\mutan\\AppData\\Local\\Programs\\Python\\Python39\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\\Users\\mutan\\AppData\\Local\\Programs\\Python\\Python39\\python.exe"
import time
from src.downloader import download_movielens
from src.loader import init_spark, load_data
from src.preprocessing import preprocess_ratings, enrich_movies
from src.train_model import train_model
from src.recommend import generate_recommendations
from src.evaluate import save_rmse, precision_at_k, recall_at_k
from src.content_filter import recommend_by_content
import pandas as pd

def main():
    start_total = time.time()
    dataset_path = download_movielens()
    spark = init_spark()

    ratings, movies, links = load_data(spark, dataset_path)

    ratings = preprocess_ratings(ratings).cache()
    movies = enrich_movies(movies, links).cache()

    model, rmse, test_df = train_model(ratings)
    save_rmse(rmse)

    recommendations = generate_recommendations(model, movies).cache()

    os.makedirs("output", exist_ok=True)
    recommendations.limit(1000).toPandas().to_csv("output/recommendations.csv", index=False)
    print("Saved to output/recommendations.csv")

    precision = precision_at_k(recommendations, test_df, k=10)
    print("Precision@10:", round(precision,4))

    recall = recall_at_k(recommendations, test_df, k=10)
    print("Recall@10:", round(recall,4))

    with open("output/rmse.txt", "a") as f:
        f.write("Precision@10: {} \n".format(precision))
        f.write("Recall@10: {} \n".format(recall))

    print("Content based recommendations for cold user")
    content_recs = recommend_by_content(userId=1, ratings_df=ratings, movies_df=movies)
    content_recs.show(5, truncate=False)

    print("Total execution time:", time.time() - start_total)

if __name__ == "__main__":
    main()
