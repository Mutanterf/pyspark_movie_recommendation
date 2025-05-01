import kagglehub
import os

def download_movielens():
    path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
    print(f"Dataset downloaded to: {path}")
    return path
