from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import count

def train_model(ratings):
    ratings = ratings.filter((ratings.userId <= 100000) & (ratings.movieId <= 100000))

    active_users = ratings.groupBy("userId").agg(count("*").alias("cnt")).filter("cnt >= 20")
    ratings = ratings.join(active_users.select("userId"), on="userId")

    popular_movies = ratings.groupBy("movieId").agg(count("*").alias("cnt")).filter("cnt >= 20")
    ratings = ratings.join(popular_movies.select("movieId"), on="movieId")

    train, test = ratings.randomSplit([0.8, 0.2], seed=42)

    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        maxIter=10,
        rank=30,
        regParam=0.05
    )
    model = als.fit(train)
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("RMSE:", round(rmse, 4))
    return model, rmse, test
