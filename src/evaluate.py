import os

def save_rmse(rmse, path="output/rmse.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("Test RMSE:{} \n".format(round(rmse, 4)))

def precision_at_k(recommendations_df, test_df, k=10):
    from pyspark.sql.functions import col, lit

    relevant = test_df.filter(col("rating") >= 3.5).select("userId", "movieId")

    predicted = recommendations_df.select("userId", "movieId").withColumn("predicted", lit(1))
    actual = relevant.withColumn("actual_label", lit(1))

    joined = predicted.join(actual, on=["userId", "movieId"], how="left")
    joined = joined.withColumn("is_relevant", col("actual_label").isNotNull().cast("int"))

    precision = joined.groupBy("userId").agg({"is_relevant": "avg"}).selectExpr("avg(`avg(is_relevant)`) as precision_at_k").collect()[0]["precision_at_k"]
    return precision

def recall_at_k(recommendations_df, test_df, k=10):
    from pyspark.sql.functions import col, lit

    relevant = test_df.filter(col("rating") >= 3.5).select("userId", "movieId")
    relevant_counts = relevant.groupBy("userId").count().withColumnRenamed("count", "relevant_count")

    predicted = recommendations_df.select("userId", "movieId")
    actual = relevant.withColumn("actual_label", lit(1))

    joined = predicted.join(actual, on=["userId", "movieId"], how="left")
    joined = joined.withColumn("is_relevant", col("actual_label").isNotNull().cast("int"))

    hits = joined.groupBy("userId").agg({"is_relevant": "sum"}).withColumnRenamed("sum(is_relevant)", "relevant_retrieved")

    recall_df = hits.join(relevant_counts, on="userId")
    recall_df = recall_df.withColumn("recall", col("relevant_retrieved") / col("relevant_count"))

    recall = recall_df.selectExpr("avg(recall) as recall_at_k").collect()[0]["recall_at_k"]
    return recall
