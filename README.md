
# PySpark Movie Recommender System

## Structure

* `src/` — source code
* `output/` — output results

## Setup

Make sure you have installed:

* Python 3.9+
* Apache Spark (version 3.5.5 or higher recommended)
* Java 8 or 11 (required for Spark)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Locally (without Spark cluster)

```bash
python main.py
```

---

## Run on Spark Cluster (with master and worker nodes)

```bash
spark-submit --master spark://<MASTER_IP>:7077 \
  --conf spark.executorEnv.PYSPARK_PYTHON=<PATH_TO_PYTHON_ON_WORKER> \
  --conf spark.pyspark.driver.python=<PATH_TO_PYTHON_ON_DRIVER> \
  main.py
```

**Example:**

```bash
spark-submit --master spark://172.20.10.2:7077 \
  --conf spark.executorEnv.PYSPARK_PYTHON=C:\\Users\\Dimash\\AppData\\Local\\Programs\\Python\\Python39\\python.exe \
  --conf spark.pyspark.driver.python=C:\\Users\\mutan\\AppData\\Local\\Programs\\Python\\Python39\\python.exe \
  main.py
```

---

## Output

* `output/recommendations.csv` — recommendations table
* `output/rmse.txt` — model quality metrics (RMSE)

---

## Notes

* Make sure the paths to `python.exe` are correct and exist on the respective machines — driver and worker.
* If the worker and driver are on the same machine, you can use the same path for both.
* Before running on a cluster, make sure Spark master and worker nodes are started.

---
