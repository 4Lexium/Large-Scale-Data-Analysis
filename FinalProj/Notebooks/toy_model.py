# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # **Model 1: Binary classifier - Predict if delay $\geq$ 5 minutes**
# This type of binary classifier is inspired by assignment 3. It is very robust and has a good accuracy, and may be used to predict missing connections in the routing algorithm.
#
#

# +
#preds2 = spark.read.parquet(f"{hadoopFS}/user/com-490/group/U1/pred_delay_gt5min.parquet")

# +
from pyspark.sql.types    import DoubleType, IntegerType, DateType
from pyspark.sql.functions import (
    to_timestamp, unix_timestamp, col,
    hour, month, when, avg
)
from pyspark.sql.window   import Window
from pyspark.ml            import Pipeline
from pyspark.ml.feature    import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import GBTClassifier

df = raw

#casting, parsing
df = (df
  .withColumn("arr_delay_sec", col("arr_delay_sec").cast(DoubleType()))
  .withColumn("arr_time_ts", to_timestamp("arr_time"))
  .withColumn("total_daily_precip", col("total_daily_precip").cast(DoubleType()))
  .withColumn("dow", col("dow").cast(IntegerType()))
)

# label & filters & time features
df = (df
  .withColumn("delay_gt5min", (col("arr_delay_sec") > 300).cast(IntegerType()))   # binary label: >5min late?
  .filter((col("arr_delay_sec") >= -600) & (col("arr_delay_sec") <= 7200))  # drop outliers
  .fillna({"total_daily_precip": 0.0, "rained": "no"})
  .withColumn("hour",  hour("arr_time_ts"))
  .withColumn("month", month("arr_time_ts"))
  .filter(col("dow").between(2, 6))
)

# delay rate per stop over a week window
hist_w = (Window.partitionBy("stop_id").orderBy(col("operating_day").cast("timestamp")).rowsBetween(-7, -1))
df = df.withColumn("hist7wd_delay_rate",avg("delay_gt5min").over(hist_w)).fillna({"hist7wd_delay_rate": 0.0})

# feature columns
categorical_cols = ["stop_id", "site", "dow", "hour", "rained", "type", "stop_name"]
numeric_cols     = ["total_daily_precip", "hist7wd_delay_rate"]

# indexers + one-hots
indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe")for c in categorical_cols]

# assemble into features
assembler = VectorAssembler(inputCols=[c + "_ohe" for c in categorical_cols] + numeric_cols,outputCol="features")

# GBT classifier
gbt = GBTClassifier(labelCol="delay_gt5min",featuresCol="features",maxIter=30,maxDepth=6,stepSize=0.1,subsamplingRate=0.8,seed=42,maxBins=256)


# +
# pipeline + train/test split
pipeline = Pipeline(stages=indexers + encoders + [assembler, gbt])
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_df)

# predictions
preds = model.transform(test_df)

# +
preds2 = spark.read.parquet(f"{hadoopFS}/user/com-490/group/U1/pred_delay_gt5min.parquet")
# overall performence (area under curve
auc_eval = BinaryClassificationEvaluator(labelCol="delay_gt5min", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = auc_eval.evaluate(preds2)
print("AUC = ", auc)

# accuracy, precision, recall, f1
metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]
for m in metrics:
    evaluator = MulticlassClassificationEvaluator(labelCol="delay_gt5min", predictionCol="prediction", metricName=m)
    print(f"{m}: {evaluator.evaluate(preds2)}")
# -

# # Metrics:
# - AUC under ROC =  0.8502439005617006
# - accuracy: 0.9279906041962382
# - weighted precision = 0.9219893472084902
# - weighted recall = 0.9279906041962382
#
# We see that the binary classifier has a very very good metrics. It is an easy task for the model to tell apart "Will the trip be more than 5 minutes late, or not?". This is useful for avoid missing connections.
#
# However, we want a more complex and creative model, which we present below
# This gives us motivation to expand our model over to a more advanced multiclass multinomial classifier to make our model more complex.
