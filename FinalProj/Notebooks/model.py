# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# # Module: Predictive delay model

# %% [markdown]
# ### We model the predictions according to the statistical analysis and EDA carried out in `delay_modeling.ipynb`.
#
#
# The delay $t_d$ is defined simply as: 
# $$
# t_d = t_{\text{arr}} - t_{\text{sched}}
# $$ 
#
# namely the difference between the arrived and scheduled departure and arrival times. We use this to calculate the arrival delay in seconds and separate it into different categories for this models.
#
# A "toy-model" binary classifier that predicts if delay is greater than 5 minutes is presented in `toy_model.py` with its respective validation pipeline.
#
# ## We present to you: 
# ### **Model Multiclass classifier (Multi logistic regression - very fast)** 
# A more sophisticated method using check if delay is between 0-5 minutes (small), 5-10 minutes (medium), and anything beyond is a big delay. Useful for optimizing routes with the routing algorithm. This model can be built very fast in under 2 minutes and has quite good metrics for its complexity. The drawbacks of this model is that the boundaries (5 minutes, 10 minutes), it can be difficult for the model to separate between two classes. This is due to class imbalance. There are many more small delays compared to medium and big delays, however it is generally able to predict correctly. A mitigation of errors is stronger features and more data.

# %%
#full_preds = spark.read.parquet(f"{hadoopFS}/user/com-490/group/U1/multiclass_model.parquet")

# %% [markdown]
# ### Imports
#
# - We import our homemade preprocessing script and any necessities

# %%
import importlib.util
import os
from pyspark.sql.functions import broadcast


# Load preprocessing module
spec = importlib.util.spec_from_file_location("utils", '../scripts/preprocessing.py')
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)
print(preprocess.test())

# %%
# Standard library imports
import os
import sys
import pwd
import time
import json
import re
import warnings
import base64 as b64
from contextlib import closing
from urllib.parse import urlparse
from random import randrange
from itertools import chain

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import trino
from trino.dbapi import connect
from trino.auth import BasicAuthentication, JWTAuthentication
import seaborn as sns

# PySpark core
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, DateType

# PySpark SQL functions
from pyspark.sql import functions as F
from pyspark.sql.functions import rank, col, avg, date_format, count, year,expr, coalesce, lit, to_timestamp,unix_timestamp, hour, month, when,create_map, monotonically_increasing_id

# PySpark ML
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler, StandardScaler,IndexToString
from pyspark.ml.classification import LogisticRegression,GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.functions import vector_to_array

# %% [markdown]
# # Connect to spark and trino

# %%
#setup spark session and trino
username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)
groupName = 'U1'
print(os.getenv('SPARK_HOME'))
print(f"hadoopFSs={hadoopFS}")
print(f"username={username}")
print(f"group={groupName}")

spark = SparkSession\
            .builder\
            .appName(pwd.getpwuid(os.getuid()).pw_name)\
            .config('spark.ui.port', randrange(4040, 4440, 5))\
            .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path)) \
            .config('spark.jars', f'{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar')\
            .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')\
            .config('spark.sql.catalog.iceberg', 'org.apache.iceberg.spark.SparkCatalog')\
            .config('spark.sql.catalog.iceberg.type', 'hadoop')\
            .config('spark.sql.catalog.iceberg.warehouse', f'{hadoopFS}/data/com-490/iceberg/')\
            .config('spark.sql.catalog.spark_catalog', 'org.apache.iceberg.spark.SparkSessionCatalog')\
            .config('spark.sql.catalog.spark_catalog.type', 'hadoop')\
            .config('spark.sql.catalog.spark_catalog.warehouse', f'{hadoopFS}/user/{username}/assignment-3/warehouse')\
            .config("spark.sql.warehouse.dir", f'{hadoopFS}/user/{username}/assignment-3/spark/warehouse')\
            .config('spark.eventLog.gcMetrics.youngGenerationGarbageCollectors', 'G1 Young Generation')\
            .config("spark.executor.memory", "6g")\
            .config("spark.executor.cores", "4")\
            .config("spark.executor.instances", "4")\
            .master('yarn')\
            .getOrCreate()

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")



def getUsername():
    payload = os.environ.get('EPFL_COM490_TOKEN').split('.')[1]
    payload=payload+'=' * (4 - len(payload) % 4)
    obj = json.loads(b64.urlsafe_b64decode(payload))
    if (time.time() > int(obj.get('exp')) - 3600):
        raise Exception('Your credentials have expired, please restart your Jupyter Hub server:'
                        'File>Hub Control Panel, Stop My Server, Start My Server.')
    time_left = int((obj.get('exp') - time.time())/3600)
    return obj.get('sub'), time_left

username, validity_h = getUsername()
hadoopFS = os.environ.get('HADOOP_FS')
namespace = 'iceberg.' + username
sharedNS = 'iceberg.com490_iceberg'

if not re.search('[A-Z][0-9]', groupName):
    raise Exception('Invalid group name {groupName}')

print(f"you are: {username}")
print(f"credentials validity: {validity_h} hours left.")
print(f"shared namespace is: {sharedNS}")
print(f"your namespace is: {namespace}")
print(f"your group is: {groupName}")

trinoAuth = JWTAuthentication(os.environ.get('EPFL_COM490_TOKEN'))
trinoUrl  = urlparse(os.environ.get('TRINO_URL'))
Query=[]

print(f"Warehouse URL: {trinoUrl.scheme}://{trinoUrl.hostname}:{trinoUrl.port}/")

conn = connect(
    host=trinoUrl.hostname,
    port=trinoUrl.port,
    auth=trinoAuth,
    http_scheme=trinoUrl.scheme,
    verify=True
)

print('Connected!')

# %% [markdown]
# # **HOW-TO**
#
# Load from parquet our pre-made table with the greater Lausanne region
#
# `raw = spark.read.parquet(f"{hadoopFS}/user/com 490/group/U1/raw_df_modelling.parquet").dropDuplicates()`
#
#
# If you wish to build your own table, follow instructions in `preprocessing.py`
#
# 1. Fill in UUID as you like. We have built here for the greater Lausanne region (Lausanne + Ouest Lausannois)
#
# 2. Join the historical table together with weather data and geographical data.
#
# 3. The table is now ready for feature engineering and model training pipeline!
#
#

 # %%
 raw = spark.read.parquet(f"{hadoopFS}/user/com 490/group/U1/raw_df_modelling.parquet").dropDuplicates()

# %% [markdown]
# # **Advanced Multiclass classification - Multinomial logistic regression**
# The delay is defined as 
# $$
# t_d = t_{\text{arr}} - t_{\text{sched}}
# $$ 
# i.e. the difference between the scheduled vs actual time of departure and arrival. We classify small, medium and big delays here into multiplce classes with a given probability assigned:
#
# $$
# t_{d}^{small} \in [0, 5] \text{ minutes}\,,
# $$
#
# $$
# t_{d}^{medium} \in [5, 10] \text{ minutes}\,,
# $$
#
# and
#
# $$
# t_{d}^{big} \in [10, 15) \text{ minutes}\,. 
# $$
#
# We limit our model to 15 minutes, because anything beyond is just stupid. We also do not look at early arrivals, because we only care about the delays.
#
# This gives more freedom to make a more favourable prediction on the travel delay. Do we want 5 minutes delay max? Or do we have more time? It is more complex and customizable for the user.

# %%
raw = spark.read.parquet(f"{hadoopFS}/user/com-490/group/U1/raw_df_modelling.parquet")
df_final = raw.dropDuplicates()

# %%
df_final.show(3)

# %%
from pyspark.sql.functions import (
    col, when, avg, stddev, count,
    hour, month, dayofmonth, weekofyear,
    date_format, to_timestamp,
    monotonically_increasing_id,
    create_map, lit
)
from pyspark.sql.types import DoubleType, IntegerType, DateType
from pyspark.sql import Window
from itertools import chain
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder,
    VectorAssembler, IndexToString,
    PolynomialExpansion, Interaction
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array

# preparing the raw data frame
df = (
    df_final
      .withColumn("row_id",            monotonically_increasing_id())
      .withColumn("arr_delay_sec",     col("arr_delay_sec").cast(DoubleType()))
      .withColumn("arr_time_ts",       to_timestamp("arr_time"))
      .withColumn("total_daily_precip",col("total_daily_precip").cast(DoubleType()))
      .withColumn("dow",               col("dow").cast(IntegerType()))
      .fillna({"total_daily_precip":0.0, "rained":"no"})
      # time features
      .withColumn("hour",    hour("arr_time_ts"))
      .withColumn("month",   month("arr_time_ts"))
      .withColumn("dom",     dayofmonth("arr_time_ts"))
      .withColumn("woy",     weekofyear("arr_time_ts"))
      .withColumn("is_raining", when(col("total_daily_precip")>0,1).otherwise(0))
      .withColumn("dow_str", date_format(col("operating_day").cast(DateType()), "E"))
      # precip buckets
      .withColumn("precip_bucket",
          when(col("total_daily_precip")==0,            "none")
         .when(col("total_daily_precip") <= 10.0,        "light")
         .when(col("total_daily_precip") <= 20.0,        "medium")
         .otherwise(                                    "heavy")
      )
      # rush‐hour flags
      .withColumn("peak_morning", when(col("hour").between(7,9),1).otherwise(0))
      .withColumn("peak_evening", when(col("hour").between(16,18),1).otherwise(0))
      # filters
      .filter(col("arr_delay_sec").between(0, 3600))
      .filter(col("dow").between(1,5)))

# re bucketing by quantiles
quantiles = df.stat.approxQuantile("arr_delay_sec", [0.33, 0.66], 0.001)
q33, q66   = quantiles

df = df.withColumn("delay_category_q",
      when(col("arr_delay_sec") <= q33, "small")
     .when(col("arr_delay_sec") <= q66, "medium")
     .otherwise("big"))

# 7-DAY HISTORICAL WINDOW - feautre engineering
hist_w = (
    Window
      .partitionBy("stop_id")
      .orderBy(col("operating_day").cast("timestamp"))
      .rowsBetween(-6, -2))

df = (df
      .withColumn("hist7_mean_delay", avg("arr_delay_sec").over(hist_w))
      .withColumn("hist7_std_delay",  stddev("arr_delay_sec").over(hist_w))
      .withColumn("hist7_count",      count("arr_delay_sec").over(hist_w))
      .na.fill({
          "hist7_mean_delay": 0.0,
          "hist7_std_delay":  0.0,
          "hist7_count":      0}))



### train test split
train_df, test_df = df.randomSplit([0.8,0.2], seed=42)

# stop level target encoding (train only)
stop_stats = train_df.groupBy("stop_id").agg(
    avg("arr_delay_sec").alias("stop_avg_delay"),
    stddev("arr_delay_sec").alias("stop_std_delay")
)
global_avg = stop_stats.select(avg("stop_avg_delay")).first()[0]
global_std = stop_stats.select(avg("stop_std_delay")).first()[0]
stop_stats = stop_stats.na.fill({
    "stop_avg_delay": global_avg,
    "stop_std_delay": global_std
})

# JOIN encoding & LAGGED previous delay
from pyspark.sql.functions import lag
lag_w = Window.partitionBy("stop_id") \
              .orderBy(col("operating_day").cast("timestamp"))

train_df = (train_df
      .join(stop_stats, on="stop_id", how="left")
      .withColumn("prev_delay_sec", lag("arr_delay_sec", 1).over(lag_w))
      .na.fill({"prev_delay_sec": 0.0}))

test_df = (test_df
      .join(stop_stats, on="stop_id", how="left")
      .withColumn("prev_delay_sec", lag("arr_delay_sec", 1).over(lag_w))
      .na.fill({"prev_delay_sec": 0.0}))


# here we find class weights based on the original split
counts = train_df.groupBy("delay_category_q").count().collect()
n_tot  = sum(r["count"] for r in counts)
n_cls  = len(counts)
base_wm = { r["delay_category_q"]: n_tot/(n_cls * r["count"]) for r in counts }

# optional fine tuning of the weights per class if some are not being predicted well
extra_weights = {"big": 1.25, "medium": 1.2, "small": 1.1}
wm = { cls: base_wm[cls] * extra_weights.get(cls,1.0) for cls in base_wm }

w_expr = create_map([lit(x) for x in chain.from_iterable(wm.items())])
train_df = train_df.withColumn("classWeight", w_expr[col("delay_category_q")])
test_df  = test_df .withColumn("classWeight", w_expr[col("delay_category_q")])

#### FEATURE PIPELINE WITH NONLINEAR LIFTS
categorical = ["stop_id","site","dow_str","type","precip_bucket"]
numeric     = ["hour","month","dom","woy","is_raining",
               "peak_morning","peak_evening",
               "hist7_mean_delay","hist7_std_delay","hist7_count",
            "stop_avg_delay","stop_std_delay","prev_delay_sec"]

# indexing & OHE for categoricals
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")for c in categorical]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")for c in categorical]

num_assembler = VectorAssembler(inputCols=numeric, outputCol="num_vec")

# adding quadratic expansion on numerical vector
poly = PolynomialExpansion(inputCol="num_vec", outputCol="poly_vec", degree=2)

# adding interaction term possibly to improve noise
inter = Interaction(inputCols=["stop_id_idx","dow_str_idx"], outputCol="stop_dow_inter")

# final assembly of all features after engineering
feat_assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in categorical] + ["poly_vec","stop_dow_inter"],
    outputCol="features")

feat_pipe = Pipeline(stages=[
    *indexers, *encoders,
    num_assembler, poly, inter,
    feat_assembler])
feat_model = feat_pipe.fit(train_df)
train_ft   = feat_model.transform(train_df)
test_ft    = feat_model.transform(test_df)

# label indexing
label_ix = StringIndexer(
    inputCol="delay_category_q",
    outputCol="label"
).fit(train_ft)
train_ft = label_ix.transform(train_ft)
test_ft  = label_ix.transform(test_ft)

# perform logistic regression
lr = LogisticRegression(
    labelCol="label",
    featuresCol="features",
    weightCol="classWeight",
    family="multinomial",
    maxIter=100,
    regParam=0.002,
    elasticNetParam=0.5)

model   = lr.fit(train_ft)
rawPred = model.transform(test_ft)

# decoding and extracting the exact probabilities
converter = IndexToString(
    inputCol="prediction",
    outputCol="predicted_category_q",
    labels=label_ix.labels)

full_preds = converter.transform(rawPred)

probs = vector_to_array("probability")
for i, lbl in enumerate(label_ix.labels):
    full_preds = full_preds.withColumn(f"prob_{lbl}", probs[i])


# %%
# Save model for validation -> go to notebooks/model_validation.py if you want the separate code

#full_preds.printSchema() # ts broadcast large task binary
#full_preds.write.parquet(f"{hadoopFS}/user/com-490/group/U1/multiclass_model_2.parquet")

# %% [markdown]
# # **Model validation**
#
# ## 1. Per-class metric
#
# We look at the typical metrics like accuracy, weighed precision per class, weighted recall per class and their harmonic mean through the F1 score.

# %%
from pyspark.sql.functions import create_map, lit, col
from itertools import chain
from pyspark.mllib.evaluation import MulticlassMetrics

label_list = label_ix.labels  # eg ["small","medium","big"]
mapping = list(chain.from_iterable(
    (lbl, float(i)) for i, lbl in enumerate(label_list)
))
label_map = create_map(*[lit(x) for x in mapping])

eval_df = full_preds \
  .withColumn("label_idx",    label_map[col("delay_category_q")]) \
  .withColumn("pred_idx",     label_map[col("predicted_category_q")])

pred_label_rdd = eval_df.select("pred_idx","label_idx") \
                        .rdd \
                        .map(lambda r: (r["pred_idx"], r["label_idx"]))

metrics = MulticlassMetrics(pred_label_rdd)

print(f"{'Class':<8} {'Precision':>9} {'Recall':>7} {'F1-score':>9}")
for idx, lbl in enumerate(label_list):
    p = metrics.precision(float(idx))
    r = metrics.recall(float(idx))
    f1 = metrics.fMeasure(float(idx))
    print(f"{lbl:<8} {p:9.3f} {r:7.3f} {f1:9.3f}")

# 6) Print weighted & overall
wp  = metrics.weightedPrecision
wr  = metrics.weightedRecall
wf1 = metrics.weightedFMeasure()
acc = metrics.accuracy

print("\n" +
      f"Weighted Precision: {wp:.3f}\n" +
      f"Weighted Recall:    {wr:.3f}\n" +
      f"Weighted F1-score:  {wf1:.3f}\n" +
      f"Overall Accuracy:   {acc:.3f}")



# %% [markdown]
# ### Our results
# ```
# big          0.499   0.600     0.545
# small        0.491   0.403     0.442
# medium       0.406   0.394     0.400
#
# Weighted Precision: 0.466
# Weighted Recall:    0.467
# Weighted F1-score:  0.463
# Overall Accuracy:   0.467
# ```
#
# This shows that it is pretty bad here. We need further analysis.

# %%
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import when, col

label_to_idx = {lbl: i for i,lbl in enumerate(label_ix.labels)}
evaluator = BinaryClassificationEvaluator(
    rawPredictionCol=None,  
    labelCol="label",
    metricName="areaUnderROC"
)

aucs = {}
for cls in ["small","medium","big"]:
    idx = label_to_idx[cls]
    prob_col = f"prob_{cls}"
    one_vs_rest = full_preds.withColumn(
        "isPos",
        when(col("label") == idx, 1.0).otherwise(0.0)
    )
    evaluator.setRawPredictionCol(prob_col).setLabelCol("isPos")
    auc = evaluator.evaluate(one_vs_rest)
    aucs[cls] = auc

print("One-vs-rest AUCs:", aucs)



# %% [markdown] jp-MarkdownHeadingCollapsed=true
#
# ## AUC OF ROC (One-vs-rest)
# - AUC (Small delay): **0.657**
# - AUC (Medium delay): **0.601**
# - AUC (Big delay): **0.707**
#
# We see that the model most accurately predicts big delays > 10 minutes, which is good to avoid missing transfers and connections and optimizing your route. It is struggling the most to capture between the two boundaries, i.e. the medium delay being 60%.

# %% [markdown]
# ## 2. Confusion matrix
#
# - Where does the classes get mixed up?

# %%
pdf = full_preds.select("delay_category_q","predicted_category_q").toPandas()
cm  = pd.crosstab(pdf.delay_category_q, pdf.predicted_category_q,rownames=['True'], colnames=['Pred'], normalize='index')

plt.figure(figsize=(8,8),dpi=1200)
sns.heatmap(cm, annot=True, fmt=".2f", vmin=0, vmax=1)
plt.title("Confusion Matrix")
plt.ylabel('True Category')
plt.xlabel('Predicted Category')
plt.savefig("../figures/confusion_matrix.png")
plt.show()


# %% [markdown]
# __Ans__: The confusion matrix shows an overall balance across the predictions with the highest percetange being on the diagonal which is good, but not perfect here. However, we want the off-diagonal terms to be as small as possible, which is not the case here. This shows the limitations of our model and probably the biggest cause for this is our "hard" boundaries between the delay categories.
#
# ![Confusion_Matrix](../figures/confusion_matrix.png)

# %% [markdown]
# # Conclusion from model validation
#
#

# %% [markdown]
# “Big” delays are easiest to spot. Highest recall (0.60) and F1 (0.545), and the biggest AUC (0.707). The model reliably flags long delays, but still misclassifies 40% of them as medium or small. “Medium” delays are the hardest class. Lowest precision (0.406), recall (0.394), F1 (0.400) and AUC (0.601). Often confused with either small or big, suggesting the current boundary or features insufficiently separate this middle band. “Small” delays are moderately well-predicted: F1 of 0.442 and AUC of 0.657. Better at catching true small delays (precision ≈0.49) than avoiding false positives (recall $\approx$ 0.40).
#
# **Overall performance is modest**
#
# Accuracy and weighted F1 sit around 0.46–0.47, above random-chance (~0.33) but short of production-grade ($\geq$0.70). The confusion-matrix diagonal is far from perfect, especially in the medium row.
#
# - Feature gaps
#
# The “medium” bucket shows the weakest signal. We need stronger predictors that discriminate that middle band (e.g., schedule-based lateness, head-ways, trip-level patterns).
#
# - Bucket boundaries
#
# The uniform 33/66 quantiles may not align with operationally meaningful thresholds.
#
# - Model choice
#
# Logistic regression can only draw linear decision boundaries in feature space, might need some other model?
#
# - Threshold tuning
#
# Might boost F1 by moving away from arg-max on raw probabilities to class-specific thresholds (especially for medium).
#
# - Additional validation
#
# Can run a k-fold cross-validation to ensure the metrics aren’t overly optimistic/biased by the single train/test split.
#
# Use calibration plots to check how well the predicted probabilities reflect true class likelihoods.
#
#
