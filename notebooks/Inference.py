# Databricks notebook source
# %pip install databricks-feature-store
# %pip install /dbfs/FileStore/sdk/dev/MLCoreSDK-0.4.5-py3-none-any.whl --force-reinstall
%pip install sparkmeasure

# COMMAND ----------

# MAGIC %pip install numpy==1.19.1
# MAGIC %pip install pandas==1.0.5

# COMMAND ----------

from sparkmeasure import StageMetrics, TaskMetrics

taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Imports
import yaml
import ast
import pickle
from MLCORE_SDK import mlclient
from pyspark.sql import functions as F

try:
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = ast.literal_eval(solution_config)
except Exception as e:
    print(e)
    with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Model & Table parameters
# MAGIC

# COMMAND ----------

# GENERAL PARAMETERS
tracking_env = solution_config["general_configs"]["tracking_env"]
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

# JOB SPECIFIC PARAMETERS FOR INFERENCE
input_table_configs = solution_config["inference"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["inference"]["datalake_configs"]['output_tables']
model_configs = solution_config["train"]["model_configs"]
feature_columns = solution_config['train']["feature_columns"]
is_scheduled = solution_config["inference"]["is_scheduled"]
batch_size = int(solution_config["inference"].get("batch_size",500))
cron_job_schedule = solution_config["inference"].get("cron_job_schedule","0 */10 * ? * *")


# COMMAND ----------

def get_name_space(table_config):
    data_objects = {}
    for table_name, config in table_config.items() : 
        catalog_name = config.get("catalog_name", None)
        schema = config.get("schema", None)
        table = config.get("table", None)

        if catalog_name and catalog_name.lower() != "none":
            table_path = f"{catalog_name}.{schema}.{table}"
        else :
            table_path = f"{schema}.{table}"

        data_objects[table_name] = table_path
    
    return data_objects

# COMMAND ----------

input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

# Table Exists or Not
def table_already_created(catalog_name, db_name, table_name):
    if catalog_name:
        db_name = f"{catalog_name}.{db_name}"
    table_exists = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]
    return any(table_exists)

# Create SQL Query
def get_task_logger(catalog_name, db_name, table_name):
    if table_already_created(catalog_name, db_name, table_name): 
        result = spark.sql(f"SELECT * FROM {catalog_name}.{db_name}.{table_name} ORDER BY timestamp desc LIMIT 1").collect()
        if result:
            task_logger = result[0].asDict()
            start_marker = task_logger["start_marker"]
            end_marker = task_logger["end_marker"]
            return start_marker, end_marker
    return 0, 0

def get_the_batch_data(catalog_name, db_name, source_data_path, task_logger_table_name, batch_size):
    start_marker, end_marker = get_task_logger(catalog_name, db_name, task_logger_table_name)
    query = f"SELECT * FROM {source_data_path}"
    if start_marker and end_marker:
        query += f" WHERE {generate_filter_condition(start_marker, end_marker)}"
    query += f" LIMIT {batch_size}"
    print(f"SQL QUERY  : {query}")
    filtered_df = spark.sql(query)
    return filtered_df, start_marker, end_marker

def generate_filter_condition(start_marker, end_marker):
    filter_column = 'timestamp'  # Replace with the actual column name
    filter_condition = f"{filter_column} > {end_marker}"
    return filter_condition

# Update Task Logger - Z-Ordered
def update_task_logger(catalog_name, db_name, task_logger_table_name, end_marker, batch_size):
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
    from pyspark.sql.window import Window
    import time
    from datetime import datetime

    start_marker = end_marker + 1
    end_marker = end_marker + batch_size
    print(f"start_marker : {start_marker}")
    print(f"end_marker : {end_marker}")
    # Determination of table name on which markers have been calculated

    # Updating task log with new metadata
    schema = StructType(
        [
            StructField("start_marker", IntegerType(), True),
            StructField("end_marker", IntegerType(), True),
            StructField("table_name", StringType(), True),
        ]
    )
    df_column_name = ["start_marker", "end_marker", "table_name"]
    df_record = [(int(start_marker), int(end_marker), task_logger_table_name)]
    df_task = spark.createDataFrame(df_record, schema=schema)
    now = datetime.now()
    date = now.strftime("%m-%d-%Y")
    df_task = df_task.withColumn("timestamp", F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"))
    df_task = df_task.withColumn("date", F.lit(date))
    df_task = df_task.withColumn("date", F.to_date(F.col("date")))
    
    if "id" not in df_task.columns:
        window = Window.orderBy(F.monotonically_increasing_id())
        df_task = df_task.withColumn("id", F.row_number().over(window))
    
    df_task.createOrReplaceTempView(task_logger_table_name)
    
    if table_already_created(catalog_name, db_name, task_logger_table_name):
        if catalog_name and catalog_name.lower() != "none":
            spark.sql(f"USE CATALOG {catalog_name}")
        spark.sql(f"INSERT INTO {db_name}.{task_logger_table_name} SELECT * FROM {task_logger_table_name}")
    else:
        if catalog_name and catalog_name.lower() != "none":
            spark.sql(f"USE CATALOG {catalog_name}")
        spark.sql(f"CREATE TABLE IF NOT EXISTS {db_name}.{task_logger_table_name} AS SELECT * FROM {task_logger_table_name}")
    
    return df_task

# COMMAND ----------

task_logger_table_name = f"{output_table_configs['output_1']['table']}_task_logger"

# COMMAND ----------

features_df,start_marker,end_marker = get_the_batch_data(output_table_configs["output_1"]["catalog_name"], output_table_configs["output_1"]["schema"], input_table_paths['input_1'], task_logger_table_name, batch_size)

gt_df,start_marker,end_marker = get_the_batch_data(output_table_configs["output_1"]["catalog_name"], output_table_configs["output_1"]["schema"], input_table_paths['input_2'], task_logger_table_name, batch_size)

print(start_marker)
print(end_marker)

# COMMAND ----------

features_df.count()

# COMMAND ----------

features_df.display()

# COMMAND ----------

if not features_df.first():
  dbutils.notebook.exit("No data is available for inference, hence exiting the notebook")

# COMMAND ----------

if input_table_configs["input_1"]["catalog_name"]:
    feature_table_path = input_table_paths["input_1"]
else:
    feature_table_path = spark.sql(f"desc formatted {input_table_paths['input_1']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

if input_table_configs["input_2"]["catalog_name"]:
    gt_table_path = input_table_paths["input_2"]
else :   
    gt_table_path = spark.sql(f"desc formatted {input_table_paths['input_2']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
print(feature_table_path, gt_table_path)

# COMMAND ----------

catalog_name = solution_config["train"]["model_configs"]["model_registry_params"]["catalog_name"]
schema_name = solution_config["train"]["model_configs"]["model_registry_params"]["schema_name"]
model_name = solution_config["train"]["model_configs"]["model_params"]["model_name"]
Model_name = f"{catalog_name}.{schema_name}.{model_name}"

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# COMMAND ----------

def get_latest_model_version(model_name):
  mlflow.set_registry_uri("databricks-uc")
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

# COMMAND ----------

if model_configs["use_latest_version"]:
  model_version = str(get_latest_model_version(model_name=Model_name))

# COMMAND ----------

mlclient.log(
    operation_type="job_run_add", 
    session_id = sdk_session_id, 
    dbutils = dbutils, 
    request_type = "inference", 
    job_config = {
        "table_name" : output_table_configs['output_1']['table'],
        "table_type" : "Inference_Output",
        "batch_size" : batch_size,
        "output_table_name" : output_table_configs['output_1']['table'],
        "model_name" : Model_name,
        "model_version" : model_version,
        "feature_table_path" : feature_table_path,
        "ground_truth_table_path" : gt_table_path,
        "env" : tracking_env,
        "quartz_cron_expression" : cron_job_schedule
    },
    spark = spark,
    tracking_env = tracking_env,
    tracking_url = tracking_url,
    verbose = True,
)

# COMMAND ----------

ground_truth = gt_df.select([input_table_configs["input_2"]["primary_keys"]] + target_columns).toPandas()
transformed_features_df = features_df.toPandas()
inference_df = transformed_features_df[feature_columns]
display(inference_df)

# COMMAND ----------

def load_model(model_name):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

# COMMAND ----------

loaded_model = load_model(Model_name)

# COMMAND ----------

# inference_df_np = inference_df.to_numpy()
predictions = loaded_model.predict(inference_df.values)
type(predictions)

# COMMAND ----------

transformed_features_df["prediction"] = predictions
transformed_features_df = pd.merge(transformed_features_df,ground_truth, on=input_table_configs["input_1"]["primary_keys"], how='inner')
output_table = spark.createDataFrame(transformed_features_df)

# COMMAND ----------

output_table.display()

# COMMAND ----------

# features_df = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")

# COMMAND ----------

# pickle_file_path = f"/mnt/FileStore/{output_table_configs['output_1']['schema']}"
# dbutils.fs.mkdirs(pickle_file_path)
# print(f"Created directory : {pickle_file_path}")
# pickle_file_path = f"/dbfs/{pickle_file_path}/{output_table_configs['output_1']['table']}.pickle"

# COMMAND ----------

# try : 
#   with open(pickle_file_path, "rb") as handle:
#       obj_properties = pickle.load(handle)
#       print(f"Instance loaded successfully")
# except Exception as e:
#   print(f"Exception while loading cache : {e}")
#   obj_properties = {}
# print(f"Existing Cache : {obj_properties}")

# if not obj_properties :
#   start_marker = 1
# elif obj_properties and obj_properties.get("end_marker",0) == 0:
#   start_marker = 1
# else :
#   start_marker = obj_properties["end_marker"] + 1
# end_marker = start_marker + batch_size - 1

# print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

# FT_DF = features_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

# if not FT_DF.first():
#   dbutils.notebook.exit("No data is available for inference, hence exiting the notebook")

# COMMAND ----------

# from MLCORE_SDK.configs.secret_mapping import SECRET_MAPPING
# from MLCORE_SDK.helpers.auth_helper import get_env_vault_scope
# import mlflow
# from mlflow.tracking import MlflowClient
# import os

# def set_mlflow_registry(model_configs):
#     _,vault_scope = get_env_vault_scope(dbutils)
#     mlflow_uri = model_configs.get("model_registry_params").get("host_url")
#     tracking_env = model_configs.get("model_registry_params").get("tracking_env")

#     mlflow.set_tracking_uri(mlflow_uri)
#     mlflow_token = (dbutils.secrets.get(vault_scope,SECRET_MAPPING.get(f"databricks-access-token-{tracking_env}", ""),))
#     os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token

# def get_latest_model_version(model_configs):
#     mlflow_uri = model_configs.get("model_registry_params").get("host_url")
#     model_name = model_configs.get("model_params").get("model_name")
#     mlflow.set_registry_uri(mlflow_uri)
#     client = MlflowClient()
#     x = client.get_latest_versions(model_name)
#     model_version = x[0].version
#     return model_version


# def load_model(model_configs):
#     set_mlflow_registry(model_configs)
#     model_version = get_latest_model_version(model_configs)
#     return mlflow.sklearn.load_model(model_uri=f"models:/{model_configs['model_params']['model_name']}/{model_version}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load Model

# COMMAND ----------

# loaded_model = load_model(model_configs)

# COMMAND ----------

# mlclient.log(
#     operation_type="job_run_add", 
#     session_id = sdk_session_id, 
#     dbutils = dbutils, 
#     request_type = "inference", 
#     job_config = {
#         "table_name" : output_table_configs['output_1']['table'],
#         "table_type" : "Inference_Output",
#         "batch_size" : batch_size,
#         "output_table_name" : output_table_configs['output_1']['table'],
#         "model_name" : model_configs["model_params"]["model_name"],
#         "model_version" : get_latest_model_version(model_configs),
#         "feature_table_path" : f"{input_table_paths['input_1']}",
#         "ground_truth_table_path" : f"{input_table_paths['input_2']}",
#         "env" : env,
#         "quartz_cron_expression" : cron_job_schedule
#     },
#     spark = spark,
#     tracking_env = env,
#     verbose = True,
# )

# COMMAND ----------

# tranformed_features_df = FT_DF.toPandas()
# tranformed_features_df.dropna(inplace=True)
# tranformed_features_df.shape

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Load Inference Features Data

# COMMAND ----------

from MLCORE_SDK.helpers.mlc_job_helper import get_job_id_run_id

job_id, run_id = get_job_id_run_id(dbutils)

output_table = output_table.withColumnRenamed(target_columns[0],"ground_truth_value")
output_table = output_table.withColumn("acceptance_status",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_time",F.lit(None).cast("long"))
output_table = output_table.withColumn("accepted_by_id",F.lit(None).cast("string"))
output_table = output_table.withColumn("accepted_by_name",F.lit(None).cast("string"))
output_table = output_table.withColumn("moderated_value",F.lit(None).cast("double"))
output_table = output_table.withColumn("inference_job_id",F.lit(job_id).cast("string"))
output_table = output_table.withColumn("inference_run_id",F.lit(run_id).cast("string"))

# COMMAND ----------

output_table = output_table.drop('date','id','timestamp')

# COMMAND ----------

output_table.display()

# COMMAND ----------

from datetime import datetime
from pyspark.sql import (
    types as DT,
    functions as F,
    Window
)
def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

now = datetime.now()
date = now.strftime("%m-%d-%Y")
output_table = output_table.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
output_table = output_table.withColumn("date", F.lit(date))
output_table = output_table.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in output_table.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  output_table = output_table.withColumn("id", F.row_number().over(window))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

now = datetime.now()
date = now.strftime("%m-%d-%Y")
output_table = output_table.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
output_table = output_table.withColumn("date", F.lit(date))
output_table = output_table.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in output_table.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  output_table = output_table.withColumn("id", F.row_number().over(window))

# COMMAND ----------

db_name = output_table_configs["output_1"]["schema"]
table_name = output_table_configs["output_1"]["table"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none":
  spark.sql(f"USE CATALOG {catalog_name}")
else:
  spark.sql(f"USE CATALOG hive_metastore")

# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

output_table.createOrReplaceTempView(table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

if catalog_name:
  output_1_table_path = output_path
else:
  output_1_table_path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Features Hive Path : {output_1_table_path}")

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

mlclient.log(operation_type = "register_inference",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    output_table_name=output_table_configs["output_1"]["table"],
    output_table_path=output_1_table_path,
    feature_table_path=feature_table_path,
    ground_truth_table_path=gt_table_path,
    model_name=Model_name,
    model_version=model_version,
    num_rows=output_table.count(),
    cols=output_table.columns,
    column_datatype = output_table.dtypes,
    table_schema = output_table.schema,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    batch_size = batch_size,
    tracking_env = tracking_env,
    compute_usage_metrics = compute_metrics,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    # register_in_feature_store=True,
    tracking_url = tracking_url,
    verbose=True)

# COMMAND ----------

df_task = update_task_logger(output_table_configs["output_1"]["catalog_name"], output_table_configs["output_1"]["schema"],task_logger_table_name,end_marker, batch_size)

logger_table_path=f"{catalog_name}.{db_name}.{task_logger_table_name}"
if catalog_name and catalog_name.lower() != "none": 
    task_logger_table_path = logger_table_path
else:
    task_logger_table_path = spark.sql(f"desc {logger_table_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

# Register Task Logger Table in MLCore
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = task_logger_table_name,
    num_rows = df_task.count(),
    tracking_env = tracking_env,
    cols = df_task.columns,
    column_datatype = df_task.dtypes,
    table_schema = df_task.schema,
    primary_keys = ["id"],
    table_path = task_logger_table_path,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    table_sub_type="Inference_Batch",
    platform_table_type = "Task_Log",
    tracking_url = tracking_url,
    verbose=True,)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict

# COMMAND ----------

# predictions = loaded_model.predict(inference_df)
# type(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Output Table

# COMMAND ----------

# tranformed_features_df["prediction"] = predictions
# output_table = spark.createDataFrame(tranformed_features_df)

# COMMAND ----------

# output_table = output_table.withColumn("acceptance_status",F.lit(None).cast("string"))
# output_table = output_table.withColumn("accepted_time",F.lit(None).cast("long"))
# output_table = output_table.withColumn("accepted_by_id",F.lit(None).cast("string"))
# output_table = output_table.withColumn("accepted_by_name",F.lit(None).cast("string"))
# output_table = output_table.withColumn("moderated_value",F.lit(None).cast("double"))

# COMMAND ----------

# output_table = output_table.drop('date','id','timestamp')

# COMMAND ----------

# from datetime import datetime
# from pyspark.sql import functions as F
# from pyspark.sql.window import Window

# def to_date_(col):
#     """
#     Checks col row-wise and returns first date format which returns non-null output for the respective column value
#     """
#     formats=(
#              "MM-dd-yyyy", "dd-MM-yyyy",
#              "MM/dd/yyyy", "yyyy-MM-dd", 
#              "M/d/yyyy", "M/dd/yyyy",
#              "MM/dd/yy", "MM.dd.yyyy",
#              "dd.MM.yyyy", "yyyy-MM-dd",
#              "yyyy-dd-MM"
#             )
#     return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

# DBTITLE 1,Adding Timestamp and Date Features to a Source 1
# now = datetime.now()
# date = now.strftime("%m-%d-%Y")
# output_table = output_table.withColumn(
#     "timestamp",
#     F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
# )
# output_table = output_table.withColumn("date", F.lit(date))
# output_table = output_table.withColumn("date", to_date_(F.col("date")))

# # ADD A MONOTONICALLY INREASING COLUMN
# if "id" not in output_table.columns : 
#   window = Window.orderBy(F.monotonically_increasing_id())
#   output_table = output_table.withColumn("id", F.row_number().over(window))

# COMMAND ----------

# DBTITLE 1,writing to output_1
# db_name = output_table_configs["output_1"]["schema"]
# table_name = output_table_configs["output_1"]["table"]
# catalog_name = output_table_configs["output_1"]["catalog_name"]
# output_path = output_table_paths["output_1"]

# # Get the catalog name from the table name
# if catalog_name and catalog_name.lower() != "none":
#   spark.sql(f"USE CATALOG {catalog_name}")


# # Create the database if it does not exist
# spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
# print(f"HIVE METASTORE DATABASE NAME : {db_name}")

# output_table.createOrReplaceTempView(table_name)

# feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

# if not any(feature_table_exist):
#   print(f"CREATING SOURCE TABLE")
#   spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
# else :
#   print(F"UPDATING SOURCE TABLE")
#   spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

# if catalog_name and catalog_name.lower() != "none":
#   output_1_table_path = output_path
# else:
#   output_1_table_path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

# print(f"Features Hive Path : {output_1_table_path}")

# COMMAND ----------

# if input_table_configs["input_1"]["catalog_name"]:
#     feature_table_path = input_table_paths["input_1"]
# else:
#     feature_table_path = spark.sql(f"desc formatted {input_table_paths['input_1']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

# if input_table_configs["input_2"]["catalog_name"]:
#     gt_table_path = input_table_paths["input_2"]
# gt_table_path = spark.sql(f"desc formatted {input_table_paths['input_2']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

# print(feature_table_path, gt_table_path)

# COMMAND ----------

# stagemetrics.end()
# taskmetrics.end()

# stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
# task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

# compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

# compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
# compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Register Inference artifacts

# COMMAND ----------

# mlclient.log(operation_type = "register_inference",
#     sdk_session_id = sdk_session_id,
#     dbutils = dbutils,
#     spark = spark,
#     output_table_name=output_table_configs["output_1"]["table"],
#     output_table_path=output_1_table_path,
#     feature_table_path=feature_table_path,
#     ground_truth_table_path=gt_table_path,
#     model_name=model_configs["model_params"]["model_name"],
#     model_version=get_latest_model_version(model_configs),
#     num_rows=output_table.count(),
#     cols=output_table.columns,
#     column_datatype = output_table.dtypes,
#     table_schema = output_table.schema,
#     table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
#     batch_size = batch_size,
#     tracking_env = env,
#     compute_usage_metrics = compute_metrics,
#     taskmetrics=taskmetrics,
#     stagemetrics=stagemetrics,
#     # register_in_feature_store=True,
#     verbose=True)

# COMMAND ----------

# obj_properties['end_marker'] = end_marker
# with open(pickle_file_path, "wb") as handle:
#     pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"Instance successfully saved successfully")
