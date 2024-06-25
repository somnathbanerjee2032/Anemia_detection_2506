# Databricks notebook source
# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics, TaskMetrics
taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

try : 
    env = dbutils.widgets.get("env")
    task = dbutils.widgets.get("task")
except :
    env, task = "dev","fe"
print(f"Input environment : {env}")
print(f"Input task : {task}")

# COMMAND ----------

# DBTITLE 1,Load the YAML config
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

# GENERAL PARAMETERS
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

# JOB SPECIFIC PARAMETERS FOR FEATURE PIPELINE
if task.lower() == "fe":
    batch_size = int(solution_config["feature_pipelines_ft"].get("batch_size",500))
    input_table_configs = solution_config["feature_pipelines_ft"]["datalake_configs"]["input_tables"]
    output_table_configs = solution_config["feature_pipelines_ft"]["datalake_configs"]['output_tables']
    is_scheduled = solution_config["feature_pipelines_ft"]["is_scheduled"]
    cron_job_schedule = solution_config["feature_pipelines_ft"].get("cron_job_schedule","0 */10 * ? * *")
else:
    # JOB SPECIFIC PARAMETERS FOR DATA PREP DEPLOYMENT
    batch_size = int(solution_config["data_prep_deployment_ft"].get("batch_size",500))
    input_table_configs = solution_config["data_prep_deployment_ft"]["datalake_configs"]["input_tables"]
    output_table_configs = solution_config["data_prep_deployment_ft"]["datalake_configs"]['output_tables']
    is_scheduled = solution_config["data_prep_deployment_ft"]["is_scheduled"]
    cron_job_schedule = solution_config["data_prep_deployment_ft"].get("cron_job_schedule","0 */10 * ? * *")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### FEATURE ENGINEERING

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### FEATURE ENGINEERING on Feature Data

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

# DBTITLE 1,Load the data
input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

source_1_df = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")

# COMMAND ----------

if is_scheduled:
  pickle_file_path = f"/mnt/FileStore/{output_table_configs['output_1']['schema']}"
  dbutils.fs.mkdirs(pickle_file_path)
  print(f"Created directory : {pickle_file_path}")
  pickle_file_path = f"/dbfs/{pickle_file_path}/{output_table_configs['output_1']['table']}.pickle"

  try : 
    with open(pickle_file_path, "rb") as handle:
        obj_properties = pickle.load(handle)
        print(f"Instance loaded successfully")
  except Exception as e:
    print(f"Exception while loading cache : {e}")
    obj_properties = {}
  print(f"Existing Cache : {obj_properties}")

  if not obj_properties :
    start_marker = 1
  elif obj_properties and obj_properties.get("end_marker",0) == 0:
    start_marker = 1
  else :
    start_marker = obj_properties["end_marker"] + 1
  end_marker = start_marker + batch_size - 1
else :
  start_marker = 1
  end_marker = source_1_df.count()

print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

# DBTITLE 1,Perform some feature engineering step. 
source_1_df = source_1_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

# DBTITLE 1,Exit the job if there is no new data
if not source_1_df.first():
  dbutils.notebook.exit("No new data is available for DPD, hence exiting the notebook")

# COMMAND ----------

if task.lower() != "fe":
    # Calling job run add for DPD job runs
    mlclient.log(
        operation_type="job_run_add", 
        session_id = sdk_session_id, 
        dbutils = dbutils, 
        request_type = task, 
        job_config = 
        {
            "table_name" : output_table_configs["output_1"]["table"],
            "table_type" : "Source",
            "batch_size" : batch_size
        },
        tracking_env = env,
        spark = spark,
        verbose = True,
        )

# COMMAND ----------

data = source_1_df.toPandas()

# COMMAND ----------

import pandas as pd
import numpy as np


# COMMAND ----------

data['mean_WBC_RBC'] = data[['WBC', 'RBC']].mean(axis=1)
data['std_WBC_RBC'] = data[['WBC', 'RBC']].std(axis=1)

# COMMAND ----------

data.drop(columns=['id'])

# COMMAND ----------

# # Extracting registration year from the date
# data["Reg_year"] = data["Dt_Customer"].apply(lambda x: x.year)

# # Extracting registration quarter from the date
# data["Reg_quarter"] = data["Dt_Customer"].apply(lambda x: x.quarter)

# # Extracting registration month from the date
# data["Reg_month"] = data["Dt_Customer"].apply(lambda x: x.month)

# # Extracting registration week from the date
# data["Reg_week"] = data["Dt_Customer"].apply(lambda x: x.day // 7)

# COMMAND ----------

data.head()

# COMMAND ----------

# data["Education"] = data["Education"].replace("2n Cycle", "Master")

# COMMAND ----------

# data["Marital_Status"] = data["Marital_Status"].replace(["YOLO", "Alone", "Absurd"], "Single")
# data["Marital_Status"] = data["Marital_Status"].replace(["Together"], "Married")

# COMMAND ----------

# data["Total_Amount_Spent"] = data[
#     [
#         "MntWines",
#         "MntFruits",
#         "MntMeatProducts",
#         "MntFishProducts",
#         "MntSweetProducts",
#         "MntGoldProds",
#     ]
# ].sum(axis=1)

# COMMAND ----------

# data[data["Income"] > 200000]

# COMMAND ----------

# data.drop(index=data[data.Income > 200000].index, inplace=True)

# COMMAND ----------

# data.MntMeatProducts.nlargest(10)

# COMMAND ----------

# data[data["MntMeatProducts"] > 1580]

# COMMAND ----------

# data["MntMeatProducts"].clip(upper=984, inplace=True)

# COMMAND ----------

# data[data["MntSweetProducts"] > 200]

# COMMAND ----------

# data["MntSweetProducts"].clip(upper=198, inplace=True)

# COMMAND ----------

# data[data["MntGoldProds"] > 250]

# COMMAND ----------

# data["MntGoldProds"].clip(upper=250, inplace=True)

# COMMAND ----------

# data[data["NumWebPurchases"] > 15]

# COMMAND ----------

# data["NumWebPurchases"].clip(upper=11, inplace=True)

# COMMAND ----------

# data[data["NumCatalogPurchases"] > 15]

# COMMAND ----------

# data["NumCatalogPurchases"].clip(upper=11, inplace=True)

# COMMAND ----------

data.head(2)

# COMMAND ----------

# data.drop(
#     columns=[
#         "Year_Birth",
#         "Dt_Customer",
#         "Reg_quarter",
#         "Total_Amount_Spent",
#     ],
#     inplace=True,
# )

# COMMAND ----------

# categ = ['Education','Marital_Status']

# from sklearn.preprocessing import LabelEncoder
# # Encode Categorical Columns
# le = LabelEncoder()
# data[categ] = data[categ].apply(le.fit_transform)

# COMMAND ----------

output_1_df = spark.createDataFrame(data)

# COMMAND ----------

output_1_df.display()

# COMMAND ----------

output_1_df = output_1_df.drop('date','timestamp')

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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

# DBTITLE 1,Adding Timestamp and Date Features to a Source 1
now = datetime.now()
date = now.strftime("%m-%d-%Y")
output_1_df = output_1_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
output_1_df = output_1_df.withColumn("date", F.lit(date))
output_1_df = output_1_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in output_1_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  output_1_df = output_1_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

# DBTITLE 1,writing to output_1
db_name = output_table_configs["output_1"]["schema"]
table_name = output_table_configs["output_1"]["table"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none": 
  spark.sql(f"USE CATALOG {catalog_name}")


# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

output_1_df.createOrReplaceTempView(table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

if catalog_name and catalog_name.lower() != "none": 
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

# MAGIC %md
# MAGIC
# MAGIC ### REGISTER THE FEATURES ON MLCORE
# MAGIC

# COMMAND ----------

if task == "FE" : 
    table_description = "This is the transformed table for marketing campaign usecase build for a retail store. The table contains feature columns post feature engineering to build a classification model"
else: 
    table_description = "This is the DPD source table for marketing campaign usecase build for a retail store with data flowing in batches. This table contains feature columns after preprocessing in batches"

# COMMAND ----------

# DBTITLE 1,Register Features Transformed Table
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = output_table_configs["output_1"]["table"],
    num_rows = output_1_df.count(),
    cols = output_1_df.columns,
    column_datatype = output_1_df.dtypes,
    table_schema = output_1_df.schema,
    primary_keys = output_table_configs["output_1"]["primary_keys"],
    table_path = output_1_table_path,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal" ,
    table_sub_type="Source",
    request_type = task,
    tracking_env = env,
    batch_size = str(batch_size),
    quartz_cron_expression = cron_job_schedule,
    compute_usage_metrics = compute_metrics,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    verbose = True,
    input_table_names=[input_table_paths['input_1']],
    )


# COMMAND ----------

if is_scheduled:
  obj_properties['end_marker'] = end_marker
  with open(pickle_file_path, "wb") as handle:
      pickle.dump(obj_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Instance successfully saved successfully")

# COMMAND ----------


