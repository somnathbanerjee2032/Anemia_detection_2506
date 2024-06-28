# Databricks notebook source
# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install sparkmeasure

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
tracking_env = solution_config["general_configs"]["tracking_env"]
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

# JOB SPECIFIC PARAMETERS FOR FEATURE PIPELINE
if task.lower() == "fe":
    batch_size = int(solution_config["feature_pipelines_gt"].get("batch_size",500))
    input_table_configs = solution_config["feature_pipelines_gt"]["datalake_configs"]["input_tables"]
    output_table_configs = solution_config["feature_pipelines_gt"]["datalake_configs"]['output_tables']
    is_scheduled = solution_config["feature_pipelines_gt"]["is_scheduled"]
    cron_job_schedule = solution_config["feature_pipelines_gt"].get("cron_job_schedule","0 */10 * ? * *")
else:
    # JOB SPECIFIC PARAMETERS FOR DATA PREP DEPLOYMENT
    batch_size = int(solution_config["data_prep_deployment_gt"].get("batch_size",500))
    input_table_configs = solution_config["data_prep_deployment_gt"]["datalake_configs"]["input_tables"]
    output_table_configs = solution_config["data_prep_deployment_gt"]["datalake_configs"]['output_tables']
    is_scheduled = solution_config["data_prep_deployment_gt"]["is_scheduled"]
    cron_job_schedule = solution_config["data_prep_deployment_gt"].get("cron_job_schedule","0 */10 * ? * *")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### FEATURE ENGINEERING

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### FEATURE ENGINEERING on Ground Truth Data

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

if task.lower() != "fe":
    task_logger_table_name = f"{output_table_configs['output_1']['table']}_task_logger"
    source_1_df ,start_marker,end_marker= get_the_batch_data(output_table_configs["output_1"]["catalog_name"], output_table_configs["output_1"]["schema"], input_table_paths['input_1'], task_logger_table_name, batch_size)
    print(start_marker)
    print(end_marker)
else :
    source_1_df = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")

# COMMAND ----------

if not source_1_df.first():
  dbutils.notebook.exit("No new data is available for DPD, hence exiting the notebook")

# COMMAND ----------

# DBTITLE 1,Load the Data
# source_1_df = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Diagnosis", outputCol="DiagnosisIndex")
indexed_df = indexer.fit(source_1_df).transform(source_1_df)
indexed_df.select("Diagnosis", "DiagnosisIndex").show()

# COMMAND ----------

indexed_df = indexed_df.drop("Diagnosis").withColumnRenamed("DiagnosisIndex", "Diagnosis")

# COMMAND ----------

# if is_scheduled:
#   # CREATE PICKLE FILE
#   pickle_file_path = f"/mnt/FileStore/{output_table_configs['output_1']['schema']}"
#   dbutils.fs.mkdirs(pickle_file_path)
#   print(f"Created directory : {pickle_file_path}")
#   pickle_file_path = f"/dbfs/{pickle_file_path}/{output_table_configs['output_1']['table']}.pickle"

#   # LOAD CACHE IF AVAILABLE
#   try : 
#     with open(pickle_file_path, "rb") as handle:
#         obj_properties = pickle.load(handle)
#         print(f"Instance loaded successfully")
#   except Exception as e:
#     print(f"Exception while loading cache : {e}")
#     obj_properties = {}
#   print(f"Existing Cache : {obj_properties}")

#   if not obj_properties :
#     start_marker = 1
#   elif obj_properties and obj_properties.get("end_marker",0) == 0:
#     start_marker = 1
#   else :
#     start_marker = obj_properties["end_marker"] + 1
#   end_marker = start_marker + batch_size - 1

# else :
#   start_marker = 1
#   end_marker = source_1_df.count()

# print(f"Start Marker : {start_marker}\nEnd Marker : {end_marker}")

# COMMAND ----------

# DBTITLE 1,Perform some feature engineering step.
# indexed_df = indexed_df.filter((F.col("id") >= start_marker) & (F.col("id") <= end_marker))

# COMMAND ----------

# if not source_1_df.first():
#   dbutils.notebook.exit("No new data is available for DPD, hence exiting the notebook")

# COMMAND ----------

from MLCORE_SDK import mlclient
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
            "table_type" : "Ground_Truth",
            "batch_size" : batch_size
        },
        spark = spark,
        tracking_env = env,
        verbose = True,
        )

# COMMAND ----------

indexed_df.display()

# COMMAND ----------

output_1_df = indexed_df.drop('date','timestamp')

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

# DBTITLE 1,Register Ground Truth Transformed Table
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
    table_sub_type="Ground_Truth",
    request_type = task,
    tracking_env = tracking_env,
    batch_size = str(batch_size),
    quartz_cron_expression = cron_job_schedule,
    compute_usage_metrics = compute_metrics,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    verbose = True,
    input_table_names = input_table_paths['input_1'],
    tracking_url = tracking_url,
    )


# COMMAND ----------

if task.lower() != "fe":
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
        table_sub_type="DPD_Batch",
        platform_table_type = "Task_Log",
        tracking_url = tracking_url,
        verbose=True,)

# COMMAND ----------


