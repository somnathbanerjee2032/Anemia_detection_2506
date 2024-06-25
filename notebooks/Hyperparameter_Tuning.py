# Databricks notebook source
# MAGIC %md ## Hyperparameter_Tuning
# MAGIC

# COMMAND ----------

# MAGIC %pip install azure-storage-blob
# MAGIC %pip install protobuf==3.17.2

# COMMAND ----------

from sklearn.model_selection import train_test_split
from hyperopt import tpe, fmin, STATUS_OK, Trials, SparkTrials, space_eval
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.spark import SparkTrials
import numpy as np
import json, time
import pandas as pd
import matplotlib.pyplot as plt

# Disable auto-logging of runs in the mlflow
import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

def get_env_vault_scope():
    """
    Returns env and vault scope
    """
    import json
    env = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    ).split("/")[2]
    try:
        if len(dbutils.fs.ls('dbfs:/FileStore/jars/MLCORE_INIT/vault_check.json')) == 1:
            # if env == "qa":
            #     with open("/dbfs/FileStore/jars/MLCORE_INIT/vault_check_qa.json", "r") as file:
            #         vault_check_data = json.loads(file.read())
            # else:
            with open("/dbfs/FileStore/jars/MLCORE_INIT/vault_check.json", "r") as file:
                vault_check_data = json.loads(file.read())
            if "@" in env:
                return "qa", vault_check_data['client_name']
            return env, vault_check_data['client_name']
        else:
            return env, vault_scope
    except:
        return env, vault_scope

def json_str_to_pythontype(param):
    param = param.replace("'", '"')
    param = json.loads(param)

    return param

def generate_run_notebook_url(job_id, run_id):
    """
    Generates the databricks job run notebook url in runtime
    """
    workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
    workspace_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
    run_notebook_url = f"{workspace_url}/?o={workspace_id}#job/{job_id}/run/{run_id}"
    return run_notebook_url

def fetch_secret_from_dbutils(dbutils, key_name):
    _, vault_scope = get_env_vault_scope()
    return dbutils.secrets.get(scope=vault_scope, key=key_name)
    
def get_gcp_auth_credentials(dbutils):
    import google.auth

    _, vault_scope = get_env_vault_scope()
    client_id = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-id")
    client_secret = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-secret")
    quota_project_id = dbutils.secrets.get(
        scope=vault_scope, key="gcp-api-quota-project-id"
    )
    refresh_token = dbutils.secrets.get(scope=vault_scope, key="gcp-api-refresh-token")
    cred_dict = {
        "client_id": client_id,
        "client_secret": client_secret,
        "quota_project_id": quota_project_id,
        "refresh_token": refresh_token,
        "type": "authorized_user",
    }

    credentials, _ = google.auth.load_credentials_from_dict(cred_dict)

    return credentials

def __upload_blob_to_azure(dbutils, container_name, blob_path, target_path):
    from azure.storage.blob import BlobServiceClient

    try:
        connection_string = fetch_secret_from_dbutils(
            dbutils, "az-api-storage-connection-string"
        )
        # Initialize the BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Get the container client for the blob object
        container_client = blob_service_client.get_container_client(
            container=container_name
        )

        # Upload the blob file
        with open(blob_path, "rb") as data:
            blob_client = container_client.get_blob_client(target_path)
            blob_client.upload_blob(data)

    except Exception as e:
        print(f"Error came while uploading blob object from azure : {e}")
        raise e


def __upload_blob_to_gcp(dbutils, container_name, blob_path, target_path):
    from google.cloud import storage

    try:
        credentials = get_gcp_auth_credentials(dbutils)
        project = fetch_secret_from_dbutils(dbutils, "gcp-api-quota-project-id")

        # Use the obtained credentials to create a client to interact with GCP services
        storage_client = storage.Client(credentials=credentials, project=project)

        bucket_client = storage_client.bucket(container_name)

        # Upload the model file to GCS
        blob = bucket_client.blob(target_path)
        blob.upload_from_filename(blob_path)

    except Exception as e:
        print(f"Error came while uploading blob object from gcp : {e}")
        raise e


def upload_blob_to_cloud(**kwargs):
    """
    Upload the blob from the cloud storage.

    This function will help upload the blob from the cloud storage service like Azure, AWS, GCP.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing operation details, including the `resource_type`.

    Returns
    -------
    The result of the dispatched operation based on the `resource_type`.

    Notes
    -----
    - The function loads the blob object from cloud storage with below parameters :
    - For Azure
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_name': The container where the blob object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - For GCP
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_namecontainer_name': The bucket where the blob  object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - It is essential to provide the correct `resource_type`. Currently supported resources are : az, gcp
    """
    resource_type = kwargs.get("resource_type", None)
    if not resource_type or resource_type in [""]:
        raise Exception("Resource type is not passed or is empty.")

    del kwargs["resource_type"]  # Delete the key since it will not be used by modules

    if resource_type not in ["az", "gcp","azure"]:
        raise Exception(f"Uploading blob object from {resource_type} is not supported.")

    if resource_type.lower() in ["az","azure"]:
        return __upload_blob_to_azure(**kwargs)

    if resource_type.lower() == "gcp":
        return __upload_blob_to_gcp(**kwargs)
def get_gcp_auth_credentials(dbutils):
    import google.auth

    _, vault_scope = get_env_vault_scope()
    client_id = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-id")
    client_secret = dbutils.secrets.get(scope=vault_scope, key="gcp-api-client-secret")
    quota_project_id = dbutils.secrets.get(
        scope=vault_scope, key="gcp-api-quota-project-id"
    )
    refresh_token = dbutils.secrets.get(scope=vault_scope, key="gcp-api-refresh-token")
    cred_dict = {
        "client_id": client_id,
        "client_secret": client_secret,
        "quota_project_id": quota_project_id,
        "refresh_token": refresh_token,
        "type": "authorized_user",
    }

    credentials, _ = google.auth.load_credentials_from_dict(cred_dict)

    return credentials

def __upload_blob_to_azure(dbutils, container_name, blob_path, target_path):
    from azure.storage.blob import BlobServiceClient

    try:
        connection_string = fetch_secret_from_dbutils(
            dbutils, "az-api-storage-connection-string"
        )
        # Initialize the BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Get the container client for the blob object
        container_client = blob_service_client.get_container_client(
            container=container_name
        )

        # Upload the blob file
        with open(blob_path, "rb") as data:
            blob_client = container_client.get_blob_client(target_path)
            blob_client.upload_blob(data)

    except Exception as e:
        print(f"Error came while uploading blob object from azure : {e}")
        raise e


def __upload_blob_to_gcp(dbutils, container_name, blob_path, target_path):
    from google.cloud import storage

    try:
        credentials = get_gcp_auth_credentials(dbutils)
        project = fetch_secret_from_dbutils(dbutils, "gcp-api-quota-project-id")

        # Use the obtained credentials to create a client to interact with GCP services
        storage_client = storage.Client(credentials=credentials, project=project)

        bucket_client = storage_client.bucket(container_name)

        # Upload the model file to GCS
        blob = bucket_client.blob(target_path)
        blob.upload_from_filename(blob_path)

    except Exception as e:
        print(f"Error came while uploading blob object from gcp : {e}")
        raise e


def upload_blob_to_cloud(**kwargs):
    """
    Upload the blob from the cloud storage.

    This function will help upload the blob from the cloud storage service like Azure, AWS, GCP.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing operation details, including the `resource_type`.

    Returns
    -------
    The result of the dispatched operation based on the `resource_type`.

    Notes
    -----
    - The function loads the blob object from cloud storage with below parameters :
    - For Azure
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_name': The container where the blob object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - For GCP
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_namecontainer_name': The bucket where the blob  object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - It is essential to provide the correct `resource_type`. Currently supported resources are : az, gcp
    """
    resource_type = kwargs.get("resource_type", None)
    if not resource_type or resource_type in [""]:
        raise Exception("Resource type is not passed or is empty.")

    del kwargs["resource_type"]  # Delete the key since it will not be used by modules

    if resource_type not in ["az", "gcp","azure"]:
        raise Exception(f"Uploading blob object from {resource_type} is not supported.")

    if resource_type.lower() in ["az","azure"]:
        return __upload_blob_to_azure(**kwargs)

    if resource_type.lower() == "gcp":
        return __upload_blob_to_gcp(**kwargs)
def detect_categorical_cols(df, threshold=5):
    """
    Get the Categorical columns with greater than threshold percentage of unique values.
    This function returns the Categorical columns with the unique values in the column
    greater than the threshold percentage.
    Parameters
    ----------
    df: pyspark.sql.DataFrame
    threshold : int , default = 5
        threshold value in percentage
    Returns
    -------
    report_data : dict
        dictionary containing the Numeric column data.
    """
    df = df.toPandas()
    no_of_rows = df.shape[0]
    possible_cat_cols = (
        df.convert_dtypes()
        .select_dtypes(exclude=[np.datetime64, "float", "float64"])
        .columns.values.tolist()
    )
    temp_series = df[possible_cat_cols].apply(
        lambda col: (len(col.unique()) / no_of_rows) * 100 > threshold
    )
    cat_cols = temp_series[temp_series == False].index.tolist()
    return cat_cols


# COMMAND ----------

import json
def json_str_to_list(param):
    param = param.replace("'", '"')
    param = json.loads(param)

    return param
env, vault_scope = get_env_vault_scope()

# COMMAND ----------

import yaml
from utils import utils

with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
    solution_config = yaml.safe_load(solution_config)

# COMMAND ----------

sdk_session_id = solution_config['sdk_session_id_dev']
# sdk_session_id = "ba6b586a25ab42ab85249e931921fe40"
env = solution_config['ds_environment']
db_name = sdk_session_id

# JOB SPECIFIC PARAMETERS
feature_table_path = solution_config['train']["feature_table_path"]+sdk_session_id
ground_truth_path = solution_config['train']["ground_truth_path"]+sdk_session_id
primary_keys = solution_config['train']["primary_keys"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
train_output_table_name = solution_config['train']["train_output_table_name"]+sdk_session_id
test_size = solution_config['train']["test_size"]
model_name = solution_config['train']["model_name"]+sdk_session_id
model_version = solution_config['train']["model_version"]
primary_metric = solution_config['train']['hyperparameter_tuning']["primary_metric"]
search_range = solution_config['train']["hyperparameter_tuning"]["search_range"]
max_evaluations = solution_config['train']["hyperparameter_tuning"]["max_evaluations"]
stop_early = solution_config['train']["hyperparameter_tuning"]["stop_early"]
run_parallel = solution_config['train']["hyperparameter_tuning"]["run_parallel"]
report_directory = dbutils.widgets.get("report_directory")

# COMMAND ----------

ft_data = spark.sql(f"SELECT * FROM {db_name}.{feature_table_path}")
gt_data = spark.sql(f"SELECT * FROM {db_name}.{ground_truth_path}")

# COMMAND ----------

final_df = ft_data.join(gt_data.select(primary_keys+target_columns), on = primary_keys)

# COMMAND ----------

final_df_pandas = final_df.toPandas()

# COMMAND ----------

final_df_pandas.info()

# COMMAND ----------

final_df_pandas = final_df_pandas.dropna()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(final_df_pandas[feature_columns], final_df_pandas[target_columns], test_size=test_size, random_state = 0)

# COMMAND ----------

X_train = X_train.fillna(X_train.mean())
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

# COMMAND ----------

def early_stop_function(iteration_stop_count=20, percent_increase=0.0):
        def stop_fn(trials, best_loss=None, iteration_no_progress=0):
            if (
                not trials
                or "loss" not in trials.trials[len(trials.trials) - 1]["result"]
            ):
                return False, [best_loss, iteration_no_progress + 1]
            new_loss = trials.trials[len(trials.trials) - 1]["result"]["loss"]
            if best_loss is None:
                return False, [new_loss, iteration_no_progress + 1]
            best_loss_threshold = best_loss - abs(
                best_loss * (percent_increase / 100.0)
            )
            if new_loss < best_loss_threshold:
                best_loss = new_loss
                iteration_no_progress = 0
            else:
                iteration_no_progress += 1
                print(
                    "No progress made: %d iteration on %d. best_loss=%.2f, best_loss_threshold=%.2f, new_loss=%.2f"
                    % (
                        iteration_no_progress,
                        iteration_stop_count,
                        best_loss,
                        best_loss_threshold,
                        new_loss,
                    )
                )

            return (
                iteration_no_progress >= iteration_stop_count,
                [best_loss, iteration_no_progress],
            )

        return stop_fn

def get_trial_data(trials, search_space):
    if not trials:
        return []

    trial_data = []
    trial_id = 0

    for trial in trials.trials:
        trial_id += 1
        trial["result"]["trial"] = trial_id
        trial["result"]["loss"] = (
            0
            if not np.isnan(trial["result"]["loss"])
            and abs(trial["result"]["loss"]) == np.inf
            else trial["result"]["loss"]
        )

        hp_vals = {}
        for hp, hp_val in trial["misc"]["vals"].items():
            hp_vals[hp] = hp_val[0]

        trial["result"]["hyper_parameters"] = space_eval(
            search_space, hp_vals
        )
        trial_data.append(trial["result"])
    return trial_data

def objective(params):
    start_time = time.time()
    metrics = {}
    model = LogisticRegression(**params)
    
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test_np)

    is_multiclass = len(np.unique(y_test)) > 2
    metrics_average = "weighted" if is_multiclass else "binary"

    f1 = f1_score(y_true=y_test, y_pred=y_test_pred, average=metrics_average)
    metrics["f1"] = f1

    precision = precision_score(
        y_true=y_test, y_pred=y_test_pred, average=metrics_average
    )
    metrics["precision"] = precision

    accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    metrics["accuracy"] = accuracy

    recall = recall_score(
        y_true=y_test, y_pred=y_test_pred, average=metrics_average
    )
    metrics["recall"] = recall

    loss = - metrics[primary_metric]
    end_time = time.time()
    
    trail_out_put = {
        "loss": abs(loss),
        "metrics": metrics,
        "status": STATUS_OK,
        "duration" : end_time - start_time,
        "primary_metric":primary_metric,
        "max_evaluations":max_evaluations,
        "early_stopping":stop_early}

    return trail_out_put

def hyperparameter_tuning_with_trials(search_space,max_evals,run_parallel,stop_early):
    if run_parallel:
        trials = SparkTrials(parallelism=4)
    else:
        trials = Trials()

    best_config = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals= max_evals,
            trials=trials,
            early_stop_fn= early_stop_function(10, -0.01)
            if stop_early
            else None,
        )

    return best_config, trials


hyperopt_mapping = {
    bool: hp.choice,
    int: hp.uniform,
    float: hp.uniform,
    str: hp.choice
}

# Converted search space
search_space = {}

for key, value in search_range.items():
    value_type = type(value[0])
    if value_type in hyperopt_mapping:
        if value_type in [bool, str]:
            search_space[key] = hyperopt_mapping[value_type](key, value)
        else:
            search_space[key] = hyperopt_mapping[value_type](key, value[0], value[1])
    else:
        raise ValueError(f"Unsupported type for {key}")

# COMMAND ----------

best_hyperparameters , tuning_trails = hyperparameter_tuning_with_trials( search_space= search_space, max_evals=max_evaluations, run_parallel=run_parallel,stop_early=stop_early)

best_hyperparameters = space_eval(search_space, best_hyperparameters)
tuning_trails_all = get_trial_data(tuning_trails, search_space)


# COMMAND ----------

tuning_trails_all

# COMMAND ----------

df = pd.json_normalize(tuning_trails_all)

# COMMAND ----------

df.display()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
ax.bar(df['trial'], df['loss'], color='blue')
ax.set_xlabel('Trial')
ax.set_ylabel('Loss')
ax.set_title('Trial vs Loss')

# COMMAND ----------

report = {
    "Tuning Trails": {
        "Loss vs Trails": fig
    }

}

# COMMAND ----------

# from tigerml.core.reports import create_report

# COMMAND ----------

# report_name = f"Tuning_trails_{int(time.time())}"
# create_report(
#     report,
#     name=report_name,
#     path="/dbfs/FileStore/MONITORING/Tuning_Trails_report",
#     format=".html",
#     columns= 1 ,
# )

# COMMAND ----------

# from MLCORE_SDK.helpers.auth_helper import get_container_name

# az_container_name = get_container_name(dbutils)

# if cloud_provider.lower() == "gcp" : 
#         container_name=f"{az_container_name}_dev"
# else :
#         container_name = az_container_name

# report_path = f"/dbfs/FileStore/MONITORING/Tuning_Trails_report/{report_name}"

# if cloud_provider.lower() == "databricks_uc" : 
#     dbutils.fs.mv("dbfs:" + report_path.split('/dbfs')[-1],f"dbfs:/mnt/{uc_container_name}/{uc_volume_name}/{report_directory}/Tuning_Trails/BiasReport_V2_{int(time.time())}.html")
# else:
#     upload_blob_to_cloud(
#         container_name=container_name,
#         blob_path=f"{report_path}.html",
#         dbutils = dbutils ,
#         target_path = f"{report_directory}/Tuning_Trails/Tuning_Trails_report{int(time.time())}.html",
#         resource_type = cloud_provider)

# dbutils.fs.rm("dbfs:" + report_path.split('/dbfs')[-1], True)
# print(f"report_directory : {report_directory}")

# COMMAND ----------

blob_path = f"/dbfs/FileStore/Bayada/Tuning_Trails_report_{int(time.time())}.png"
print('Blob Path: ',blob_path)
fig.savefig(blob_path, dpi=100)

target_path = report_directory + f"/Tuning_Trails_report_{int(time.time())}.png"
print('Target path: ',target_path)
from MLCORE_SDK import mlclient
mlclient.log(
operation_type = 'upload_blob_to_cloud',
dbutils = dbutils,
container_name = "mlcore",
blob_path = blob_path,
target_path =  target_path,
resource_type = 'az')

# COMMAND ----------

hp_tuning_result = {
    "best_hyperparameters":best_hyperparameters,
    "tuning_trails":tuning_trails_all,
}

# COMMAND ----------

hp_tuning_result

# COMMAND ----------

dbutils.notebook.exit(json.dumps(hp_tuning_result))

# COMMAND ----------


