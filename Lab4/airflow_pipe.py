from ucimlrepo import fetch_ucirepo 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta, datetime
from train_model import train 

def download_data():
    wine_quality = fetch_ucirepo(id=186) 
    X = wine_quality.data.features
    y = wine_quality.data.targets
    df = pd.concat([X, y], axis=1)
    df = df.drop_duplicates() 
    df.to_csv("wine_quality.csv")
    return df

def preprocessing():
    df = pd.read_csv("wine_quality.csv")
    def del_emissions(feature_name, data):
        q_low = df[feature_name].quantile(0.25)
        q_high = df[feature_name].quantile(0.75)

        tentacle_length = q_high - q_low
        upper_tentacle = q_high + 1.5 * tentacle_length
        lower_tentacle = q_low - 1.5 * tentacle_length

        new_data = data[(data[feature_name] > lower_tentacle) &
                        (data[feature_name] < upper_tentacle)]

        #sns.boxplot(y = new_data[feature_name])
        #plt.show()
        return new_data

    for col in df:
        if col == 'quality':
            continue
        df = del_emissions(col, df)

    for col in df:
        variability = df[col].value_counts(normalize = True).max()
        if variability > 0.8:
            df = df.drop(col, axis = 1)

    df["alcohol_density_ratio"] = df["alcohol"] / df["density"]

    y = df["quality"]
    X = df.drop("quality", axis = 1)

    features_names = list(X.columns)
    scaler = MinMaxScaler()
    scaler.fit(X)
    scaled_X = scaler.transform(X)
    scaled_df = pd.DataFrame(scaled_X, columns = features_names)
    scaled_df["quality"] = y.values
    scaled_df.to_csv("scaled_df.csv")


dag_wine = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
#    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(python_callable=download_data, task_id = "download_dataset", dag = dag_wine)
preprocessing_task = PythonOperator(python_callable=preprocessing, task_id = "preprocessing", dag = dag_wine)
train_task = PythonOperator(python_callable=train, task_id = "train_model", dag = dag_wine)
download_task >> preprocessing_task >> train_task