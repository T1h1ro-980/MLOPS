import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml

def load_config(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config


def del_emissions(feature_name, data):
    q_low = data[feature_name].quantile(0.25)
    q_high = data[feature_name].quantile(0.75)
    tentacle_length = q_high - q_low
    upper = q_high + 1.5 * tentacle_length
    lower = q_low - 1.5 * tentacle_length
    return data[(data[feature_name] > lower) & (data[feature_name] < upper)]

def preproccessing(config):
    df = pd.read_csv("data/raw.csv")
    variability_threshold = config["preprocessing"]["variability_threshold"]
    

    #Удаление выбросов
    for col in df:
        if col == 'quality': continue
        df = del_emissions(col, df)

    #Удаление неинформативных признаков
    for col in df.columns:
        if df[col].value_counts(normalize=True).max() > variability_threshold:
            df = df.drop(col, axis=1)

    #Новый признак
    df["alcohol_density_ratio"] = df["alcohol"] / df["density"]
    y = df["quality"]
    X = df.drop("quality", axis=1)

    #Масшатбирование
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df["quality"] = y.values

    scaled_df.to_csv("data/scaled_df.csv", index=False)

if __name__ == "__main__":
    config = load_config("./src/params.yaml")
    preproccessing(config)