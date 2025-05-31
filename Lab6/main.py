from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from typing import List, Optional
import uvicorn
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
import yaml
from ucimlrepo import fetch_ucirepo 
#-----------------------------------------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели (замените путь на свой)
try:
    with open("best_wine.pkl", "rb") as f:
        model = joblib.load(f)
    logger.info("Model loaded successfully")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    # raise

app = FastAPI(title="Wine Quality")

#-----------------------------------------------------------------------------------------------------------------



def download_dataset():
    wine_quality = fetch_ucirepo(id=186) 
    X = wine_quality.data.features
    y = wine_quality.data.targets
    df = pd.concat([X, y], axis=1)
    df = df.drop_duplicates() 
    df.to_csv("raw.csv", index=False)

download_dataset()

#Обучние scaler, чтобы признаки на входе имели такое же масштабирование как и датасет на котором обучалась модель
#-----------------------------------------------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

config = load_config("params.yaml")
df = pd.read_csv("raw.csv")
variability_threshold = config["preprocessing"]["variability_threshold"]

#Удаление неинформативных признаков
for col in df.columns:
    if df[col].value_counts(normalize=True).max() > variability_threshold:
        df = df.drop(col, axis=1)

df["alcohol_density_ratio"] = df["alcohol"] / df["density"]
y = df["quality"]
X = df.drop("quality", axis=1)

#Масшатбирование
scaler = MinMaxScaler()
scaler = scaler.fit(X)
#-----------------------------------------------------------------------------------------------------------------


# Модель входных данных
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    alcohol_density_ratio: float

@app.post("/predict", summary="Predict wine quality")
async def predict(wine: WineFeatures):
    """
    Предсказывает качество вина
    """
    try:
        columns_names = ["fixed_acidity", 
                         "volatile_acidity", 
                         "citric_acid", 
                         "residual_sugar",
                         "chlorides", 
                         "free_sulfur_dioxide", 
                         "total_sulfur_dioxide", 
                         "density",
                         "pH",
                         "sulphates",
                         "alcohol",
                         "alcohol_density_ratio"]
        input_data = pd.DataFrame([wine.dict()])
        input_data.columns = columns_names
        feautures = scaler.transform(input_data)
        print(feautures)
        predict = model.predict(feautures)[0]
        # logger.info(f"Predicted price: {price}")
        
        return {"predicted_quality": round(float(predict), 2)}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
    
#-----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)