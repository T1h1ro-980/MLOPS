from ucimlrepo import fetch_ucirepo 
import pandas as pd

def download_dataset():
    wine_quality = fetch_ucirepo(id=186) 
    X = wine_quality.data.features
    y = wine_quality.data.targets
    df = pd.concat([X, y], axis=1)
    df = df.drop_duplicates() 
    df.to_csv("data/raw.csv", index=False)

if __name__ == "__main__":
    download_dataset() 