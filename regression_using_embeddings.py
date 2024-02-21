import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# https://github.com/openai/openai-cookbook/blob/main/examples/data/fine_food_reviews_with_embeddings_1k.csv
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

def run():
    df = pd.read_csv(datafile_path)
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

    X_train, X_test, y_train, y_test = train_test_split(list(df.embedding.values), df.Score, test_size=0.2,
                                                        random_state=42)

    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)
    preds = rfr.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"text-embedding-3-small performance on 1k Amazon reviews: mse={mse:.2f}, mae={mae:.2f}")

    bmse = mean_squared_error(y_test, np.repeat(y_test.mean(), len(y_test)))
    bmae = mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))
    print(
        f"Dummy mean prediction performance on Amazon reviews: mse={bmse:.2f}, mae={bmae:.2f}"
    )

