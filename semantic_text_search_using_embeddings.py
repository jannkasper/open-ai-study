import pandas as pd
import numpy as np
from ast import literal_eval

# https://github.com/openai/openai-cookbook/blob/main/examples/data/fine_food_reviews_with_embeddings_1k.csv
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

from embeddings_utils import get_embedding, cosine_similarity

# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        model="text-embedding-3-small"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


def run():
    df = pd.read_csv(datafile_path)
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

    results = search_reviews(df, "delicious beans", n=3)
    print(results)
    results = search_reviews(df, "whole wheat pasta", n=3)
    print(results)
    results = search_reviews(df, "bad delivery", n=1)
    print(results)

