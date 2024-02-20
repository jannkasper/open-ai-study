import os
from openai import OpenAI
import pandas as pd
import tiktoken

EMBEDDING_MODEL = "text-embedding-3-small"

# Download https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
datafile_path = "data/fine_food_reviews_1k.csv"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

count = 0;
def get_embedding(text, model="text-embedding-3-small"):
    global count
    print(count)
    count += 1
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def run():
    embedding_model = "text-embedding-3-small"
    embedding_encoding = "cl100k_base"
    max_tokens = 8000  # the maximum for text-embedding-3-small is 8191
    # load & inspect dataset
    input_datapath = "data/fine_food_reviews_1k.csv"  # to save space, we provide a pre-filtered dataset
    df = pd.read_csv(input_datapath, index_col=0)
    df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
    df = df.dropna()
    df["combined"] = ("Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip())
    print(df.head(2))

    # subsample to 1k most recent reviews and remove samples that are too long
    top_n = 1000
    df = df.sort_values("Time").tail(
        top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
    df.drop("Time", axis=1, inplace=True)

    encoding = tiktoken.get_encoding(embedding_encoding)

    # omit reviews that are too long to embed
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens].tail(top_n)
    print(len(df))

    # Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

    # This may take a few minutes
    df["embedding"] = df.combined.apply(lambda x: get_embedding(x, model=embedding_model))
    df.to_csv("data/fine_food_reviews_with_embeddings_1k.csv")