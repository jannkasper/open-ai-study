import pandas as pd
import pickle

from embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

EMBEDDING_MODEL = "text-embedding-3-small"

# load data https://github.com/openai/openai-cookbook/blob/main/examples/data/AG_news_samples.csv
dataset_path = "data/AG_news_samples.csv"

# establish a cache of embeddings to avoid recomputing
# cache is a dict of tuples (text, model) -> embedding, saved as a pickle file

# set path to embedding cache
embedding_cache_path = "data/recommendations_embeddings_cache.pkl"

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)


# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
        string: str,
        model: str = EMBEDDING_MODEL,
        embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

def print_recommendations_from_strings(
        strings: list[str],
        index_of_source_string: int,
        k_nearest_neighbors: int = 1,
        model=EMBEDDING_MODEL,
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all strings
    embeddings = [embedding_from_string(string, model=model) for string in strings]

    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]

    # get distances between the source embedding and other embeddings (function from utils.embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")

    # get indices of nearest neighbors (function from utils.utils.embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # print out source string
    query_string = strings[index_of_source_string]
    print(f"Source string: {query_string}")
    # print out its k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_string == strings[i]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # print out the similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i]}
        Distance: {distances[i]:0.3f}"""
        )

    return indices_of_nearest_neighbors


def run():
    df = pd.read_csv(dataset_path)

    n_examples = 5
    df.head(n_examples)

    # print the title, description, and label of each example
    for idx, row in df.head(n_examples).iterrows():
        print("")
        print(f"Title: {row['title']}")
        print(f"Description: {row['description']}")
        print(f"Label: {row['label']}")


    # as an example, take the first description from the dataset
    example_string = df["description"].values[0]
    print(f"\nExample string: {example_string}")

    # print the first 10 dimensions of the embedding
    example_embedding = embedding_from_string(example_string)
    print(f"\nExample embedding: {example_embedding[:10]}...")

    article_descriptions = df["description"].tolist()

    tony_blair_articles = print_recommendations_from_strings(
        strings=article_descriptions,  # let's base similarity off of the article description
        index_of_source_string=0,  # articles similar to the first one about Tony Blair
        k_nearest_neighbors=5,  # 5 most similar articles
    )






