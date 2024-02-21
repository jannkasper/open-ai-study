import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval

from embeddings_utils import cosine_similarity

import matplotlib.pyplot as plt
import statsmodels.api as sm

def run():
    # https://github.com/openai/openai-cookbook/blob/main/examples/data/fine_food_reviews_with_embeddings_1k.csv
    df = pd.read_csv('data/fine_food_reviews_with_embeddings_1k.csv',
                     index_col=0)  # note that you will need to generate this file to run the code below
    df.head(2)

    df['babbage_similarity'] = df["embedding"].apply(literal_eval).apply(np.array)
    X_train, X_test, y_train, y_test = train_test_split(df, df.Score, test_size=0.2, random_state=42)

    user_embeddings = X_train.groupby('UserId').babbage_similarity.apply(np.mean)
    prod_embeddings = X_train.groupby('ProductId').babbage_similarity.apply(np.mean)
    print(len(user_embeddings))
    print(len(prod_embeddings))

    # evaluate embeddings as recommendations on X_test
    def evaluate_single_match(row):
        user_id = row.UserId
        product_id = row.ProductId
        try:
            user_embedding = user_embeddings[user_id]
            product_embedding = prod_embeddings[product_id]
            similarity = cosine_similarity(user_embedding, product_embedding)
            return similarity
        except Exception as e:
            return np.nan

    X_test['cosine_similarity'] = X_test.apply(evaluate_single_match, axis=1)
    X_test['percentile_cosine_similarity'] = X_test.cosine_similarity.rank(pct=True)

    correlation = X_test[['percentile_cosine_similarity', 'Score']].corr().values[0, 1]
    print(
        'Correlation between user & vector similarity percentile metric and review number of stars (score): %.2f%%' % (
                    100 * correlation))

    # boxplot of cosine similarity for each score
    X_test.boxplot(column='percentile_cosine_similarity', by='Score')
    plt.title('')
    plt.show()
