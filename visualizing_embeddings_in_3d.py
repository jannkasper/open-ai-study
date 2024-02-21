import pandas as pd
from embeddings_utils import get_embeddings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


samples = pd.read_json("data/dbpedia_samples.jsonl", lines=True)


def run():
    # 1. Load the dataset and query embeddings
    categories = sorted(samples["category"].unique())
    print("Categories of DBpedia samples:", samples["category"].value_counts())
    samples.head()
    # NOTE: The following code will send a query of batch size 200 to /embeddings
    matrix = get_embeddings(samples["text"].to_list(), model="text-embedding-3-small")

    # 2. Reduce the embedding dimensionality
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(matrix)
    samples["embed_vis"] = vis_dims.tolist()

    # 3. Plot the embeddings of lower dimensionality

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(projection='3d')
    cmap = plt.get_cmap("tab20")

    # Plot each sample category individually such that we can set label name.
    for i, cat in enumerate(categories):
        sub_matrix = np.array(samples[samples["category"] == cat]["embed_vis"].to_list())
        x = sub_matrix[:, 0]
        y = sub_matrix[:, 1]
        z = sub_matrix[:, 2]
        colors = [cmap(i / len(categories))] * len(sub_matrix)
        ax.scatter(x, y, zs=z, zdir='z', c=colors, label=cat)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(bbox_to_anchor=(1.1, 1))
    plt.show()
