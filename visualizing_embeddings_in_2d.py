import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Load the embeddings
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

def run():
    df = pd.read_csv(datafile_path)
    # Convert to a list of lists of floats
    matrix = np.array(df.embedding.apply(literal_eval).to_list())

    # Create a t-SNE model and transform the data
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    print(vis_dims.shape)

    colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
    x = [x for x,y in vis_dims]
    y = [y for x,y in vis_dims]
    color_indices = df.Score.values - 1

    colormap = matplotlib.colors.ListedColormap(colors)
    plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
    for score in [0,1,2,3,4]:
        avg_x = np.array(x)[df.Score-1==score].mean()
        avg_y = np.array(y)[df.Score-1==score].mean()
        color = colors[score]
        plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)

    plt.title("Amazon ratings visualized in language using t-SNE")
    plt.show()