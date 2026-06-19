# No external libraries are allowed to be imported in this file
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data():
    """
    Loads the Swiss Roll dataset and corresponding color labels from files.

    Returns:
    tuple: The data (X) and color labels (color)
    """
    # TO DO: Load dataset from files

    X = np.load("swiss_roll_larger.npy")
    color = np.load("color_larger.npy")

    return X, color

# Function to apply t-SNE to the dataset
def apply_tsne(X, n_components, perplexity, max_iter, init, random_state=2024):
    """
    Applies t-SNE to the Swiss Roll dataset after scaling it.

    Parameters:
    X (array): The input dataset.
    perplexity (float): t-SNE perplexity parameter.
    random_state (int): Random seed for reproducibility.

    Returns:
    array: The t-SNE transformed dataset with 2 components.
    """
    # TO DO: Create a pipeline to apply StandardScaler and t-SNE

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=max_iter,
        init=init,
        learning_rate="auto",
        random_state=random_state,
    )
    
    # Standardize first, then apply t-SNE
    pipe = make_pipeline(StandardScaler(), tsne)
    X_tsne_2d = pipe.fit_transform(X)

    return X_tsne_2d

# Function to plot the 2D t-SNE projection
def plot_tsne_projection(X_tsne_2d, color):
    """
    Plots the 2D projection of the t-SNE transformed Swiss Roll dataset.

    Parameters:
    X_tsne_2d (array): The t-SNE transformed dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the 2D projection from t-SNE

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=color, s=8, alpha=0.85)
    plt.title("Swiss Roll larger: 2D t-SNE projection")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, color = load_data()

    # TO DO: Fill in the appropriate values for n_components, perplexity, max_iter, and init
    X_tsne_2d = apply_tsne(X, n_components=2, perplexity=30, max_iter=5000, init="pca", random_state=2024)

    plot_tsne_projection(X_tsne_2d, color)
    
    
# REGARDING THE LETTER:

# With perplexity < 30, the structure is very fragmented
# With perplexity=30, I find the same C/L-like structure as in task 4
# With perplexity >= 50, I cant identify any letter-like structure
# I also varied max_iter and init, but they did not seem to have a big impact