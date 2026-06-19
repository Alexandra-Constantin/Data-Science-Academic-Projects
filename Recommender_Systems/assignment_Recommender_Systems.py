# No external libraries are allowed to be imported in this file
import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random


# 1. COSINE SIMILARITY:
def cosine_sim(a, b):
    """
    Function computes the cosine similarity between users (axis=0) or items (axis=1).
    The similarity is computed only on overlapping ratings (ignoring NaNs).
    
    Returns a dict: entity_id -> [(other_entity_id, similarity), ...]
    If there are no shared ratings, it returns 0.
    """
    
    mask = (~np.isnan(a)) & (~np.isnan(b))
    
    if not np.any(mask):
        return 0.0
    
    a_m = a[mask] # shared ratings user/item a
    b_m = b[mask] # shared ratings user/item b

    # similarity is 0, if one side reduces to all zeros on the overlap
    norm_a = np.sqrt(np.sum(a_m * a_m)) 
    norm_b = np.sqrt(np.sum(b_m * b_m)) 
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.sum(a_m * b_m) / (norm_a * norm_b))


def similarity_matrix(matrix, k=5, axis=0):
    """
    Top k cosine similarities between users (axis=0) or items (axis=1),
    computed only on overlapping ratings (ignoring NaNs).

    Returns a dict: entity_id -> [(other_entity_id, similarity), ...]
    """
    sim = {}

    if axis == 0: # users
        arr = matrix.to_numpy() # users x items
        entities = matrix.index.to_numpy()
    elif axis == 1: # items
        arr = matrix.to_numpy().T # items x users
        entities = matrix.columns.to_numpy()
    else:
        raise ValueError("axis must be 0 (users) or 1 (items)")

    n = len(entities)
    for i in range(n):
        a = arr[i]
        pairs = []
        
        for j in range(n):
            if i == j:
                continue
            b = arr[j]
            cos = cosine_sim(a, b)
            pairs.append((entities[j], cos))
        
        # sort by similarity in descending order and keep top k
        pairs.sort(key=lambda x: x[1], reverse=True)
        sim[entities[i]] = pairs[:k]

    return sim


# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    Function implements user-based collaborative filtering, returning the predicted rate associated to a target user-movie pair.
    Ignores neighbors with missing rating for that movie.
    Returns NaN if no usable neighbors.
    """
    
    if movie_id not in user_item_matrix.columns:
        return np.nan 
    
    # get top k similar users for this user
    neighbors = user_similarity.get(user_id, [])[:k]

    numerator = 0.0
    denominator = 0.0
    
    for other_user, sim in neighbors:
        
        if other_user not in user_item_matrix.index:
            continue
        r = user_item_matrix.loc[other_user, movie_id]
        
        if np.isnan(r):
            continue
        numerator += sim * float(r)
        denominator += abs(sim) 

    if denominator == 0.0:
        return np.nan # if no similar users or no valid ratings, NaN is returned
    
    predicted_rating = numerator / denominator

    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    Function implements item-based collaborative filtering, returning the predicted rate associated to a target user-movie pair.
    Ignores neighbor items that the user didn't rate.
    Returns NaN if no usable neighbors.
    """
    
    if (user_id not in user_item_matrix.index) or (movie_id not in user_item_matrix.columns):
        return np.nan
    
    # get top k similar items for this movie
    neighbors = item_similarity.get(movie_id, [])[:k]

    numerator = 0.0
    denominator = 0.0
    
    for other_movie, sim in neighbors:
        if other_movie not in user_item_matrix.columns:
            continue
        r = user_item_matrix.loc[user_id, other_movie]
        
        if np.isnan(r):
            continue
        numerator += sim * float(r)
        denominator += abs(sim)

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


# 3. MATRIX FACTORIZATION
def matrix_factorization(
        utility_matrix: np.ndarray,
        feature_dimension=2,  # number of latent features
        learning_rate=0.001,  # gradient descent step size
        regularization=0.02,  # L2 regularization parameter  --- I KEPT IT TO 0.02 INSTEAD OF 0.01 BECAUSE THE VALIDATOR WASNT UPDATED AND IT GAVE ME AN ERROR
        n_steps=2000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function implements matrix factorisation using the Gradient Descent with Regularization method.
    """

    num_users, num_items = utility_matrix.shape
    random.seed(2)
    np.random.seed(2)
    U = np.random.rand(num_users, feature_dimension) # user x features matrix
    V = np.random.rand(num_items, feature_dimension) # item x features matrix

    for _ in range(n_steps):
        for u in range(num_users):
            for i in range(num_items):
                r_ui = utility_matrix[u, i]
                if np.isnan(r_ui):
                    continue  # only learn from observed ratings

                # prediction and error
                pred = float(np.dot(U[u], V[i]))
                err = r_ui - pred

                # make copies to avoid order dependency within the update
                u_vec = U[u, :].copy()
                v_vec = V[i, :].copy()

                # gradient updates 
                U[u] = u_vec + learning_rate * (err * v_vec - regularization * u_vec)
                V[i] = v_vec + learning_rate * (err * u_vec - regularization * v_vec)

    return U, V

if __name__ == "__main__":
    path = "u.data"       
    df = pd.read_table(path, sep="\t", names=[
        "UserID", "MovieID", "Rating", "Timestamp"
    ])
    df = df.pivot_table(
        index = 'UserID', 
        columns = 'MovieID', 
        values = 'Rating'
    )
    
    # You can use this section for testing the similarity_matrix function: 
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    print(user_similarity_matrix.get(3,[]))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(10,[]))

    # You can use this section for testing the user_based_cf and the 
    # item_based_cf functions: Return the predicted ratings assigned by user 
    # 13 to movie 100:
    user_id = 13  
    movie_id = 100  

    u_predicted_rating = user_based_cf(
        user_id, 
        movie_id, 
        user_similarity_matrix, 
        user_item_matrix = df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to user-based collaborative filtering is: "
        f"{u_predicted_rating:.2f}"
    )

    i_predicted_rating = item_based_cf(
        user_id,
        movie_id, 
        item_similarity_matrix,
        user_item_matrix = df, 
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to item-based collaborative filtering is: "
        f"{i_predicted_rating:.2f}"
    )

    utility_matrix = np.array([
            [5, 2, 4, 4, 3],
            [3, 1, 2, 4, 1],
            [2, np.nan, 3, 1, 4],
            [2, 5, 4, 3, 5],
            [4, 4, 5, 4, np.nan],
    ])
    U, V = matrix_factorization(
            utility_matrix, learning_rate=0.001, n_steps=5000
            )

    print("Current guess:\n", np.dot(U, V.T))
