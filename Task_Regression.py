import pandas as pd
import requests
import zipfile
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

# Data loading and preprocessing functions
def load_and_preprocess_data():
    """
    Load and preprocess the bike sharing dataset. This dataset includes dates and other features related to the number of bike rentals.
    Returns the feature matrix X and the target variable y.
    """
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    
    resp = requests.get(data_url)
    resp.raise_for_status()

    # TODO: Extract the 'day.csv' file from the zip archive and load it into the data DataFrame.
    
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("day.csv") as f:
            data = pd.read_csv(f)
    
    # TODO: Convert the 'dteday' column from a string to a datetime format and extract 
    # and create a 'day_of_month' column from it, 
    # which contains only the "day" information for each date
    
    data["dteday"] = pd.to_datetime(data["dteday"], errors="coerce")
    data["day_of_month"] = data["dteday"].dt.day

    # TODO: Remove the original date column and other irrelevant columns ('dteday', 'casual', 'registered', 'cnt').
    # 'dteday' column contains date information and has been specially handled above.
    # 'casual' and 'registered' column would introduce a bias and lead to overfitting
    # 'cnt' column already contains the target information
    
    y = data["cnt"].astype(float)
    X = data.drop(columns=["dteday", "casual", "registered", "cnt"])

    return X, y

# Define the function for the random forest regression experiment
def random_forest_regression_experiment(n_estimators, min_samples_leaf, X, y):
    """
    Trains a RandomForestRegressor with the given parameters and performs 10-fold cross-validation.

    Parameters:
    - n_estimators: The number of trees in the forest.
    - min_samples_leaf: The minimum number of samples required for a leaf node.
    - X: Feature matrix (bike features).
    - y: Target vector (number of bike rentals).

    Returns:
    - The average negative mean squared error of the cross-validation.
    """

    # TO DO: Create a RandomForestRegressor model with given parameters
    # n_estimators and min_sample_leaf as input, set random_state to 42
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",     
        bootstrap=True,          
        random_state=42,
        n_jobs=-1
    )

    # TO DO: Implement 10-fold cross-validation with random_state set to 42
    # and compute the mean negative mean squared error using cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    neg_mse = cross_val_score(model, X, y, cv=cv,
                                scoring="neg_mean_squared_error",
                                n_jobs=-1)
    mse = -neg_mse.mean()

    # Return the computed mean negative mean squared error.
    return mse

# Function to find the best hyperparameter combination
def get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y):
    """
    Find the best hyperparameter combination based on cross validation negative mean squared error.

    Parameters:
    - n_estimators_list: list of different values ​​for the number of trees.
    - min_samples_leaf_list: list of different values ​​for the minimum number of leaf samples.
    - X: feature matrix.
    - y: target vector.

    Returns:
    - The best hyperparameter combination and the corresponding negative mean squared error.
    """
    results = []

    # TODO: Iterate over all combinations of hyperparameters, calculate the cross-validation negative mean squared error 
    # using the random_forest_regression_experiment function for each combination, and append the results (n_estimators, 
    # min_samples_leaf, and mse) to the results list.
    
    # TODO: Select the result with the smallest negative mean squared error (closest to zero)
    # from the results list as the best result.
    
    best_result = None
    for n_est in n_estimators_list:
        for min_leaf in min_samples_leaf_list:
            mean_neg_mse = random_forest_regression_experiment(n_est, min_leaf, X, y)

            result = {
                "n_estimators": n_est,
                "min_samples_leaf": min_leaf,
                "mse": float(mean_neg_mse)  
            }
            results.append(result)

            if (best_result is None) or (result["mse"] > best_result["mse"]):
                best_result = result

    # Return the best combination of hyperparameters and their corresponding smallest negative mean squared error
    return best_result # best_result is a dictionary

    # TO DO: After running the above function, manually input your best result here
def manually_entered_best_params_and_mse():
    best_params = {'n_estimators': 10, 'min_samples_leaf': 100}  # retrieved best parameters
    best_mse = 1613282.6348  # replaced with my achieved mse
    return best_params, best_mse

if __name__ == "__main__":
    # Experiment with different values for n_estimators and min_samples_leaf
    # to find the best parameters setting among the following options:
    X, y = load_and_preprocess_data()
    n_estimators_list = [10, 50, 100, 1000] # Do not modify these parameter settings, experiment with them
    min_samples_leaf_list = [1, 10, 50, 100] # Do not modify these parameter settings, experiment with them

    best_params = get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y)
    print(f"Best Hyperparameter: n_estimators = {best_params['n_estimators']}, min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"Best negative mean square error: {best_params['mse']:.4f}")
