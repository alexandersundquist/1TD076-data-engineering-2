from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from ray import tune
import ray
import numpy as np

def train_model(config):
    # Load data
    data = fetch_covtype(as_frame=True)
    X = data.data
    y = data.target

    # Put the data in Ray's object store
    X_ref = ray.put(X)
    y_ref = ray.put(y)

    # Fetch data from the object store within the function (to avoid OOM)
    X_data = ray.get(X_ref)
    y_data = ray.get(y_ref)

    # Define the model
    clf = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        n_jobs=-1,
        random_state=42
    )

    # Perform cross-validation (use ray.get() to avoid memory overload)
    scores = cross_val_score(clf, X_data, y_data, cv=3, scoring='accuracy')
    mean_score = np.mean(scores)

    # Report the results to Ray
    tune.report(mean_accuracy=mean_score)
