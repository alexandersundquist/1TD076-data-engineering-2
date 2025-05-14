from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from ray import tune
import numpy as np

def train_model(config):
    data = fetch_covtype(as_frame=True)
    X = data.data
    y = data.target

    clf = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        n_jobs=-1,
        random_state=42
    )

    # Stratified 3-fold cross-validation
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    mean_score = np.mean(scores)

    tune.report(mean_accuracy=mean_score)
