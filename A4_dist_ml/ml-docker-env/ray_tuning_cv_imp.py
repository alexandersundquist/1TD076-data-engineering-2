import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import time

# -----------------------------
# Load and upload data to Ray
# -----------------------------
def load_and_preprocess_data():
    data = fetch_covtype()
    X, y = data.data, data.target

    # Normalize for efficiency (optional)
    X = X.astype(np.float32) / X.max(axis=0)

    # First, split off a test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return (
        ray.put(X_trainval),
        ray.put(y_trainval),
        ray.put(X_test),
        ray.put(y_test),
    )

# -----------------------------
# Trainable function for tuning
# -----------------------------
def train_rf_cv(config, data_refs):
    # Load training data (cross-validation only on train)
    X_trainval, y_trainval, _, _ = ray.get(data_refs)

    model = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        random_state=42,
        n_jobs=1
    )

    start_time = time.time()

    # Cross-validation (5 folds)
    scores = cross_val_score(model, X_trainval, y_trainval, cv=5, n_jobs=-1)
    mean_accuracy = scores.mean()

    train_duration = time.time() - start_time

    tune.report({"mean_cv_accuracy": mean_accuracy, "cv_time": train_duration})

# -----------------------------
# Main execution
# -----------------------------
def main():
    ray.init()

    # Load and put data into Ray
    X_trainval_ref, y_trainval_ref, X_test_ref, y_test_ref = load_and_preprocess_data()
    data_refs = [X_trainval_ref, y_trainval_ref, X_test_ref, y_test_ref]

    # Hyperparameter search space
    config = {
        "max_depth": tune.grid_search([10, 20]),
        "n_estimators": tune.grid_search([50, 100]),
        "ccp_alpha": tune.grid_search([0.0, 0.01])
    }

    # ASHA scheduler
    scheduler = ASHAScheduler(
        metric="mean_cv_accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    # Define trainable with data_refs
    trainable = tune.with_parameters(train_rf_cv, data_refs=data_refs)

    # Tuner
    trainable_with_resources = tune.with_resources(trainable, {"cpu": 2})
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
        ),
    )

    tuning_start = time.time()
    results = tuner.fit()
    tuning_end = time.time()

    total_tuning_time = tuning_end - tuning_start

    # Get best config
    best_result = results.get_best_result(metric="mean_cv_accuracy", mode="max")
    best_config = best_result.config
    print("\nBest config found during tuning:", best_config)
    print("Mean CV accuracy of best model:", best_result.metrics["mean_cv_accuracy"])

    # -----------------------------
    # Final model evaluation
    # -----------------------------
    # Load test data
    X_trainval, y_trainval, X_test, y_test = ray.get(data_refs)

    final_model = RandomForestClassifier(
        max_depth=best_config["max_depth"],
        n_estimators=best_config["n_estimators"],
        ccp_alpha=best_config["ccp_alpha"],
        random_state=42,
        n_jobs=1
    )

    final_model.fit(X_trainval, y_trainval)
    test_accuracy = final_model.score(X_test, y_test)

    print(f"Test accuracy of best model: {test_accuracy:.4f}")
    print(f"Total Ray Tune optimization time: {total_tuning_time:.2f} seconds")

    ray.shutdown()

if __name__ == "__main__":
    main()
