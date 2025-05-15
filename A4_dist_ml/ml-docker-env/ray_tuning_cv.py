import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import cross_val_score
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

    # Upload entire dataset to Ray object store
    return ray.put(X), ray.put(y)

# -----------------------------
# Trainable function with cross-validation
# -----------------------------
def train_rf_cv(config, data_refs):
    # Resolve ObjectRefs inside the Ray worker
    X, y = ray.get(data_refs)

    model = RandomForestClassifier(
        max_depth=config.get("max_depth"),
        n_estimators=config.get("n_estimators"),
        ccp_alpha=config.get("ccp_alpha"),
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()

    # 5-fold cross-validation with accuracy scoring
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    mean_accuracy = scores.mean()

    train_duration = time.time() - start_time

    # Report to Ray Tune
    tune.report({"accuracy":mean_accuracy, "training_time":train_duration})

# -----------------------------
# Main execution
# -----------------------------
def main():
    ray.init()

    # Load and upload full dataset
    X_ref, y_ref = load_and_preprocess_data()
    data_refs = [X_ref, y_ref]

    # Define search space
    config = {
        "max_depth": tune.choice([10, 20, 30, None]),
        "n_estimators": tune.choice([50, 100, 200]),
        "ccp_alpha": tune.choice([0.0, 0.01, 0.1])
    }

    # Optional: Use ASHA for early stopping
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=1,  # only one round in CV so this doesn’t help much here
        grace_period=1,
        reduction_factor=2
    )

    # Wrap trainable
    trainable = tune.with_parameters(train_rf_cv, data_refs=data_refs)

    # Set up the tuner
    tuner = tune.Tuner(
        trainable,
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=10
        )
    )

    # Measure total tuning time
    tuning_start = time.time()
    results = tuner.fit()
    total_tuning_time = time.time() - tuning_start

    # Best result
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("\nBest config:", best_result.config)
    print("Best CV accuracy:", best_result.metrics["accuracy"])
    print("Best training time (per trial):", best_result.metrics["training_time"])
    print(f"Total Ray Tune optimization time: {total_tuning_time:.2f} seconds")

    ray.shutdown()

if __name__ == "__main__":
    main()
