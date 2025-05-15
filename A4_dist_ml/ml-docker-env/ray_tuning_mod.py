import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return (
        ray.put(X_train),
        ray.put(y_train),
        ray.put(X_test),
        ray.put(y_test),
    )

# -----------------------------
# Trainable function with data refs
# -----------------------------
def train_rf_with_data(config, data_refs):
    # Resolve ObjectRefs INSIDE the Ray worker
    X_train, y_train, X_test, y_test = ray.get(data_refs)

    model = RandomForestClassifier(
        max_depth=config.get("max_depth"),
        n_estimators=config.get("n_estimators"),
        ccp_alpha=config.get("ccp_alpha"),
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - start_time

    accuracy = model.score(X_test, y_test)

    tune.report({"accuracy":accuracy, "training_time":train_duration})

# -----------------------------
# Main execution
# -----------------------------
def main():
    ray.init()

    # Load and upload data
    X_train_ref, y_train_ref, X_test_ref, y_test_ref = load_and_preprocess_data()
    data_refs = [X_train_ref, y_train_ref, X_test_ref, y_test_ref]

    # Define search space
    config = {
        "max_depth": tune.choice([10, 20, 30, None]),
        "n_estimators": tune.choice([50, 100, 200]),
        "ccp_alpha": tune.choice([0.0, 0.01, 0.1])
    }

    # Use ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    # Wrap train function with data refs
    trainable = tune.with_parameters(train_rf_with_data, data_refs=data_refs)

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
    tuning_end = time.time()
    total_tuning_time = tuning_end - tuning_start

    # Best result
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print("\nBest config:", best_result.config)
    print("Best accuracy:", best_result.metrics["accuracy"])
    print("Best training time (per trial):", best_result.metrics["training_time"])
    print(f"Total Ray Tune optimization time: {total_tuning_time:.2f} seconds")

    ray.shutdown()

if __name__ == "__main__":
    main()
