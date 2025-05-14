import ray
from ray import tune
from rf_tune import train_model
import time

def main():
    ray.init(address="auto")  # Connect to Ray cluster

    config = {
        "max_depth": tune.grid_search([10, 20, 30]),
        "n_estimators": tune.grid_search([50, 100, 200]),
        "ccp_alpha": tune.grid_search([0.0, 0.01])
    }

    tuner = tune.Tuner(
        train_model,
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max"
        )
    )

    start_time = time.time()

    results = tuner.fit()

    end_time = time.time()
    elapsed_time = end_time - start_time

    best_result = results.get_best_result()
    print("Best config:", best_result.config)
    print("Best accuracy:", best_result.metrics["mean_accuracy"])
    print(f"uning completed in {elapsed_time:.2f} seconds.")
if __name__ == "__main__":
    main()
