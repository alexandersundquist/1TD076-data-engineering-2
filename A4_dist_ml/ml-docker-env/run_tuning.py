import ray
from ray import tune
from ray.air import RunConfig # Import RunConfig
from rf_tune import train_model
import time

def main():
    # For local testing or if your cluster setup is minimal,
    # explicitly setting num_cpus can help Ray manage resources better.
    # If on a larger cluster, address="auto" is typically fine.
    # Example for local: ray.init(num_cpus=2) # Assuming you have at least 2 cores
    ray.init(address="auto")

    # Consider reducing search space for initial testing if memory is very tight
    config = {
        "max_depth": tune.grid_search([5, 10]), # Smaller grid for example
        "n_estimators": tune.grid_search([10, 20]), # Smaller grid
        "ccp_alpha": tune.grid_search([0.0, 0.01])
    }

    # Define resources per trial - crucial for memory management
    # This tells Ray that each trial needs 1 CPU. If your node has few CPUs,
    # this will naturally limit concurrency.
    resources_per_trial = {"cpu": 1}

    tuner = tune.Tuner(
        tune.with_resources(train_model, resources=resources_per_trial), # Apply resource request
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            # You could also use max_concurrent_trials if you prefer,
            # but with_resources is often more flexible.
            # max_concurrent_trials=1
        ),
        run_config=RunConfig(
            name="rf_tuning_experiment", # Good practice to name experiments
            # Optional: configure retries if tasks fail (e.g. due to transient OOM)
            # failure_config=RunConfig.FailureConfig(max_failures=1), # Retry once
        )
    )

    start_time = time.time()
    results = tuner.fit()
    end_time = time.time()
    elapsed_time = end_time - start_time

    if results.errors:
        print("Errors encountered during tuning:")
        for i, trial_result in enumerate(results):
            if trial_result.error:
                print(f"Trial {trial_result.trial_id} failed with error: {trial_result.error}")

    best_result = results.get_best_result(metric="mean_accuracy", mode="max")
    if best_result:
        print("Best config:", best_result.config)
        print("Best accuracy:", best_result.metrics["mean_accuracy"])
    else:
        print("No successful trials completed.")
    print(f"Tuning completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
