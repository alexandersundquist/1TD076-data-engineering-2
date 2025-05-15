import ray
from ray import tune
# from ray.tune.search.grid_search import GridSearchCv # Not strictly needed if using tune.grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype # For the covertype dataset
from sklearn.preprocessing import StandardScaler # Good practice for some datasets, though RF is less sensitive
import pandas as pd # For potential data inspection, not strictly needed for covtype numpy arrays
import numpy as np
import time
import os

# 1. Define the training function for Ray Tune
def train_rf(config, X_train_ref, y_train_ref, X_test_ref, y_test_ref):
    """
    Trainable function for Ray Tune.
    Args:
        config (dict): Dictionary of hyperparameters.
        X_train_ref, y_train_ref, X_test_ref, y_test_ref: Ray ObjectRefs for the data.
    """
    # Ray will automatically resolve these ObjectRefs (effectively doing a ray.get())
    # when they are accessed by the scikit-learn functions on the worker node.
    X_train = ray.get(X_train_ref)
    y_train = ray.get(y_train_ref)
    X_test = ray.get(X_test_ref)
    y_test = ray.get(y_test_ref)

    # It's good practice to scale data, especially for some algorithms,
    # though RandomForest is often robust to feature scaling.
    # For simplicity and focus on Ray Tune, skipping scaling here, but consider it.
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        max_depth=config.get("max_depth"),
        n_estimators=config.get("n_estimators"),
        ccp_alpha=config.get("ccp_alpha"),
        random_state=42, # for reproducibility
        n_jobs=1        # Use all cores *allocated to this trial*
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - start_time

    accuracy = model.score(X_test, y_test)
    # The covertype dataset has classes 1 through 7.
    # Ensure your evaluation metric makes sense for multiclass. Accuracy is common.

    tune.report(accuracy=accuracy, training_time=train_duration, done=True)


# 2. Prepare and Distribute Data using ray.put()
def load_and_distribute_data():
    print("Fetching covtype dataset...")
    covtype = fetch_covtype()
    X, y = covtype.data, covtype.target
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Optional: Subsample for faster tuning during development/testing
    X, _, y, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42) # 10% subsample
    print(f"Subsampled dataset shape: X={X.shape}, y={y.shape}")


    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # For RandomForest, scaling is not strictly necessary but can sometimes help.
    # We'll skip it for this example to keep it focused, but consider it for other models/datasets.
    # print("Scaling data...")
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # print("Data scaling complete.")

    print("Putting data into Ray object store...")
    X_train_ref = ray.put(X_train)
    y_train_ref = ray.put(y_train)
    X_test_ref = ray.put(X_test)
    y_test_ref = ray.put(y_test)
    print("Data successfully placed in Ray object store.")

    return X_train_ref, y_train_ref, X_test_ref, y_test_ref


# 3. Define the Hyperparameter Search Space
search_space = {
    "max_depth": tune.grid_search([10, 20, 30]), # Adjusted for potentially longer training times
    "n_estimators": tune.grid_search([50, 100, 150]), # Adjusted
    "ccp_alpha": tune.grid_search([0.0, 0.001, 0.005]) # Adjusted
}

# Calculate the total number of trials for Grid Search
num_trials = 1
for param_values in search_space.values():
    if isinstance(param_values, dict) and "grid_search" in param_values:
        num_trials *= len(param_values["grid_search"])
    elif hasattr(param_values, "__iter__") and not isinstance(param_values, str): # for simple lists
        num_trials *= len(param_values)

if __name__ == "__main__":
    # --- Ray Cluster Initialization ---
    # (Same instructions as before for starting head and worker nodes on VMs)
    if not ray.is_initialized():
        print("Initializing Ray for local run or connecting to existing cluster...")
        # On the head node of a pre-configured cluster, or for local run:
        ray.init(address="auto", ignore_reinit_error=True)
        # If submitting from a client machine not part of the cluster:
        # ray.init(address="ray://<HEAD_NODE_IP>:10001", ignore_reinit_error=True)
        # Ensure `num_cpus` is not set if you want Ray to detect all CPUs in the cluster.
        # If you set it, it limits the CPUs Ray uses on the local machine if starting a new cluster.

    num_connected_nodes = len(ray.nodes())
    total_cpus_in_cluster = ray.cluster_resources().get("CPU", 0)
    print(f"Ray cluster detected with {num_connected_nodes} nodes and {total_cpus_in_cluster} total CPUs.")

    # --- Load and Distribute Data ---
    X_train_ref, y_train_ref, X_test_ref, y_test_ref = load_and_distribute_data()

    # --- Configure Ray Tune ---
    resources_per_trial = {"cpu": 1} # Each trial uses 1 CPU for the RandomForestClassifier itself.
                                     # RF's n_jobs will be limited by this.
                                     # If you want each RF to use more cores, increase this, e.g., {"cpu": 4}
                                     # and ensure n_jobs in RF is also set (e.g., -1 or matching number).

    # Use tune.with_parameters to pass the ObjectRefs to the trainable function
    trainable_with_data_refs = tune.with_parameters(
        train_rf,
        X_train_ref=X_train_ref,
        y_train_ref=y_train_ref,
        X_test_ref=X_test_ref,
        y_test_ref=y_test_ref
    )

    tuner = tune.Tuner(
        trainable_with_data_refs,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            # num_samples is effectively 1 for grid search, as it's exhaustive.
            # The number of trials is determined by the grid itself.
        ),
        run_config=ray.air.RunConfig(
            name="rf_covtype_gridsearch_tuning",
            stop={"training_iteration": 1}, # Our trainable reports in one iteration
            log_to_file=True,
            # local_dir="~/ray_results_covtype", # Optional: specify custom log directory
            failure_config=ray.air.FailureConfig(max_failures=2), # Allow a couple of trial failures
            verbose=1 # 0 = silent, 1 = progress bar and summary, 2 = detailed trial results
        ),
    )

    print(f"\nStarting Ray Tune with Grid Search for Covertype dataset.")
    print(f"Total trials to run: {num_trials}")
    print(f"Resources requested per trial: {resources_per_trial}")
    available_cpus = ray.cluster_resources().get("CPU", 0)
    max_parallel_trials = int(available_cpus / resources_per_trial.get("cpu", 1))
    print(f"Estimated maximum parallel trials based on cluster CPUs: {max_parallel_trials}")
    print(f"Ensure your Ray cluster has enough resources. Each trial will train a RandomForestClassifier.")
    print("This may take a significant amount of time depending on the dataset size and cluster resources.\n")


    # Run the tuning process
    start_tune_time = time.time()
    results_grid = tuner.fit()
    tune_duration = time.time() - start_tune_time

    # --- Get Results ---
    if results_grid.errors:
        print(f"There were errors during tuning: {len(results_grid.errors)} trials failed.")
        # for trial_result in results_grid:
        # if trial_result.error:
        # print(f"Error in trial {trial_result.path}: {trial_result.error}")

    best_result = results_grid.get_best_result(metric="accuracy", mode="max")

    if best_result:
        best_config = best_result.config
        best_accuracy = best_result.metrics["accuracy"]
        print("\n--- Tuning Complete ---")
        print(f"Total time for tuning: {tune_duration:.2f} seconds")
        print(f"Best hyperparameters found: {best_config}")
        print(f"Best accuracy on test set: {best_accuracy:.4f}")

        print("\nAll results (or results that did not error):")
        for i, result in enumerate(results_grid):
            if result.error:
                print(f"Trial {result.trial_id} (path: {result.path}) had an error: {result.error}")
            elif result.metrics:
                print(f"Trial {result.trial_id}: config={result.config}, accuracy={result.metrics.get('accuracy', 'N/A')}, training_time={result.metrics.get('training_time', 'N/A')}")
            else:
                print(f"Trial {result.trial_id}: No metrics reported (possibly due to error or early termination).")

    else:
        print("\n--- Tuning Complete ---")
        print(f"Total time for tuning: {tune_duration:.2f} seconds")
        print("No successful trials completed or no best result found.")


    # Clean up
    # You might want to keep Ray running if you plan to do more work,
    # or shut it down if you're done.
    # For script purposes, it's good practice to shut down if you initialized it.
    # However, if you connected to an existing cluster, you probably don't want the script to shut it down.
    # ray.shutdown() # Consider when to use this.
    print("\nScript finished. Ray may still be running if you started it externally.")
