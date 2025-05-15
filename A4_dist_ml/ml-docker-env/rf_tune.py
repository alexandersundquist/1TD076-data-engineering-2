import sys # For checking memory
import numpy as np
import pandas as pd # Assuming fetch_covtype(as_frame=True) returns pandas DataFrame
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from ray import tune
import ray
import gc

def train_model(config):
    # Load data
    print("Loading data...")
    covtype_data = fetch_covtype(as_frame=True) # data.data is a DataFrame, data.target is a Series
    X_full_df = covtype_data.data
    y_full_series = covtype_data.target

    # --- Memory Check (Initial Full Data as Pandas DataFrame) ---
    # For Pandas DataFrame, .memory_usage(deep=True).sum() is more accurate
    x_full_mem_mb = X_full_df.memory_usage(deep=True).sum() / (1024**2)
    y_full_mem_mb = y_full_series.memory_usage(deep=True).sum() / (1024**2)
    print(f"Initial X_full (Pandas) memory: {x_full_mem_mb:.2f} MB")
    print(f"Initial y_full (Pandas) memory: {y_full_mem_mb:.2f} MB")

    # Convert to NumPy and change dtype for X
    X_full_np = X_full_df.to_numpy(dtype=np.float32)
    y_full_np = y_full_series.to_numpy() # Target is likely int, usually fine

    del X_full_df, y_full_series # Delete pandas objects
    gc.collect()

    # --- Memory Check (Full Data as NumPy arrays) ---
    print(f"Full X_np (float32) memory: {X_full_np.nbytes / (1024**2):.2f} MB")
    print(f"Full y_np memory: {y_full_np.nbytes / (1024**2):.2f} MB")

    # --- AGGRESSIVE SUBSAMPLING ---
    # Try with a very small fraction, e.g., 5% or 2%
    # If your classes are imbalanced, stratify is important.
    # If dataset is huge, even train_test_split can use memory.
    # Adjust test_size for the fraction you want to KEEP for training
    # e.g. sample_fraction = 0.05 means we want 5% of the data
    sample_fraction = 0.05 # Start with 5%, if still OOM, try 0.02 (2%)
    try:
        # Using train_test_split to get a stratified sample. We'll use X_sample, y_sample.
        # The first parts (X_discard, y_discard) are the larger chunk we're not using.
        _, X_sample, _, y_sample = train_test_split(
            X_full_np, y_full_np, test_size=sample_fraction, random_state=42, stratify=y_full_np
        )
        print(f"Subsampled to {sample_fraction*100}% of data.")
        print(f"X_sample memory: {X_sample.nbytes / (1024**2):.2f} MB")
    except ValueError as e:
        # This can happen if sample_fraction is too small for stratification across all classes for cv
        print(f"Error during train_test_split (possibly due to stratification with small sample): {e}")
        print("Attempting non-stratified sample or using full data with caution.")
        # Fallback or error, for now, let's just take a simple slice if stratification fails
        # This is a basic fallback, stratification is preferred.
        num_samples_to_take = int(len(X_full_np) * sample_fraction)
        if num_samples_to_take < 1: num_samples_to_take = 1 # Ensure at least one sample
        X_sample = X_full_np[:num_samples_to_take]
        y_sample = y_full_np[:num_samples_to_take]


    del X_full_np, y_full_np # Delete the full NumPy arrays immediately
    gc.collect()
    print("Full dataset deleted from memory.")

    X_ref = ray.put(X_sample)
    y_ref = ray.put(y_sample)
    del X_sample, y_sample # Delete local copies after sending to object store
    gc.collect()

    X_data = ray.get(X_ref)
    y_data = ray.get(y_ref)
    print(f"Data for training (from Ray object store): X shape {X_data.shape}, y shape {y_data.shape}")

    # Define the model
    clf = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        n_jobs=1,       # Keep this at 1
        random_state=42,
        # Potentially add min_samples_split or min_samples_leaf if depth is an issue
        # min_samples_leaf=config.get("min_samples_leaf", 1) # If you add this to search space
    )
    print("Model defined. Starting cross-validation...")

    try:
        scores = cross_val_score(clf, X_data, y_data, cv=3, scoring='accuracy') # cv=3 or even cv=2 for less memory
        mean_score = np.mean(scores)
        print(f"Cross-validation scores: {scores}, Mean accuracy: {mean_score}")
    except ValueError as e:
        # This can happen if a fold ends up with too few samples of a certain class,
        # especially with aggressive subsampling.
        print(f"ValueError during cross_val_score: {e}. Reporting 0 accuracy.")
        mean_score = 0.0 # Report a bad score to Ray Tune
    except Exception as e:
        print(f"An unexpected error occurred during cross_val_score: {e}. Reporting 0 accuracy.")
        mean_score = -1.0 # Or some other indicator of failure

    tune.report(mean_accuracy=mean_score)

    del X_data, y_data, X_ref, y_ref, clf, scores # Extensive cleanup
    gc.collect()
    print("Trial cleanup complete.")
