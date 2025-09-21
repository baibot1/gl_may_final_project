# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train dataset directory")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset directory")
    parser.add_argument("--model_output", type=str, required=True, help="Path of output model directory")
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="The number of trees in the forest",
    )
    # Use -1 sentinel to represent 'no maximum depth' to satisfy AML input parsing
    parser.add_argument(
        "--max_depth",
        type=int,
        default=-1,
        help="The maximum depth of the tree (-1 means no limit)",
    )
    return parser.parse_args()


def main(args):
    """Read train and test datasets, train model, evaluate model, save trained model"""

    # Resolve CSV paths inside the provided directories
    train_csv = Path(args.train_data) / "train.csv"
    test_csv = Path(args.test_data) / "test.csv"

    # Read train and test data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Split the data into features (X) and target (y)
    y_train = train_df["price"]
    X_train = train_df.drop(columns=["price"])
    y_test = test_df["price"]
    X_test = test_df.drop(columns=["price"])

    # Map -1 sentinel to None for sklearn
    max_depth = None if (args.max_depth is None or int(args.max_depth) < 0) else int(args.max_depth)

    # Initialize and train a RandomForest Regressor
    model = RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", int(args.n_estimators))
    mlflow.log_param("max_depth", -1 if max_depth is None else int(max_depth))

    # Predict and evaluate
    yhat_test = model.predict(X_test)
    mse = mean_squared_error(y_test, yhat_test)
    print("Mean Squared Error of RandomForest Regressor on test set: {:.2f}".format(mse))
    mlflow.log_metric("MSE", float(mse))

    # Save the model (directory path; will be created if it doesn't exist)
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    for line in [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth (raw arg): {args.max_depth}",
    ]:
        print(line)

    main(args)

    mlflow.end_run()
