from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

## "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"

## train.csv and test.csv generated from automl.ipynb and we are using this to maintain the consistancy and for proper comparison

run = Run.get_context()


def main():
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    y_train = train.pop("DEATH_EVENT")
    x_train = train
    y_test = test.pop("DEATH_EVENT")
    x_test = test
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--solver', type=str, default='lbfgs', help="Algorithm to train the model")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Algorithm: ", args.solver)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver=args.solver).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')

if __name__ == '__main__':
    main()