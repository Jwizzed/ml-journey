import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os


def calculate_results(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array
    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


def compare_results_df(old_pred: np.ndarray, new_pred: np.ndarray, true_val: np.ndarray) -> pd.DataFrame:
    """
    Compare old and new predictions with true values
    **Note: Diff is New - Old
    :param old_pred: old predictions
    :param new_pred: new predictions
    :param true_val: true values
    :return: dataframe of old, new and diff
    """
    old_results = calculate_results(true_val, old_pred)
    new_results = calculate_results(true_val, new_pred)

    df = pd.DataFrame(
        columns=["Old", "New", "Diff"],
        index=["accuracy", "precision", "recall", "f1"])
    df["Old"] = old_results
    df["New"] = new_results
    df["Diff"] = round((df["New"] - df["Old"]), 4)
    return df


def report_dir(dir_path: str) -> None:
    """
    Walks through dir_path returning its contents
    :param dir_path:
    :return: None
    """
    print("*********************************************************")
    for dir_path, dir_names, file_names in os.walk(dir_path):
        if ".git" not in dir_path:
            print(f"In {dir_path}\n"
                  f"Has {len(dir_names)} folder(s): {dir_names} \n"
                  f"{len(file_names)} file(s): {file_names}\n"
                  f"*********************************************************")


def unzip(file_path: str) -> None:
    """
    Unzips a file
    :param file_path:
    :return: None
    """
    import zipfile
    zip_ref = zipfile.ZipFile(file_path, "r")
    zip_ref.extractall()
    zip_ref.close()
    return None
