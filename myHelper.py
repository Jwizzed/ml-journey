import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_results(y_true, y_pred):
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


def compare_results_df(old_pred, new_pred, true_val):
    """
    Compare old and new predictions with true values
    **Note: Diff is New - Old
    :param old_pred:
    :param new_pred:
    :param true_val:
    :return: dataframe
    """
    old_results = calculate_results(true_val, old_pred)
    new_results = calculate_results(true_val, new_pred)

    df = pd.DataFrame(
        columns=["Old", "New", "Diff(%)"],
        index=["accuracy", "precision", "recall", "f1"])
    df["Old"] = old_results
    df["New"] = new_results
    df["Diff(%)"] = round((df["New"] - df["Old"]) * 100, 4)
    return df


