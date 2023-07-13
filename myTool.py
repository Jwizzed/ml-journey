import os
import random
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import pipeline


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


def compare_results_df(old_pred: np.ndarray, new_pred: np.ndarray,
                       true_val: np.ndarray) -> pd.DataFrame:
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


def get_lines(file_name: str) -> List[str]:
    """
    Read the contents of the file and return them as a list
    :param file_name:
    :return:
    """
    with open(file_name, "r") as f:
        return f.readlines()


def evaluate_preds(y_true, y_pred):
    """
    Evaluates the difference between true labels[1:] and predicted labels
    :param y_true:
    :param y_pred:
    :return: None
    >> evaluate_preds(y_test[1:], naive_forecast) [1:] in here bc we need the same shape
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true,
                                              y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mae / tf.reduce_mean(tf.abs(
        y_true[1:] - y_true[:-1]))  # [1:] is a formula, don't worry about it.

    if mae.ndim > 0:  # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}


def view_random_image(target_dir: str, target_class: str):
    """
    Plots a random image from target_dir and target_class
    :param target_dir:
    :param target_class:
    :return:
    >> view_random_image(view_random_image(target_dir="pizza_steak/train/",
                        target_class="steak"))
    """

    random_img = random.sample(os.listdir(target_dir + target_class), 1)
    img = mpimg.imread(target_dir + target_class + "/" + random_img[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")
    return img


def history_plot(model_history):
    """
    Plots model history of accuracy and loss over time
    :param model_history: History object containing training/validation metrics
    :return: None
    >> history_plot(history_3)
    """
    tacc = model_history.history['accuracy']
    tloss = model_history.history['loss']
    vacc = model_history.history['val_accuracy']
    vloss = model_history.history['val_loss']

    epochs = np.arange(1, len(tacc) + 1)

    index_loss = np.argmin(vloss)
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]

    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    axes[0].plot(epochs, tloss, 'r', label='Training loss')
    axes[0].plot(epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1, val_lowest, s=150, c='blue',
                    label=f'Best epoch = {index_loss + 1}')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1, acc_highest, s=150, c='blue',
                    label=f'Best epoch = {index_acc + 1}')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_image_detector(img_url: str = "") -> None:
    """
    Plot object detection boxes by using huggingface's transformers pipeline.
    :param img_url: String of image url
    :return:
    """

    image_data = requests.get(img_url, stream=True).raw
    image = Image.open(image_data)

    # Allocate a pipeline for object detection
    object_detector = pipeline('object-detection')
    results = object_detector(image)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for result in results:
        label = result['label']
        score = result['score']
        box = result['box'].values()
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline='red')
        draw.text((x1, y1 - 10), f'{label}: {score:.2f}', fill='green')

    # Display the image with bounding boxes
    image.show()
    return None
