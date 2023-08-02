import os
import random
import zipfile
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import numpy as np

import random
from typing import List


def report_dir(dir_path: str) -> None:
    """
    Walks through dir_path and prints its contents.

    Parameters:
        dir_path (str): The path to the directory.

    Returns:
        None
    """
    print("*********************************************************")
    for dir_path, dir_names, file_names in os.walk(dir_path):
        if ".git" not in dir_path:
            print(f"In {dir_path}\n"
                  f"Has {len(dir_names)} folder(s): {dir_names} \n"
                  f"{len(file_names)} file(s): {file_names}\n"
                  f"*********************************************************")
    return None


def unzip(file_path: str, delete_original: bool = False) -> None:
    """
    Unzips a file.

    Parameters:
        file_path (str): The path to the zip file.
        delete_original (bool): If True, deletes the original zip file after extraction.

    Returns:
        None
    """
    zip_ref = zipfile.ZipFile(file_path, "r")
    zip_ref.extractall()
    zip_ref.close()
    if delete_original:
        os.remove(file_path)
    print("Unzipped Successfully")
    return None


def get_lines(file_name: str) -> List[str]:
    """
    Read the contents of the file and return them as a list.

    Parameters:
        file_name (str): The name of the file.

    Returns:
        List[str]: A list of strings containing the lines of the file.
    """
    with open(file_name, "r") as f:
        return f.readlines()


def view_random_image(target_dir: str, target_class: str) -> None:
    """
    Plots a random image from target_dir and target_class.

    Parameters:
        target_dir (str): The path to the target directory.
        target_class (str): The name of the target class.

    Returns:
        None
    """
    random_img = random.sample(
        os.listdir(os.path.join(target_dir, target_class)), 1)
    img = mpimg.imread(os.path.join(target_dir, target_class, random_img[0]))
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")
    return None


def get_train_time(start, end, device=None, machine=None):
    """
    Prints difference between start and end time.

    Parameters:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    if device:
        print(
            f"\nTrain time on {machine} using PyTorch device {device}: {total_time:.3f} seconds\n")
    else:
        print(f"\nTrain time: {total_time:.3f} seconds\n")
    return round(total_time, 3)


def display_random_images(dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    """
    Displays n random images from a dataset.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to display images from.
        classes (List[str]): A list of class names.
        n (int): The number of images to display.
        display_shape (bool): If True, displays the shape of the image.
        seed (int): The random seed to use.

    Returns:
        None
    """

    # Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(
            f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # Set random seed
    if seed:
        random.seed(seed)

    # Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # Setup plot
    plt.figure(figsize=(16, 8))

    # Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample]["image"], \
                                 dataset[targ_sample]["label"]

        # Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    return None


def plot_model_results(results):
    """
    Plots the results of a model.

    Parameters:
        results (dict): A dictionary containing the results of a model.

    Returns:
        None
    """

    epochs = len(results['train_loss'])
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    axs[0].plot(range(epochs), results['train_loss'], label='Train Loss')
    axs[0].plot(range(epochs), results['test_loss'], label='Test Loss')
    axs[0].set_title('Loss')
    axs[0].legend()
    axs[1].plot(range(epochs), results['train_acc'], label='Train Accuracy')
    axs[1].plot(range(epochs), results['test_acc'], label='Test Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].legend()
    plt.show()
    return None


def info() -> None:
    """
    Prints the information about the functions in this module.

    Returns:
        None
    """
    data = {
        'Function': ['report_dir',
                     'unzip',
                     'get_lines',
                     'view_random_image',
                     'get_train_time',
                     'display_random_images',
                     'plot_model_results',


                     ],
        'Description': ['Walks through dir_path returning its contents',
                        'Unzips a file',
                        'Read the contents of the file and return them as a list',
                        'Visualize the difference in shape between two DataFrames',
                        'Gets difference between start and end time',
                        'Displays n random images from a dataset',
                        'Plots the results of a model',
                        
                        ]
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Print the DataFrame in a nice tabular format
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    return None
