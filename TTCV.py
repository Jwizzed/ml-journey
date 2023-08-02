import os
import random
import zipfile
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


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

                     ],
        'Description': ['Walks through dir_path returning its contents',
                        'Unzips a file',
                        'Read the contents of the file and return them as a list',
                        'Visualize the difference in shape between two DataFrames',
                        'Gets difference between start and end time',

                        ]
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Print the DataFrame in a nice tabular format
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    return None
