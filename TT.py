import os
import numpy as np
import random
from typing import List
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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
