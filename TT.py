import os
import numpy as np
import random
from typing import List
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def unzip(file_path: str, delete_original: bool = False) -> None:
    """
    Unzips a file
    :param delete_original:
    :param file_path:
    :return: None
    """
    import zipfile
    zip_ref = zipfile.ZipFile(file_path, "r")
    zip_ref.extractall()
    zip_ref.close()
    if delete_original:
        os.remove(file_path)
    print("Unzipped Successfully")
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


def plot_shape_difference(df1, df2):
    """
    Visualize the difference in shape between two DataFrames
    :param df1:
    :param df2:
    :return:
    >> visualize_shape_difference(orig_df, df)
    """
    # Get the shapes of the DataFrames
    shape_df1 = df1.shape
    shape_df2 = df2.shape

    # Create a DataFrame to store the shape information
    df_shape = pd.DataFrame({'DataFrame': ['DataFrame 1', 'DataFrame 2'],
                             'Rows': [shape_df1[0], shape_df2[0]],
                             'Columns': [shape_df1[1], shape_df2[1]]})

    # Melt the DataFrame for visualization
    df_shape_melted = df_shape.melt(id_vars='DataFrame', var_name='Category', value_name='Count')

    # Create a bar plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Category', y='Count', hue='DataFrame', data=df_shape_melted, palette=['red', 'green'])
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Difference in Shape between the Two DataFrames')
    plt.legend(title='DataFrames', loc='upper right')
    plt.show()


def info():
    """
    Prints the information about the functions in this module
    :return:
    """
    data = {
        'Function': ['report_dir', 'unzip', 'get_lines', 'view_random_image',
                     'plot_shape_difference'],
        'Description': ['Walks through dir_path returning its contents',
                        'Unzips a file',
                        'Read the contents of the file and return them as a list',
                        'Plots a random image from target_dir and target_class',
                        'Visualize the difference in shape between two DataFrames']
    }

    # Create the DataFrame
    df = pd.DataFrame(data)
    print(df)
