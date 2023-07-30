import os
import random
import zipfile
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm


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


def plot_shape_difference(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    Visualize the difference in shape between two DataFrames.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        None
    """
    # Get the shapes of the DataFrames
    shape_df1 = df1.shape
    shape_df2 = df2.shape

    # Create a DataFrame to store the shape information
    df_shape = pd.DataFrame({'DataFrame': ['DataFrame 1', 'DataFrame 2'],
                             'Rows': [shape_df1[0], shape_df2[0]],
                             'Columns': [shape_df1[1], shape_df2[1]]})

    # Melt the DataFrame for visualization
    df_shape_melted = df_shape.melt(id_vars='DataFrame', var_name='Category',
                                    value_name='Count')

    # Create a bar plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Category', y='Count', hue='DataFrame', data=df_shape_melted,
                palette=['red', 'green'])
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Difference in Shape between the Two DataFrames')
    plt.legend(title='DataFrames', loc='upper right')
    plt.show()
    return None


def plot_binary_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Plots the distribution of the binary columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (List[str]): A list of column names (strings) representing binary columns to plot.

    Returns:
        None. The function plots the count distribution of the binary columns.

    Example:
        >> binary_columns = ["telecommuting", "has_company_logo", "has_questions", "fraudulent"]
        >> plot_binary_columns(df, binary_columns)

    This function takes a DataFrame and a list of binary column names, and then creates a 2x2 grid of subplots.
    Each subplot displays the count distribution of a binary column in the DataFrame. The function uses Seaborn's
    countplot to visualize the distribution of binary values (0 and 1) in each column. The x-axis represents the
    binary values (0 or 1), and the y-axis shows the count of occurrences for each value in the column.
    """

    # Set a larger figure size for better visibility
    plt.figure(figsize=(12, 8))
    # Create subplots with 2 columns and 2 rows
    fig, axes = plt.subplots(2, 2)

    for index, column in tqdm(enumerate(columns)):
        ax = axes[index // 2, index % 2]  # row index, columns index
        sns.countplot(x=column, data=df, ax=ax, palette='Set1')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {column}')

    # Add space between subplots
    plt.tight_layout()

    # Display the plots
    plt.show()
    return None


def info() -> None:
    """
    Prints the information about the functions in this module.

    Returns:
        None
    """
    data = {
        'Function': ['report_dir', 'unzip', 'get_lines', 'view_random_image',
                     'plot_shape_difference', 'plot_binary_columns'],
        'Description': ['Walks through dir_path returning its contents',
                        'Unzips a file',
                        'Read the contents of the file and return them as a list',
                        'Plots a random image from target_dir and target_class',
                        'Visualize the difference in shape between two DataFrames',
                        'Plots the distribution of the binary columns in the DataFrame'],
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Print the DataFrame in a nice tabular format
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    return None
