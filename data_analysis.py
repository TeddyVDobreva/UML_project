import os
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def _plot_gray_hist(images: list[Image], dataset_name) -> None:
    """
    Plots and saves the gray histogram of an image.

    Args:
        images (list[Image]): a list of images.

    Returns:
        None: Plots and saves the histogram.
    """
    for i in range(len(images)):
        g = cv2.cvtColor(np.array(images[i]), cv2.COLOR_BGR2GRAY)
        plt.subplot(122), plt.hist(g.ravel(), 256, [0, 256], color="k")
        plt.savefig(f"images/histograms_{dataset_name}/gray_histogram_{i}")
        plt.close()


def _open_image_files(path: str) -> list[Image]:
    """
    Opens the image files in a directory.

    Args:
        path (str): The path to a folder with the image data.

    Returns:
        list[Image]: List of all the images loaded from the provided path.
    """
    p = Path(path)
    images = []

    # Find an image file in the dataset
    for file in p.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".png", ".jpg"}:
            images.append(Image.open(file))

    if not images:
        raise FileNotFoundError(f"No images found in {p}")

    return images


def _plot_image_dims(images: list[Image], dataset_name: str) -> None:
    """
    Plots the dimensions of the images in a dataset.

    Args:
        images (list[Image]): List of images for which to plot dimensions.
        dataset_name (str): the name of the dataset.

    Returns:
        None: Displays and saves the plots.
    """
    counter = Counter()

    for image in images:
        counter.update([image.size])

    # Plot resolutions
    most_common = counter.most_common(15)
    resolutions = [f"{w[0]}x{w[1]}" for w, _ in most_common]
    counts = [c for _, c in most_common]

    # Plot bar chart of top 15 resolutions
    plt.figure(figsize=(8, 6))
    plt.bar(resolutions, counts)
    plt.xticks(rotation=45)
    plt.title("Top 15 Resolutions in Images")
    plt.savefig(f"images/im_dimensions_{dataset_name}.png")
    plt.close()


def do_analysis(path: str) -> None:
    """
    Performs analysis on the images in a dataset by plotting their dimensions.

    Args:
        path (str): The path to a folder with the image data.

    Returns:
        None: Displays and saves the plots of image dimensions.
    """
    dataset_name = path.split("/")[-1]
    plot_composition(dataset_name, path)
    images = _open_image_files(path)
    _plot_image_dims(images, dataset_name)
    _plot_gray_hist(images, dataset_name)


def plot_composition(dataset_name: str, dataset_path: str) -> None:
    """
    Plots the composition of a dataset as a bar chart.

    Args:
        dataset_name (str): The name of the dataset (used for labeling the plot).
        dataset_path (str): The path to the dataset directory.

    Returns:
        None: Displays and saves the plot of dataset composition.

    Raises:
        ValueError: If dataset_name is not 'reptiles' or 'sea_creatures
                    or if dataset_path is not a valid path to the dataset directory.
    """
    # Validate inputs
    if dataset_name.lower() not in ["reptiles", "sea_creatures"]:
        raise ValueError("dataset_name must be either 'reptiles' or 'sea_creatures'")
    if dataset_path is None or not os.path.exists(dataset_path):
        raise ValueError("dataset_path must be a valid path to the dataset directory")

    # Create Path object for dataset directory
    dataset_image_dir = Path(dataset_path)

    # Get filepaths and labels
    filepaths = (
        list(dataset_image_dir.glob(r"**/*.JPG"))
        + list(dataset_image_dir.glob(r"**/*.jpg"))
        + list(dataset_image_dir.glob(r"**/*.png"))
        + list(dataset_image_dir.glob(r"**/*.PNG"))
    )

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name="Filepath").astype(str)
    labels = pd.Series(labels, name="Label")

    # Concatenate filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)

    label_counts = image_df["Label"].value_counts()

    # Format the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(y=label_counts.index, x=label_counts.values, palette="viridis")

    plt.title(
        "Dataset Composition: Samples per Category",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Count", fontsize=14)
    plt.ylabel(f"{dataset_name.capitalize()} Species", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.savefig(f"images/{dataset_name}_dataset_composition.png")


# Running the analysis for both datasets
if __name__ == "__main__":
    do_analysis("datasets/reptiles")
    do_analysis("datasets/sea_creatures")
