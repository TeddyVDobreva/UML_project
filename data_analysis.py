import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


def _open_image_files(path: str) -> list[Image]:
    """
    Opens the image files in a directory.

    Args:
        path (str): The path to a folder with the image data

    Returns:
        list[Image]: List of all the images loaded from the provided path
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


def _plot_image_dims(images: list[Image]) -> None:
    """
    Plots the dimensions of the images in a dataset.

    Args:
        images (list[Image]): List of images for which to plot dimensions.

    Returns:
        None: Displays and saves the plots.
    """
    counter = Counter()
    heights = Counter()
    widths = Counter()

    for image in images:
        counter.update([image.size])
        widths.update([image.size[0]])
        heights.update([image.size[1]])

    # Plot resolutions
    most_common = counter.most_common(50)
    resolutions = [f"{w[0]}x{w[1]}" for w, _ in most_common]
    counts = [c for _, c in most_common]

    # Plot bar chart of top 50 resolutions
    plt.figure()
    plt.bar(resolutions, counts)
    plt.xticks(rotation=45)
    plt.title(f"Top {50} Resolutions in Images")
    plt.savefig("images/im_dimensions.png")
    plt.close()

    # Plot histogram of image heights
    plt.figure()
    plt.bar(heights.keys(), heights.values())
    plt.xticks(rotation=45)
    plt.title("Histogram of image heights")
    plt.savefig("images/height_hist.png")
    plt.close()

    # Plot histogram of image widths
    plt.figure()
    plt.bar(widths.keys(), widths.values())
    plt.xticks(rotation=45)
    plt.title("Histogram of image widths")
    plt.savefig("images/width_hist.png")
    plt.close()


def do_analysis(path: str) -> None:
    """
    Performs analysis on the images in a dataset by plotting their dimensions.

    Args:
        path (str): The path to a folder with the image data.

    Returns:
        None: Displays and saves the plots of image dimensions.
    """
    images = _open_image_files(path)
    _plot_image_dims(images)


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
    plt.show()
    plt.savefig(f"images/{dataset_name}_dataset_composition.png")


# Running the analysis for both datasets
if __name__ == "__main__":
    plot_composition("reptiles", "datasets/reptiles")
    plot_composition("sea_creatures", "datasets/sea_creatures")

    do_analysis("datasets/reptiles")
    do_analysis("datasets/sea_creatures")
