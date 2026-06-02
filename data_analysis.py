import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


if __name__ == "__main__":
    plot_composition("reptiles", "datasets/reptiles")
    plot_composition("sea_creatures", "datasets/sea_creatures")
