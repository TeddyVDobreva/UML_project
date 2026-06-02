import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Reptiles dataset
reptiles_dataset = "datasets/reptiles"

reptiles_image_dir = Path(reptiles_dataset)

# Get filepaths and labels
filepaths = (
    list(reptiles_image_dir.glob(r"**/*.JPG"))
    + list(reptiles_image_dir.glob(r"**/*.jpg"))
    + list(reptiles_image_dir.glob(r"**/*.png"))
    + list(reptiles_image_dir.glob(r"**/*.PNG"))
)

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name="Filepath").astype(str)
labels = pd.Series(labels, name="Label")

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)

label_counts = image_df["Label"].value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(y=label_counts.index, x=label_counts.values, palette="viridis")

plt.title(
    "Dataset Composition: Samples per Category", fontsize=18, fontweight="bold", pad=20
)
plt.xlabel("Count", fontsize=14)
plt.ylabel("Reptile Species", fontsize=14)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.show()
plt.savefig("images/reptiles_dataset_composition.png")


# Sea creatures dataset
sea_creatures_dataset = "datasets/sea_creatures"

sea_creatures_image_dir = Path(sea_creatures_dataset)

# Get filepaths and labels
filepaths = (
    list(sea_creatures_image_dir.glob(r"**/*.JPG"))
    + list(sea_creatures_image_dir.glob(r"**/*.jpg"))
    + list(sea_creatures_image_dir.glob(r"**/*.png"))
    + list(sea_creatures_image_dir.glob(r"**/*.PNG"))
)

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name="Filepath").astype(str)
labels = pd.Series(labels, name="Label")

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)

label_counts = image_df["Label"].value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(y=label_counts.index, x=label_counts.values, palette="viridis")

plt.title(
    "Dataset Composition: Samples per Category", fontsize=18, fontweight="bold", pad=20
)
plt.xlabel("Count", fontsize=14)
plt.ylabel("Sea Creature Species", fontsize=14)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.show()
plt.savefig("images/sea_creatures_dataset_composition.png")
