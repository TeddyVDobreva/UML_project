import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = (64, 64)
EXCLUDE_CLASSES = {"Turtle_Tortoise"}


def load_resize_images(dataset_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads images from the specified directories and returns resized image data,
    labels, and class names.
    Skips the class 'Turtle_Tortoise'.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        images (np.ndarray): Array of image data.
        labels (np.ndarray): Array of corresponding labels for the images.
        class_names (list[str]): List of class names corresponding to the labels.
    """
    images = []
    labels = []
    class_names = []

    for class_name in sorted(os.listdir(dataset_path)):
        if class_name in EXCLUDE_CLASSES:
            continue

        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        label = len(class_names)
        class_names.append(class_name)

        for dataset_name in sorted(os.listdir(class_path)):
            img_path = os.path.join(class_path, dataset_name)

            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(label)
            except Exception:
                continue

    return np.array(images), np.array(labels), class_names


def normalize(images: np.ndarray) -> np.ndarray:
    """
    Normalizes image pixel values to [0, 1].

    Args:
        images (np.ndarray): Array of images with integer or float pixel values.

    Returns:
        np.ndarray: Array of images with float pixel values in [0, 1].
    """
    return images.astype(np.float32) / 255.0


def split(
    images: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits images and labels into train, validation, and test sets (70:15:15).

    Args:
        images (np.ndarray): Array of images.
        labels (np.ndarray): Array of corresponding labels.

    Returns:
        tuple: The dataset split into train, val, and test sets as np.ndarrays.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.30, random_state=42, stratify=labels, shuffle=True
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp, shuffle=True
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(dataset_path: str) -> tuple:
    """
    Loads the data of a dataset, resizes the images, normalizes them,
    and splits into train, val, and test sets.

    Args:
        dataset (str): Dataset to be preprocessed.

    Returns:
        tuple: The split of preprocessed dataset.
    """
    images, labels, class_names = load_resize_images(dataset_path)
    norm_images = normalize(images)

    return split(norm_images, labels)
