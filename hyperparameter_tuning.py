from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from preprocessing import preprocess
from train import train_loop

EPOCHS = 100
BATCH_SIZE = 1024
LR = 0.1
MOMENTUM = 0.9
NESTEROV = True
DECAY = 5e-4
PRINT_FREQ = 10
LAYERS = 40
WIDE_LAYERS = 2
DROPRATE = 0.3
NAME = "WideResNet-40-2"
NUM_CLASSES = 22
LOGNORM_TEMP = 1

HP1 = {"lr": [0.001, 0.01, 0.1]}
HP2 = {"lognorm_temperature": [0.01, 0.02, 0.05]}


def do_hyperparameter_evaluation(
    model_name: str,
    hyperparameter1: dict[str, list],
    hyperparameter2: dict[str, list],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    **kwargs,
) -> None:
    """
    Function iterates over the provided hyperparameters, trains a new model
    for every combination, evaluates it, then stores the result in a 2D
    accuracy matrix.

    Args:
        model_name (str): The model name with which the results will be saved.
        hyperparameter1 (dict[str, list]): Dictionary containing the 1st hyperparameter name and values.
        hyperparameter2 (dict[str, list]): Dictionary containing the 2nd hyperparameter name and values.
        train_images(np.ndarray): the images in the training dataset.
        train_labels (np.ndarray): the corresponding labels of the images in the training dataset.
        validation_images(np.ndarray): the images in the validation dataset.
        validation_labels (np.ndarray): the corresponding labels of the images in the validation dataset.
        **kwargs: Arguments that the models share during the training.
    """
    print("Started hyper parameter tuning")
    hp1_name = list(hyperparameter1.keys())[0]
    hp2_name = list(hyperparameter2.keys())[0]

    accuracy_matrix = np.full(
        (len(hyperparameter1[hp1_name]), len(hyperparameter2[hp2_name])), np.nan
    )

    for i, hp1 in enumerate(hyperparameter1[hp1_name]):
        print(f"Current {hp1_name}: {hp1}")
        validation_precision = 0

        for j, hp2 in enumerate(hyperparameter2[hp2_name]):
            print(f"Current {hp2_name}: {hp2}")
            hyperparameter_dic = {hp1_name: hp1, hp2_name: hp2}

            _, validation_precision = train_loop(
                train_images,
                train_labels,
                validation_images,
                validation_labels,
                model_name=f"{model_name}_{hp1}_{hp2}",
                **hyperparameter_dic,
                **kwargs,
            )

            accuracy_matrix[i, j] = validation_precision

        print(f"Validation precision for the best model: {validation_precision}")

    make_heatmap(accuracy_matrix, hyperparameter1, hyperparameter2)


def make_heatmap(
    accuracy_matrix: np.ndarray,
    hyperparameter1: dict[str, list],
    hyperparameter2: dict[str, list],
) -> None:
    """
    Saves the accuracy matrix by using a heatmap, which shows how performance varies for the models
        when using different hyperparameters.

    Args:
        accuracy_matrix: np.ndarray: 2D matrix containing accuracies for each hyperparameter combination.
        hyperparameter1: dict[str, list]- Dictionary containing the 1st hyperparameter name and values.
        hyperparameter2: dict[str, list]- Dictionary containing the 2nd hyperparameter name and values.
    """
    Path("images").mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    hp1_name = list(hyperparameter1.keys())[0]
    hp2_name = list(hyperparameter2.keys())[0]

    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt=".3f",
        yticklabels=hyperparameter1[hp1_name],
        xticklabels=hyperparameter2[hp2_name],
        cmap="viridis",
    )

    plt.title("Heatmap")
    plt.ylabel(hp1_name)
    plt.xlabel(hp2_name)
    plt.savefig(f"images/heatmap_{hp1_name}_{hp2_name}")
    plt.close()


# Running the hyperparameter tuning
if __name__ == "__main__":
    # -------- Preprocess ID data --------
    (
        ID_train_images,
        ID_val_images,
        _,
        ID_train_labels,
        ID_val_labels,
        _,
    ) = preprocess("datasets/sea_creatures")

    # -------- Hyper-parameter tuning --------
    do_hyperparameter_evaluation(
        model_name=NAME,
        hyperparameter1=HP1,
        hyperparameter2=HP2,
        train_images=ID_train_images,
        train_labels=ID_train_labels,
        validation_images=ID_val_images,
        validation_labels=ID_val_labels,
        loss="logit-normalization",
        num_classes=NUM_CLASSES,
        num_layers=LAYERS,
        num_wide_layers=WIDE_LAYERS,
        droprate=DROPRATE,
        decay=DECAY,
        optimizer_momentum=MOMENTUM,
        nesterov=NESTEROV,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        print_freq=PRINT_FREQ,
    )
