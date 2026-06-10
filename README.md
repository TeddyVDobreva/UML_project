# Uncertainty in Machine Learning Project: OOD Detection.

## Description
This project focuses on performing an Out-of-Distribution (OOD) detection task.

We use two separate datasets:
- sea creatures - The In-Distribution (ID) dataset, which is used to train and evaluate the model
- reptiles - The OOD dataset, which is used as a test set on the trained ID model

We train the two ID models: one with cross entropy loss and one with logit normalization loss. 

We tune the following hyperparameters: learning rate and logit norm temperature.

We evaluate the performance of the model by returnning the FPR95, AUROC, AUPR, and accuracy.

## Prerequisites 

- uv installed

## Code Structure

```
datasets/                   Folder with the images of our dataset
    reptiles/               Folder containing all images of reptiles - OOD dataset (9 classes).
    sea_creatures/          Folder containing all images of sea creatures - ID dataset (22 classes).
images/                     Folder with code-generated images and plots
runs/WideResNet-40-2        Folder with the stored results.
models.py                   File with the functions building the Wide Residual Network.
evaluation.py               File with evaluation functions for the Wide Residual Network.
hyperparameter_tuning.py    File with hyperparameter tuning functions.
main.py                     Main file which loads and preprocesses the data, trains, and evaluates the models
preprocessing.py            File with the preprocessing functions.
train.py                    File with the training functions.
```

## Running the model

Clone the repository

```bash
    git clone https://github.com/TeddyVDobreva/UML_project.git
    cd UML_project/
```

Create a virtual environment and install dependencies

```bash
    uv sync
```

Activate the virtual environment

```bash
    source .venv/bin/activate
```

In order to run the model, run `main.py`. 

```bash
    python3 main.py
```