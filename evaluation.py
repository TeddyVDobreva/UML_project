import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def load_df(y_test: np.ndarray, scores: np.ndarray, loss_name: str) -> pd.DataFrame:
    """
    Loads the model predictions and true labels into a DataFrame.

    Args:
        y_test (np.ndarray): the true labels for the test set.
        scores (np.ndarray): the predicted scores for the test set.
        loss_name (str): the name of the loss function.

    Returns:
        pd.DataFrame: a DataFrame containing the true labels and model predictions.
    """
    return pd.DataFrame({"True": y_test, f"{loss_name}": scores})


def eval_fpr95(
    ID_y_test: np.ndarray,
    ID_scores: np.ndarray,
    OOD_y_test: np.ndarray,
    OOD_scores: np.ndarray,
    loss_name: str,
) -> str:
    """
    Evaluation metric that shows the false positive rate (FPR) of OOD examples,
    when the true positive rate (TPR) of ID examples is 95%.

    Args:
        ID_y_test (np.ndarray): the true labels for the ID test set.
        ID_scores (np.ndarray): the predicted scores for the ID test set.
        OOD_y_test (np.ndarray): the true labels for the OOD test set.
        OOD_scores (np.ndarray): the predicted scores for the OOD test set.
        loss_name (str): the name of the loss function.

    Returns:
        str: the FPR95 score.
    """
    y_true = np.concatenate([np.ones(len(ID_y_test)), np.zeros(len(OOD_y_test))])
    y_scores = np.concatenate([ID_scores, OOD_scores])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]

    return f"FPR95 for WRN with {loss_name} loss: {fpr95:.4f}"


def eval_auroc(y_test: np.ndarray, scores: np.ndarray, loss_name: str) -> str:
    """
    Evaluation metric that shows the area under the ROC curve (AUROC).

    Args:
        y_test (np.ndarray): the true labels for the test set.
        scores (np.ndarray): the predicted scores for the test set.
        loss_name (str): the name of the loss function.

    Returns:
        str: the AUROC score.
    """
    test_df = load_df(y_test, scores, loss_name)

    fpr, tpr, _ = roc_curve(test_df["True"], test_df[loss_name])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{loss_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "r--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for the Wide Residual Network model with {loss_name} loss")
    plt.legend()
    plt.savefig(f"images/auroc_{loss_name}.png")
    plt.close()

    return f"AUROC for WRN with {loss_name} loss: {roc_auc:.4f}"


def eval_aupr(y_test: np.ndarray, scores: np.ndarray, loss_name: str) -> str:
    """
    Evaluation metric that shows the area under the precision-recall curve (AUPR).

    Args:
        y_test (np.ndarray): the true labels for the test set.
        scores (np.ndarray): the predicted scores for the test set.
        loss_name (str): the name of the loss function.

    Returns:
        str: the AUPR score.
    """
    test_df = load_df(y_test, scores, loss_name)

    precision, recall, _ = precision_recall_curve(
        test_df["True"], test_df[f"{loss_name}"]
    )
    pr_ap = average_precision_score(test_df["True"], test_df[loss_name])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"{loss_name} (AP = {pr_ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision-Recall Curve for the Wide Residual Network model with {loss_name} loss"
    )
    plt.legend()
    plt.savefig(f"images/aupr_{loss_name}.png")
    plt.close()

    return f"AUPR for WRN with {loss_name} loss: {pr_ap:.4f}"


def eval_acc(y_test: np.ndarray, preds: np.ndarray, loss_name: str) -> str:
    """
    Evaluation metric that shows the accuracy of the model.

    Args:
        y_test (np.ndarray): the true labels for the test set.
        preds (np.ndarray): the predicted labels for the test set.
        loss_name (str): the name of the loss function.

    Returns:
        str: the accuracy score.
    """
    test_df = load_df(y_test, preds, loss_name)

    acc = accuracy_score(test_df["True"], test_df[f"{loss_name}"])

    return f"Accuracy for {loss_name}: {acc:.4f}"
