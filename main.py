import json

import numpy as np
import torch
from evaluation import eval_acc, eval_aupr, eval_auroc, eval_fpr95
from models import WideResNet
from preprocessing import preprocess
from train import get_scores, train_loop

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
LOGNORM_TEMP = 0.05


def main():
    # -------- Data Preprocessing --------
    (
        ID_train_images,
        ID_val_images,
        ID_test_images,
        ID_train_labels,
        ID_val_labels,
        ID_test_labels,
    ) = preprocess("datasets/sea_creatures")

    (
        OOD_train_images,
        OOD_val_images,
        OOD_test_images,
        OOD_train_labels,
        OOD_val_labels,
        OOD_test_labels,
    ) = preprocess("datasets/reptiles")

    print("Preprocessing done!")

    # -------- Training the models and evaluate on the validation sets --------
    # train baseline CE model
    ID_model_ce, _ = train_loop(
        ID_train_images,
        ID_train_labels,
        ID_val_images,
        ID_val_labels,
        loss="cross-entropy",
        num_classes=NUM_CLASSES,
        model_name=NAME,
        num_layers=LAYERS,
        num_wide_layers=WIDE_LAYERS,
        droprate=DROPRATE,
        lr=LR,
        decay=DECAY,
        optimizer_momentum=MOMENTUM,
        nesterov=NESTEROV,
        lognorm_temperature=LOGNORM_TEMP,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        print_freq=PRINT_FREQ,
    )

    # Load the best LogiNorm model from hyperparameter tuning.
    best_model_path = (
        "runs/WideResNet-40-2_0.1_0.05/_logit-normalizationmodel_best.pth.tar"
    )
    ID_model_ln = WideResNet(
        LAYERS,
        NUM_CLASSES,
        WIDE_LAYERS,
        dropRate=DROPRATE,
    )
    ID_model_ln.load_state_dict(torch.load(best_model_path)["state_dict"])

    # ---------------- Evaluate the models ----------------

    # Generate the true labels for the combined ID and OOD test sets
    y_OOD = np.concatenate(
        [np.ones(len(ID_test_labels)), np.zeros(len(OOD_test_labels))]
    )

    # Get the predicted scores for the ID and OOD test sets
    ID_scores_ce, ID_preds_ce = get_scores(ID_model_ce, ID_test_images)
    ID_scores_ln, ID_preds_ln = get_scores(ID_model_ln, ID_test_images)

    OOD_scores_ce, OOD_preds_ce = get_scores(ID_model_ce, OOD_test_images)
    OOD_scores_ln, OOD_preds_ln = get_scores(ID_model_ln, OOD_test_images)

    combined_scores_ce = np.concatenate([ID_scores_ce, OOD_scores_ce])
    combined_scores_ln = np.concatenate([ID_scores_ln, OOD_scores_ln])

    # FPR95
    fpr95_CE = eval_fpr95(
        ID_test_labels,
        ID_scores_ce,
        OOD_test_labels,
        OOD_scores_ce,
    )
    fpr95_LN = eval_fpr95(
        ID_test_labels,
        ID_scores_ln,
        OOD_test_labels,
        OOD_scores_ln,
    )

    # AUROC
    auroc_CE = eval_auroc(y_OOD, combined_scores_ce, "cross entropy")
    auroc_LN = eval_auroc(y_OOD, combined_scores_ln, "logit normalization")

    # AUPR
    aupr_CE = eval_aupr(y_OOD, combined_scores_ce, "cross entropy")
    aupr_LN = eval_aupr(y_OOD, combined_scores_ln, "logit normalization")

    # Accuracy of the model on the ID test set
    acc_CE = eval_acc(ID_test_labels, ID_preds_ce)
    acc_LN = eval_acc(ID_test_labels, ID_preds_ln)

    # ---------------- Save results ----------------
    results_CE = {
        "loss:": "CE",
        "FPR95:": fpr95_CE,
        "AUROC:": auroc_CE,
        "AUPR:": aupr_CE,
        "Accuracy:": acc_CE,
    }

    with open("results_CE.json", "w") as f:
        json.dump(results_CE, f, indent=4)

    print("Results of WRN-40-2 with Cross-Entropy loss saved to results_CE.json")

    results_LN = {
        "loss:": "LN",
        "FPR95:": fpr95_LN,
        "AUROC:": auroc_LN,
        "AUPR:": aupr_LN,
        "Accuracy:": acc_LN,
    }

    with open("results_LN.json", "w") as f:
        json.dump(results_LN, f, indent=4)

    print("Results of WRN-40-2 with Logit-Normalization loss saved to results_CE.json")


# Run the main function
if __name__ == "__main__":
    main()
