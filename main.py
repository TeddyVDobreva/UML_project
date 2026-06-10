import numpy as np

from data_analysis import do_analysis
from evaluation import eval_acc, eval_aupr, eval_auroc, eval_fpr95
from preprocessing import preprocess
from train import get_scores, train_loop

EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1  # hp tuning 0.1, 0.05 0.01, 0.005, 0.001
MOMENTUM = 0.9
NESTEROV = True
DECAY = 5e-4
PRINT_FREQ = 10
LAYERS = 40
WIDE_LAYERS = 2
DROPRATE = 0.3
NAME = "WideResNet-40-2"
NUM_CLASSES = 22
LOGNORM_TEMP = 1  # hp tuning 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05


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
    # ID models
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

    ID_model_ln, _ = train_loop(
        ID_train_images,
        ID_train_labels,
        ID_val_images,
        ID_val_labels,
        loss="logit-normalization",
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
    eval_fpr95(
        ID_test_labels,
        ID_scores_ce,
        OOD_test_labels,
        OOD_scores_ce,
        "cross_entropy",
    )
    eval_fpr95(
        ID_test_labels,
        ID_scores_ln,
        OOD_test_labels,
        OOD_scores_ln,
        "logit_normalization",
    )

    # AUROC
    eval_auroc(y_OOD, combined_scores_ce, "cross entropy")
    eval_auroc(y_OOD, combined_scores_ln, "logit normalization")

    # AUPR
    eval_aupr(y_OOD, combined_scores_ce, "cross entropy")
    eval_aupr(y_OOD, combined_scores_ln, "logit normalization")

    # Accuracy of the model on the ID test set
    eval_acc(ID_test_labels, ID_preds_ce, "cross entropy")
    eval_acc(ID_test_labels, ID_preds_ln, "logit normalization")


# Run the main function
if __name__ == "__main__":
    main()
