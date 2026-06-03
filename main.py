import numpy as np

from data_analysis import do_analysis
from evaluation import eval_acc, eval_aupr, eval_auroc, eval_fpr95
from preprocessing import preprocess
from train import get_scores, train_loop

EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
NESTEROV = True
DECAY = 5e-4
PRINT_FREQ = 10
LAYERS = 40
WIDE_LAYERS = 2
DROPRATE = 0
NAME = "WideResNet-40-2"
NUM_CLASSES = 22
LOGNORM_TEMP = 1


def main():
    # Preliminaru data analysis
    do_analysis("datasets/sea_creatures")
    do_analysis("datasets/reptiles")
    print("Analysis done!")

    # Preprocessing
    sea_creatures_split = preprocess("datasets/sea_creatures")
    reptiles_split = preprocess("datasets/reptiles")
    print("Preprocessing done!")

    # Train the models and evaluate on the validation sets
    # ID models
    ID_model_ce = train_loop(
        sea_creatures_split[0],
        sea_creatures_split[3],
        sea_creatures_split[1],
        sea_creatures_split[4],
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

    ID_model_ln = train_loop(
        sea_creatures_split[0],
        sea_creatures_split[3],
        sea_creatures_split[1],
        sea_creatures_split[4],
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

    # OOD models
    OOD_model_ce = train_loop(
        reptiles_split[0],
        reptiles_split[3],
        reptiles_split[1],
        sea_creatures_split[4],
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

    OOD_model_ln = train_loop(
        reptiles_split[0],
        reptiles_split[3],
        reptiles_split[1],
        sea_creatures_split[4],
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

    # Evaluate the models
    # Generate the true labels for the combined ID and OOD test sets
    y_OOD = np.concatenate(
        [np.ones(len(sea_creatures_split[5])), np.zeros(len(reptiles_split[5]))]
    )

    # Get the predicted scores for the ID and OOD test sets
    ID_scores_ce, ID_preds_ce = get_scores(ID_model_ce, sea_creatures_split[2])
    ID_scores_ln, ID_preds_ln = get_scores(ID_model_ln, sea_creatures_split[2])

    OOD_scores_ce, _ = get_scores(OOD_model_ce, reptiles_split[2])
    OOD_scores_ln, _ = get_scores(OOD_model_ln, reptiles_split[2])

    # Combine the scores for the ID and OOD test sets
    combined_scores_ce = np.concatenate([ID_scores_ce, OOD_scores_ce])
    combined_scores_ln = np.concatenate([ID_scores_ln, OOD_scores_ln])

    # FPR95
    eval_fpr95(
        sea_creatures_split[5],
        ID_scores_ce,
        "cross_entropy",
        reptiles_split[5],
        OOD_scores_ce,
        "cross_entropy",
    )
    eval_fpr95(
        sea_creatures_split[5],
        ID_scores_ln,
        "logit_normalization",
        reptiles_split[5],
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
    eval_acc(sea_creatures_split[5], ID_preds_ce, "cross entropy")
    eval_acc(sea_creatures_split[5], ID_preds_ln, "logit normalization")


# Run the main function
if __name__ == "__main__":
    main()
