from data_analysis import do_analysis
from preprocessing import preprocess
from train import train_loop

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

    # train the models
    model_ce = train_loop(
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
    model_ln = train_loop(
        sea_creatures_split[0],
        sea_creatures_split[3],
        sea_creatures_split[1],
        sea_creatures_split[4],
        "logit-normalization",
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


if __name__ == "__main__":
    main()
