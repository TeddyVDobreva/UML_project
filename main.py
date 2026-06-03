from data_analysis import do_analysis
from preprocessing import preprocess
from train import train_loop

def main():
    # Preliminaru data analysis
    # do_analysis("datasets/sea_creatures")
    # do_analysis("datasets/reptiles")
    # print("Analysis done!")
    # Preprocessing
    sea_creatures_split = preprocess("datasets/sea_creatures")
    # reptiles_split = preprocess("datasets/reptiles")
    print("Preprocessing done!")

    train_loop(sea_creatures_split[0], sea_creatures_split[3], sea_creatures_split[1], sea_creatures_split[4])
    

if __name__ == "__main__":
    main()
