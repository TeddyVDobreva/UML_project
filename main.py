from data_analysis import do_analysis
from preprocessing import preprocess

def main():
    # Preliminaru data analysis
    do_analysis("datasets/sea_creatures")
    do_analysis("datasets/reptiles")

    # Preprocessing
    sea_creatures_split = preprocess("datasets/sea_creatures")
    reptiles_split = preprocess("datasets/reptiles")
    

if __name__ == "__main__":
    main()
