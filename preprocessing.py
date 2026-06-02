from pathlib import Path
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt


def _open_image_files(path: str) -> list[Image]:
    """
    Opens the files in a directory
    
    Parameters:
        path (str): the path to a folder with the image data

    Returns:
        list[Image]: list of all the images loaded from the provided path
    """
    p = Path(path)
    images = []
    # find an audio file in the dataset
    for file in p.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".png", ".jpg"}:      
            images.append(Image.open(file))
    if not images:
        raise FileNotFoundError(f"No images found in {p}")
    
    return images

def _plot_image_dims(images):    
    counter = Counter()
    heights = Counter()
    widths = Counter()
    for image in images:
        counter.update([image.size])
        widths.update([image.size[0]])
        heights.update([image.size[1]])

    # plot resolutions
    most_common = counter.most_common(50)
    resolutions = [f"{w[0]}x{w[1]}" for w, _ in most_common]
    counts = [c for _, c in most_common]

    plt.figure()
    plt.bar(resolutions, counts)
    plt.xticks(rotation=45)
    plt.title(f"Top {50} Resolutions in Images")
    plt.savefig("images/im_dimensions.png")
    plt.close()

    # plot width and height histograms
    plt.figure()
    plt.bar(heights.keys(), heights.values())
    plt.xticks(rotation=45)
    plt.title("Histogram of image heights")
    plt.savefig("images/height_hist.png")
    plt.close()

    plt.figure()
    plt.bar(widths.keys(), widths.values())
    plt.xticks(rotation=45)
    plt.title("Histogram of image widths")
    plt.savefig("images/width_hist.png")
    plt.close()

def do_analysis(path):
    images = _open_image_files(path)
    _plot_image_dims(images)