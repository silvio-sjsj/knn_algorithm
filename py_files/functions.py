"""Functions for creating an image folder and saving figures in it and also to load a dataset"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

IMAGES_PATH = Path().resolve().parent / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

image_path = Path().resolve().parent / "images"

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def load_dataset(dataset="penguins"):

    df = sns.load_dataset(dataset)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    return df