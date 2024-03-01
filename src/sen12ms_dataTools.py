import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import main_config as cfg
from sen12ms_dataLoader import SEN12MSDataset, Seasons, LCBands, S1Bands, S2Bands

sen12ms_path = os.path.join(os.path.dirname(__file__), "..", "data")
spring_only_path = os.path.join(os.path.dirname(__file__), "..", "data", "spring_only.txt")

def sen2_color_image(image : np.array):
    out = np.zeros((image.shape[0],image.shape[1],3))
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            color_idx = 0
            # loop over the bands in rgb order
            for k in [3,2,1]:
                out[i][j][color_idx] = image[i][j][k]
                color_idx += 1
    return out / np.max(out)

def igbp_color_image(image : np.array):
    colors = None
    match cfg.MODE:
        case cfg.RunMode.SIMPLE:
            colors = cfg.simple_colors        
        case cfg.RunMode.COMPLEX:
            colors = cfg.colors

    out = np.zeros((image.shape[0], image.shape[1], 3))

    # Iterate over the pixels and replace them with the colors from the class dictionary
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
                out[i][j] = colors[image[i][j]]
    return out.astype(int)

# Load a list of all the available files and return it
class SEN12MSDataTools():
    def __init__(self):
        self.sen12ms = SEN12MSDataset(sen12ms_path)
    
        # Load all the season ids
        if not os.path.exists(spring_only_path):
            spring_ids  = self.sen12ms.get_season_ids(Seasons.SPRING)
            summer_ids  = self.sen12ms.get_season_ids(Seasons.SUMMER)
            fall_ids    = self.sen12ms.get_season_ids(Seasons.FALL)
            winter_ids  = self.sen12ms.get_season_ids(Seasons.WINTER)
            ids_list = [spring_ids, summer_ids, fall_ids, winter_ids]
        else:
            spring_ids  = self.sen12ms.get_season_ids(Seasons.SPRING)
            ids_list = [spring_ids]

        # Calculate total length of the dataset
        dataset_length = 0
        for ids in ids_list:
            for key in ids.keys():
                dataset_length += len(ids[key])

        # Initialize to the correct size to avoid memory allocation slowdowns
        self.data : np.array = np.zeros((dataset_length, 3))

        # Transform the ids into a numpy array of parameters
        season_idx = 0
        dataset_idx = 0
        for ids in ids_list:
            for scene in ids.keys():
                for patch in ids[scene]:
                    
                    self.data[dataset_idx][0] = season_idx  # Season
                    self.data[dataset_idx][1] = scene       # Scene id
                    self.data[dataset_idx][2] = patch       # Patch id

                    # Next entry in the total dataset
                    dataset_idx += 1
            # Next season in the ids_list
            season_idx += 1

    def get_data(self):
        return self.data

    def plot_prediction(x, y, y_pred, filename):
        colors = None
        names = None
        match cfg.MODE:
            case cfg.RunMode.SIMPLE:
                colors = cfg.simple_colors        
                names = cfg.simple_names
            case cfg.RunMode.COMPLEX:
                colors = cfg.colors
                names = cfg.names

        # Replace each label with a color
        # subplot all 3 next to eachother
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=[15,5])
        
        # ax1 is a color sentinel 2 image
        ax1.imshow(sen2_color_image(x))
        ax1.set_title("RGB Image")

        # ax2 is the actual labels
        ax2.imshow(igbp_color_image(y))
        ax2.set_title("True IGBP Labels")

        # ax3 is the predicted labels
        ax3.imshow(igbp_color_image(y_pred))
        ax3.set_title("Predicted IGBP Labels")

        # Plot the legend for the labels
        # Pack labels and colors into handles array
        handles = []
        for key in colors.keys():
            color = (np.array(colors[key]) / 255.0)
            name = names[key]
            handles.append(Patch(color=color, label=name))
        ax4.legend(handles=handles, loc='center', frameon=False)
        ax4.axis('off')

        fig.tight_layout()

        plt.savefig(filename)
        plt.close()