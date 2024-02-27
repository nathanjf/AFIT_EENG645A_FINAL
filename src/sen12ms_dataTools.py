import os
import numpy as np
import matplotlib.pyplot as plt

from sen12ms_dataLoader import SEN12MSDataset, Seasons, LCBands, S1Bands, S2Bands

sen12ms_path = os.path.join(os.path.dirname(__file__), "..", "data")
spring_only_path = os.path.join(os.path.dirname(__file__), "..", "data", "spring_only.txt")

seasons = [Seasons.SPRING, Seasons.SUMMER, Seasons.FALL, Seasons.WINTER]
classes = {
    1   : [230,25,75],      # Evergreen Needleleaf Forests
    2   : [60,180,75],      # Evergreen Broadleaf Forests
    3   : [255,225,25],     # Deciduous Needleleaf Forests
    4   : [0,130,200],      # Deciduous Broadleaf Forests
    5   : [235,130,48],     # Mixed Forests
    6   : [145,30,180],     # Closed (Dense) Shrubland
    7   : [70,240,240],     # Open (Sparse) Shrubland
    8   : [240,50,230],     # Woody Savannas
    9   : [210,245,60],     # Savannas
    10  : [250,190,212],    # Grasslands
    11  : [0,128,128],      # Permanent Wetlands
    12  : [220,190,255],    # Croplands
    13  : [170,110,40],     # Urban and Built-Up Lands
    14  : [255,250,200],    # Croplands/Natural Vegetation Mosaics
    15  : [120,0,0],        # Permanent Snow and Ice
    16  : [170,255,195],    # Barren
    17  : [128,128,0],      # Water Bodies
}
names = {
    1   : 'Evergreen Needleleaf Forests',
    2   : 'Evergreen Broadleaf Forests',
    3   : 'Deciduous Needleleaf Forests',
    4   : 'Deciduous Broadleaf Forests',
    5   : 'Mixed Forests',
    6   : 'Closed (Dense) Shrubland',
    7   : 'Open (Sparse) Shrubland',
    8   : 'Woody Savannas',
    9   : 'Savannas',
    10  : 'Grasslands',
    11  : 'Permanent Wetlands',
    12  : 'Croplands',
    13  : 'Urban and Built-Up Lands',
    14  : 'Croplands/Natural Vegetation Mosaics',
    15  : 'Permanent Snow and Ice',
    16  : 'Barren',
    17  : 'Water Bodies'
}

def sen2_color_image(image : np.array):
    out = np.zeros((image.shape[0],image.shape[1],3))
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            color_idx = 0
            # loop over the bands in rgb order
            for k in [3,2,1]:
                # Scale to the range of [-1 to 1]
                out[i][j][color_idx] = image[i][j][k] / (2**15)
                # TODO : Experiment with shifting by 2**15 and norming by w^16

                color_idx += 1

    return out / np.max(out)

def igbp_color_image(image : np.array):
    out = np.zeros((image.shape[0], image.shape[1], 3))

    # Iterate over the pixels and replace them with the colors from the class dictionary
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            # It seems like there is missing data for some of the images
            if image[i][j] not in classes.keys():
                print("missing pixel")
                out[i][j] = [0, 0, 0]
            else:
                out[i][j] = classes[image[i][j]]
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
        
        # Replace each label with a color
        # subplot all 3 next to eachother
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[10,10])
        
        # ax1 is a color sentinel 2 image
        ax1.imshow(sen2_color_image(x))

        # ax2 is the actual labels
        ax2.imshow(igbp_color_image(y))

        # ax3 is the predicted labels
        ax3.imshow(igbp_color_image(y_pred))


        fig.tight_layout()

        plt.savefig(filename)