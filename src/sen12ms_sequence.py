
import numpy as np
import tensorflow as tf
import math

from os import path
from sen12ms_dataLoader import SEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def flatten_image(image : np.array):
    out = np.zeros((256,256))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[0]):
            out[x][y] = image[x][y][0]
    return out

def reorder_image(image : np.array):
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    return image

def sentinel_2_image(image : np.array):
    out = np.zeros((256,256,3))
    for i in range(0,256):
        for j in range(0,256):
            color_idx = 0
            for k in [3,2,1]:
                out[i][j][color_idx] = image[i][j][k] # / (2**13)
                color_idx += 1
    return out

def onehot_decode(image : np.array):

    classes = {
        1   : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Evergreen Needleleaf Forests
        2   : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Evergreen Broadleaf Forests
        3   : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Deciduous Needleleaf Forests
        4   : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Deciduous Broadleaf Forests
        5   : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],  # Mixed Forests
        6   : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],  # Closed (Dense) Shrubland
        7   : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],  # Open (Sparse) Shrubland
        8   : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],  # Woody Savannas
        9   : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],  # Savannas
        10  : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],  # Grasslands
        11  : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],  # Permanent Wetlands
        12  : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],  # Croplands
        13  : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],  # Urban and Built-Up Lands
        14  : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],  # Croplands/Natural Vegetation Mosaics
        15  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],  # Permanent Snow and Ice
        16  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],  # Barren
        17  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]   # Water Bodies
    }
    
    # Decode onehot back into numbers
    y_decode = np.zeros(shape=(image.shape[0], image.shape[1]))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[0]):
            
            for key in classes:
                # print(image[x][y].astype(int))
                # print(np.array(classes[key]).astype(int))
                # print(np.equal(image[x][y].astype(int),np.array(classes[key]).astype(int)))
                # print(np.all(np.equal(image[x][y].astype(int),np.array(classes[key]).astype(int))))
                if np.all(np.equal(image[x][y].astype(int),np.array(classes[key]).astype(int))):
                    y_decode[x][y] = key

    return y_decode

def onehot_encode(y : np.array):

    classes = {
        1   : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Evergreen Needleleaf Forests
        2   : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Evergreen Broadleaf Forests
        3   : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Deciduous Needleleaf Forests
        4   : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Deciduous Broadleaf Forests
        5   : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],  # Mixed Forests
        6   : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],  # Closed (Dense) Shrubland
        7   : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],  # Open (Sparse) Shrubland
        8   : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],  # Woody Savannas
        9   : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],  # Savannas
        10  : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],  # Grasslands
        11  : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],  # Permanent Wetlands
        12  : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],  # Croplands
        13  : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],  # Urban and Built-Up Lands
        14  : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],  # Croplands/Natural Vegetation Mosaics
        15  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],  # Permanent Snow and Ice
        16  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],  # Barren
        17  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]   # Water Bodies
    }

    y_onehot = np.zeros(shape=(y.shape[0], y.shape[1], y.shape[2], len(classes)))

    for i in range(0, y.shape[0]):
        for j in range(0,y.shape[1]):
            for k in range(0,y.shape[2]):
                if y[i][j][k][0] == 0:
                    y_onehot[i][j][k] = classes[10]
                else:
                    y_onehot[i][j][k] = classes[y[i][j][k][0]]
                #print(y_onehot[i][j][k])

    return y_onehot

    # Tranform from (x, 256, 256, 1) to x, 256, 256, num_classes

class SEN12MSSequence(tf.keras.utils.Sequence):

    def __init__(self, sen12ms = SEN12MSDataset(path.dirname(__file__)), batch_size=32, data=None):
        self.sen12ms = SEN12MSDataset(path.dirname(__file__))
    
        # Load all the season ids
        spring_ids  = self.sen12ms.get_season_ids(Seasons.SPRING)
        summer_ids  = self.sen12ms.get_season_ids(Seasons.SUMMER)
        fall_ids    = self.sen12ms.get_season_ids(Seasons.FALL)
        winter_ids  = self.sen12ms.get_season_ids(Seasons.WINTER)

        # TODO : Expand to the full id list
        ids_list = [spring_ids]#, summer_ids, fall_ids, winter_ids]

        self.data : np.array = None

        # Transform the ids into a numpy array of parameters
        id_idx = 0
        for ids in ids_list:
            for key in ids.keys():
                for value in ids[key]:
                    row = np.zeros(shape=(1,3))
                    row[0][0] = id_idx
                    row[0][1] = key
                    row[0][2] = value

                    if self.data is None:
                        self.data = row
                    else:
                        self.data = np.vstack([self.data, row])
            id_idx += 1

        # TODO : Count how many of each class there is in the dataset
        # and update a global class count dictionary
        tally = {
            1:0,
            2:0,
            3:0,
            4:0,
            5:0,
            6:0,
            7:0,
            8:0,
            9:0,
            10:0,
            11:0,
            12:0,
            13:0,
            14:0,
            15:0,
            16:0,
            17:0
        }
        counts = {
            1: 35169562, 
            2: 114461610, 
            3: 8516, 
            4: 173018642, 
            5: 91931871, 
            6: 15266338, 
            7: 59174553, 
            8: 177388877, 
            9: 518445158, 
            10: 439273502, 
            11: 31064884, 
            12: 507802578, 
            13: 335295405, 
            14: 42061406, 
            15: 37774, 
            16: 1904431, 
            17: 137003181
        }
        # seasons = [Seasons.SPRING, Seasons.SUMMER, Seasons.FALL, Seasons.WINTER]
        # for i in range(0,self.data.shape[0]):
        #    row = self.data[i][:]
        #    lc, bounds = self.sen12ms.get_patch(seasons[int(row[0])],int(row[1]),int(row[2]),LCBands.IGBP)
        #    lc = lc[0]
        #    for x in range(0, lc.shape[0]):
        #        for y in range(0, lc.shape[1]):
        #            tally[lc[x][y]] += 1

        # print(tally)

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.data.shape[0] / self.batch_size)
    
    def __getitem__(self, idx):
        
        x : np.array = np.zeros(shape=(self.batch_size, 256, 256, 13))
        y : np.array = np.zeros(shape=(self.batch_size, 256, 256, 1))

        start_idx = self.batch_size * idx
        end_idx = min(start_idx + self.batch_size, self.data.shape[0])
        row_idx = 0
        for i in range(start_idx, end_idx):
            
            # Load triplet from index
            row = self.data[i][:]
            seasons = [Seasons.SPRING, Seasons.SUMMER, Seasons.FALL, Seasons.WINTER]
            s1, s2, lc, bounds = self.sen12ms.get_s1s2lc_triplet(
                seasons[int(row[0])], 
                int(row[1]), 
                int(row[2]), 
                s1_bands=S1Bands.ALL,
                s2_bands=S2Bands.ALL, 
                lc_bands=LCBands.IGBP)

            # Format provided by this function is [channel][x][y]
            # I want it to be restructured to [x][y][channel]

            

            #s1 = reorder_image(s1)
            s2 = reorder_image(s2)
            lc = reorder_image(lc)


            row_x = s2  #   np.dstack([s1, s2])
            row_y = lc

            # Append images to x
            #if x is None:
            #    x = row_x
            #else:
            #    x = np.vstack([x, row_x])

            ## Append landcover to label
            #if y is None:
            #    y = row_y
            #else:
            #    y = np.vstack([y, row_y])

            x[row_idx] = row_x / 2**13
            y[row_idx] = row_y
            row_idx += 1

        #print(x.shape, " ", y.shape)

        x = x
        y = onehot_encode(y)

        return x, y
    
    def plot_item(self, idx):

        x, y = self.__getitem__(idx)

        # LABEL HEATMAP
        plt.figure()
        color_list = ((230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0))
        color_list = np.array(color_list).astype(float) / 255.0
        cmap = LinearSegmentedColormap.from_list('IGBP', color_list, 17)
        ax = sns.heatmap(onehot_decode(y[0][:][:]), vmin=1, vmax=17, cmap=cmap)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
        colorbar.set_ticklabels([
            'Evergreen Needleleaf Forests',
            'Evergreen Broadleaf Forests',
            'Deciduous Needleleaf Forests',
            'Deciduous Broadleaf Forests',
            'Mixed Forests',
            'Closed (Dense) Shrubland',
            'Open (Sparse) Shrubland',
            'Woody Savannas',
            'Savannas',
            'Grasslands',
            'Permanent Wetlands',
            'Croplands',
            'Urban and Built-Up Lands',
            'Croplands/Natural Vegetation Mosaics',
            'Permanent Snow and Ice',
            'Barren',
            'Water Bodies'
        ])
        plt.show()

        # TRUE COLOR
        plt.figure()
        plt.imshow(sentinel_2_image(x[0][:][:][:]))
        plt.show()

        plt.figure()
        ax = sns.heatmap(onehot_decode(y[0][:][:]), vmin=1, vmax=17, cmap=cmap, alpha=0.5, zorder=2)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
        colorbar.set_ticklabels([
            'Evergreen Needleleaf Forests',
            'Evergreen Broadleaf Forests',
            'Deciduous Needleleaf Forests',
            'Deciduous Broadleaf Forests',
            'Mixed Forests',
            'Closed (Dense) Shrubland',
            'Open (Sparse) Shrubland',
            'Woody Savannas',
            'Savannas',
            'Grasslands',
            'Permanent Wetlands',
            'Croplands',
            'Urban and Built-Up Lands',
            'Croplands/Natural Vegetation Mosaics',
            'Permanent Snow and Ice',
            'Barren',
            'Water Bodies'
        ])
        ax.imshow(sentinel_2_image(x[0][:][:][:]), aspect=ax.get_aspect(), extent=ax.get_xlim() + ax.get_ylim(), zorder = 1)
        plt.show()

    def plot_predictions(self, x, y, y_pred, idx=0):
        
        # X is image data

        # Y is onehot encoded
        plt.figure()
        color_list = ((230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0))
        color_list = np.array(color_list).astype(float) / 255.0
        cmap = LinearSegmentedColormap.from_list('IGBP', color_list, 17)
        ax = sns.heatmap(onehot_decode(y[idx][:][:]), vmin=1, vmax=17, cmap=cmap)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
        colorbar.set_ticklabels([
            'Evergreen Needleleaf Forests',
            'Evergreen Broadleaf Forests',
            'Deciduous Needleleaf Forests',
            'Deciduous Broadleaf Forests',
            'Mixed Forests',
            'Closed (Dense) Shrubland',
            'Open (Sparse) Shrubland',
            'Woody Savannas',
            'Savannas',
            'Grasslands',
            'Permanent Wetlands',
            'Croplands',
            'Urban and Built-Up Lands',
            'Croplands/Natural Vegetation Mosaics',
            'Permanent Snow and Ice',
            'Barren',
            'Water Bodies'
        ])
        plt.show()

        plt.figure()
        ax = sns.heatmap(flatten_image(y_pred[idx]), vmin=1, vmax=17, cmap=cmap)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
        colorbar.set_ticklabels([
            'Evergreen Needleleaf Forests',
            'Evergreen Broadleaf Forests',
            'Deciduous Needleleaf Forests',
            'Deciduous Broadleaf Forests',
            'Mixed Forests',
            'Closed (Dense) Shrubland',
            'Open (Sparse) Shrubland',
            'Woody Savannas',
            'Savannas',
            'Grasslands',
            'Permanent Wetlands',
            'Croplands',
            'Urban and Built-Up Lands',
            'Croplands/Natural Vegetation Mosaics',
            'Permanent Snow and Ice',
            'Barren',
            'Water Bodies'
        ])
        plt.show()

        # Y pred is not onehot encoded

        
        pass